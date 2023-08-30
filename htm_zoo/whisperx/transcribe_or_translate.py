"""Given language detection output (EN -- transcribe mode, or nonEN -- translate mode),
Use WhisperX to get ASR with timestamps. """

import os
import whisperx
import torch
import glob
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import default_collate
import copy

try:
    os.system('module load apps/ffmpeg-4.2.1')
except:
    print('failed to load ffmpeg from module')
    out = os.system('which ffmpeg')
    print(f'using ffmpeg from: {out}')


device = "cuda" 
audio_file = '/scratch/shared/beegfs/htd/DATA/HowTo100M/audio/video_FE_part1/6xlbuJzlZgE.m4a'
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print("warmup finishes")

class AudioDataset():
    def __init__(self, source):
        if isinstance(source, str):
            self.audio_list = sorted(glob.glob(os.path.join(root, '*')))
            print(f'construct dataset from {source} with {len(self.audio_list)} files')
        elif isinstance(source, list) and isinstance(source[0], str):
            self.audio_list = sorted(source)
            print(f'construct dataset from the provided list with {len(self.audio_list)} files')
            tmp = []
            non_existed_list = []  # for debugging
            for fp in tqdm(self.audio_list, desc='check existence'):
                if os.path.exists(fp):
                    tmp.append(fp)
                else:
                    non_existed_list.append(fp)
            self.audio_list = tmp
    
    def __getitem__(self, index):
        audio_path = self.audio_list[index]
        audio = whisperx.load_audio(audio_path)
        out_dict = {'audio':audio, 'path':audio_path}
        return out_dict
    
    def collate_fn(self, batch):
        out_dict = {}
        out_dict['audio'] = [i['audio'] for i in batch]  # keep numpy format
        out_dict['path'] = default_collate([i['path'] for i in batch])
        return out_dict
    
    def __len__(self):
        return len(self.audio_list)


writer_args = {"highlight_words": False, 
               "max_line_count": None, 
               "max_line_width": None}


@torch.no_grad()
def main(root, output_dir):
    """transcribe mode, for EN only. """
    partname = os.path.basename(root)
    output_path = os.path.join(output_dir, partname)
    os.makedirs(output_path, exist_ok=True)
    writer = whisperx.utils.get_writer('json', output_path)

    language_code = 'en'
    lang_detect_path = f'language_output/{partname}.csv'
    lang_detect_df = pd.read_csv(lang_detect_path)
    todo_path = lang_detect_df[lang_detect_df['language']==language_code]['filename'].tolist()
    total_num = len(todo_path)

    drop_list = glob.glob(os.path.join(output_path, '*.json'))
    drop_list = set([os.path.basename(i).split('.')[0] for i in drop_list])
    todo_path = [i for i in tqdm(todo_path, desc='remove existed files') if os.path.basename(i).split('.')[0] not in drop_list]
    print(f"TODO files: {total_num} --> {len(todo_path)}")

    # prepare Align model
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)

    print(f"working on {root}")
    D = AudioDataset(todo_path)
    loader = DataLoader(D, batch_size=1, num_workers=8, shuffle=False, collate_fn=D.collate_fn, pin_memory=True)

    for idx, data in tqdm(enumerate(loader), total=len(loader), desc=partname):
        audio, filenames = data['audio'], data['path']
        B = len(filenames)
        assert B == 1
        audio = audio[0]
        audio_path = filenames[0]

        result = model.transcribe(audio, batch_size=batch_size, language=language_code)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        writer(result, audio_path, writer_args)

    print(root, 'finishes')


@torch.no_grad()
def batch_translate(model, tokenizer, sentence_list, batch_size=4):
    sentence_chunks = np.array_split(sentence_list, len(sentence_list)//batch_size+1)
    output = []
    for sentence_batch in sentence_chunks:
        encoded_text = tokenizer(sentence_batch.tolist(), return_tensors='pt', padding=True)
        encoded_text = {k:v.cuda() for k,v in encoded_text.items()}
        generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id("en"))
        output_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        output.extend(output_batch)
    return output


@torch.no_grad()
def main_translate(root, output_dir):
    """translate mode, for non-EN.
    We provide: 
      * ASR in original language (with word-level timestamps if phoneme model is available),
      * ASR in English, with sentence-wise timestamps. Translated by "facebook/m2m100_418M"
    """

    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").cuda()

    partname = os.path.basename(root)
    output_path = os.path.join(output_dir, partname)
    os.makedirs(output_path, exist_ok=True)
    writer = whisperx.utils.get_writer('json', output_path)
    drop_list = glob.glob(os.path.join(output_path, '*.json'))
    drop_list = set([os.path.basename(i).split('.')[0] for i in drop_list])

    lang_detect_path = f'language_output/{partname}.csv'
    lang_detect_df = pd.read_csv(lang_detect_path)
    todo_df = lang_detect_df[lang_detect_df['language']!='en']
    other_language_code = todo_df['language'].unique().tolist()
    
    print(f"working on {root}")

    for language_code in tqdm(other_language_code):
        print(f"working on language {language_code}")
        todo_path = lang_detect_df[lang_detect_df['language']==language_code]['filename'].tolist()
        total_num = len(todo_path)

        todo_path = [i for i in tqdm(todo_path, desc='remove existed files') if os.path.basename(i).split('.')[0] not in drop_list]
        print(f"TODO files: {total_num} --> {len(todo_path)}")
        if len(todo_path) == 0:
            continue

        DO_ALIGN = True
        # prepare Align model
        try:
            model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
            translation_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang=language_code)
        except:
            print('fall back to plain Whisper')
            DO_ALIGN = False

        D = AudioDataset(todo_path)
        loader = DataLoader(D, batch_size=1, num_workers=8, shuffle=False, collate_fn=D.collate_fn, pin_memory=True)

        for idx, data in tqdm(enumerate(loader), total=len(loader), desc=partname):
            audio, filenames = data['audio'], data['path']
            B = len(filenames)
            assert B == 1
            audio = audio[0]
            audio_path = filenames[0]

            if DO_ALIGN:
                # whisperX alignment, then translate to EN with other toolbox
                result = model.transcribe(audio, batch_size=batch_size, language=language_code, task='transcribe')
                result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
                aligned_sentences = [i['text'].strip() for i in result['segments']]
                if len(aligned_sentences) == 0:
                    print(f'empty sentence list at {audio_path} with language {language_code}, continue')
                    continue
                translated_sentences = batch_translate(translation_model, translation_tokenizer, aligned_sentences)
                result['language'] = language_code
                result['segments_native'] = copy.deepcopy(result['segments'])
                segment_translate = copy.deepcopy(result['segments'])
                assert len(segment_translate) == len(translated_sentences)
                for seg_idx in range(len(translated_sentences)):
                    segment_translate[seg_idx]['text'] = translated_sentences[seg_idx]
                    del segment_translate[seg_idx]['words']
                result['segments'] = segment_translate
            else:
                # rely on whisper's end-to-end translation, timestamp might be inaccurate
                result = model.transcribe(audio, batch_size=batch_size, language=language_code, task='translate')

            writer(result, audio_path, writer_args)

    print(root, 'finishes')



if __name__ == '__main__':
    output_dir = '/scratch/shared/beegfs/htd/whisper/whisperX_large_v2'
    # e.g. root = '/scratch/shared/beegfs/htd/DATA/HowTo100M/audio/video_FE_part1'
    rootlist = sorted(glob.glob('/scratch/shared/beegfs/htd/DATA/HowTo100M/audio/video_FE_*'))

    for root in tqdm(rootlist):
        main(root, output_dir)  # for transcribe mode
        main_translate(root, output_dir)  # for translate mode
