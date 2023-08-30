"""use WhisperX to detect language only, save in csv"""

import os
import glob
from tqdm import tqdm
import numpy as np
import whisperx
import torch
from torch.utils.data import DataLoader
import pandas as pd


device = "cuda" 
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# hard-coded audio hyperparameters from WhisperX
SAMPLE_RATE = 16000
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk

# warmup, otherwise it returns error in batch processing
audio = whisperx.load_audio('/scratch/shared/beegfs/htd/DATA/HowTo100M/audio/video_FE_part1/6xlbuJzlZgE.m4a')
audio = audio[: N_SAMPLES]
_ = model.transcribe(audio)
print("warmup finishes")
print("=="*10)


class AudioDataset():
    def __init__(self, root):
        self.audio_list = sorted(glob.glob(os.path.join(root, '*')))
        print(f'construct dataset from {root} with {len(self.audio_list)} files')
    
    def __getitem__(self, index):
        audio_path = self.audio_list[index]
        audio = whisperx.load_audio(audio_path)
        audio = audio[: N_SAMPLES]
        if audio.shape[0] < N_SAMPLES:
            audio = np.concatenate((audio, np.zeros(N_SAMPLES-audio.shape[0])))
        logmel = whisperx.audio.log_mel_spectrogram(audio.astype(np.float32))
        return logmel, audio_path
    
    def __len__(self):
        return len(self.audio_list)



def main(root):
    partname = os.path.basename(root)
    output_name = f'language_output/{partname}.csv'
    if os.path.exists(output_name):
        print(f'{output_name} exists, skipping ...')
        return

    D = AudioDataset(root)
    loader = DataLoader(D, batch_size=32, num_workers=16, shuffle=False)
    all_results = []

    for idx, (logmel, filenames) in tqdm(enumerate(loader), total=len(loader)):
        B = logmel.shape[0]
        segment = logmel
        encoder_output = model.model.encode(segment)
        results = model.model.model.detect_language(encoder_output)
        assert len(results) == B
        result_tuple = [res[0] for res in results]
        result = [[fn, i[0][2:-2], i[1]] for i, fn in zip(result_tuple, filenames)]
        all_results.extend(result)

    df = pd.DataFrame.from_records(all_results, columns=['filename', 'language', 'prob'])
    df.to_csv(output_name, index=False)
    print(root, 'finishes')


if __name__ == '__main__':
    # e.g. root = '/scratch/shared/beegfs/htd/DATA/HowTo100M/audio/video_FE_part1'
    rootlist = sorted(glob.glob('/scratch/shared/beegfs/htd/DATA/HowTo100M/audio/video_*'))
    for root in tqdm(rootlist):
        main(root)