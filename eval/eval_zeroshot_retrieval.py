import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('../')
import matplotlib.pyplot as plt


def compute_metrics(x):
    """From https://github.com/antoine77340/MIL-NCE_HowTo100M/blob/master/metrics.py """
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1
    return {k:np.array(v) for k,v in metrics.items()}


class YouCook2_Feature():
    def __init__(self,
                 mode='val',
                 source='feature_inria_milnce_16fps_center',
                 num_clips=4,
                 seq_len=32,):
        mode_ = 'val' if mode == 'test' else mode
        self.mode = mode 
        self.num_clips = num_clips
        self.seq_len = seq_len

        anno_path = '/scratch/shared/beegfs/htd/DATA/YouCook2'
        with open(os.path.join(anno_path, f'feat_csv_vid2path_{mode}.json')) as fp:
            self.vid2path = json.load(fp)

        # get split 
        self.video_feature_path = os.path.join(anno_path, '{}/{}'.format(source, mode_))

        vlen_path = os.path.join(anno_path, 'splits/{}_duration_totalframe.csv'.format(mode_))
        vlen_df = pd.read_csv(vlen_path)
        vlen_dict = {vid:[duration,num_frame] \
                     for vid,duration,num_frame in zip(vlen_df['vid_id'], 
                                                       vlen_df['duration'], 
                                                       vlen_df['total_frame'])}
        self.vlen_dict = vlen_dict
        video_info = sorted(list(self.vlen_dict.keys()))

        # check existence
        video_info = [i for i in video_info \
                      if os.path.exists(
                          os.path.join(anno_path, source, mode_, 
                              "{}_{}.pth.tar".format(self.vid2path[i].split('/')[-2], i)))]

        # drop certain features because problematic video length
        drop_list = ['FtHLUsOntqI', 'HQtOXHghaL0', 'ffoRmenLSLs', 'wKHC2gbRdA0']
        video_info = [i for i in video_info if i not in drop_list]

        with open(os.path.join(anno_path, 'youcookii_annotations_trainval.json')) as fp:
            anno = json.load(fp)
        self.anno = anno['database']

        video_info_clip = []
        for vid in video_info:
            for seg in self.anno[vid]['annotations']:
                video_info_clip.append({'vid':vid, **seg})

        self.video_info = video_info_clip
        print('YouCook2 {} feature loader get {} video clips from {}'\
              .format(mode, len(self.video_info), self.video_feature_path))

    def __len__(self):
        return len(self.video_info)

    def __getitem__(self, idx):
        info = self.video_info[idx]
        vid = info['vid']
        text = info['sentence']
        text_start, text_end = info['segment']

        if self.seq_len == -1:
            video, start_idx, end_idx = self._get_video_feature(vid, text_start, text_end, self.num_clips)
            return {'video': video,
                    'start': text_start,
                    'end': text_end,
                    'vid':vid,
                    'str':text,
                    'start_idx': start_idx,
                    'end_idx': end_idx}
        else:
            video = self._get_video_feature(vid, text_start, text_end, self.num_clips)
            return {'video': video,
                    'start': text_start,
                    'end': text_end,
                    'vid':vid,
                    'str':text,}

    def _get_video_feature(self, vid, start, end, num_clips=4):
        path = os.path.join(self.video_feature_path, "{}_{}.pth.tar".format(self.vid2path[vid].split('/')[-2], vid))
        feature = torch.load(path, map_location='cpu')
        assert abs(feature.size(0) - self.vlen_dict[vid][0]) <= 2

        vlen = feature.size(0) 
        trim_start = start
        trim_end = end

        if self.seq_len == -1:
            duration = np.floor(trim_end - trim_start).astype(int)
            chosen_vlen = np.clip(duration * 2, a_min=32, a_max=256)

            if chosen_vlen >= duration:
                # take longer windows covering the targeted segment
                chosen_lead = np.floor(np.linspace(0.25 *(chosen_vlen - duration), 0.75*(chosen_vlen - duration), num_clips)).astype(int)
                chosen_start = trim_start - chosen_lead
            else:
                # take windows inside the targeted segment
                chosen_lag = np.floor(np.linspace(0.25 *(duration - chosen_vlen), 0.75*(duration - chosen_vlen), num_clips)).astype(int)
                chosen_start = trim_start + chosen_lag

            seq_idx = np.arange(chosen_vlen).astype(int)

            chosen_frame_idx = np.expand_dims(chosen_start, 1) + np.expand_dims(seq_idx, 0)
            chosen_frame_idx = chosen_frame_idx.flatten('C')
            chosen_frame_idx = np.clip(chosen_frame_idx, a_min=0, a_max=vlen-1)

            chosen_feature = feature[torch.as_tensor(chosen_frame_idx), :]
            chosen_feature = chosen_feature.view(num_clips, chosen_vlen, -1)

            if chosen_vlen >= duration:
                return chosen_feature, chosen_lead, chosen_lead + duration
            else:
                return chosen_feature, np.zeros_like(chosen_lag), np.zeros_like(chosen_lag) + chosen_vlen

        else:
            chosen_start = np.floor(np.linspace(0, trim_end - trim_start - self.seq_len - 1, num_clips)).astype(int)
            chosen_start = chosen_start + trim_start
            seq_idx = np.arange(self.seq_len).astype(int)

            chosen_frame_idx = np.expand_dims(chosen_start, 1) + np.expand_dims(seq_idx, 0)
            chosen_frame_idx = chosen_frame_idx.flatten('C')
            chosen_frame_idx = np.clip(chosen_frame_idx, a_min=0, a_max=vlen-1)

            chosen_feature = feature[torch.as_tensor(chosen_frame_idx), :]
            chosen_feature = chosen_feature.view(num_clips, self.seq_len, -1)

            return chosen_feature


@torch.no_grad()
def test_retrieval_yc2(lang_model, get_visual_feature, get_text_feature, device, args):
    feature_source = 'feature_inria_milnce_16fps_center'
    D = YouCook2_Feature(mode='val',
            num_clips=10,  # taken from MIL-NCE
            seq_len=-1,
            source=feature_source,)
    data_loader = DataLoader(D, batch_size=1, num_workers=args.num_workers)

    all_visual_feature = []
    all_text_feature = []
    all_text_string = []
    all_info = []
    all_duration = []

    tokenizer = args.tokenizer
    seq_len = args.seq_len

    for input_data in tqdm(data_loader, total=len(data_loader)):
        video = input_data['video'][0].to(device)
        text_str = input_data['str'][0]
        vid = input_data['vid'][0]
        duration = input_data['end'][0] - input_data['start'][0]

        visual_feature = get_visual_feature(
            video, 
            torch.zeros(video.shape[0:2], device=video.device, dtype=torch.bool),
            interpolate_from=seq_len if video.shape[1] >= seq_len else None,  # args.seq_len
            )

        if visual_feature.dim() == 4:
            visual_feature = visual_feature[:,-1,:]

        token = tokenizer([text_str], return_tensors='pt', padding=True)
        token = {k:v.to(device) for k,v in token.items()}
        lang_embed = lang_model(**token)
        lang_embed = lang_embed['pooler_output']
        text_feature = get_text_feature(lang_embed)

        if 'start_idx' in input_data:
            buff = []
            for i in range(visual_feature.size(0)):
                buff.append(visual_feature[i, input_data['start_idx'][0,i].item(): input_data['end_idx'][0,i].item(), :])
            visual_feature = torch.stack(buff, 0)
            
            if args.sim == 'cos':  # norm (first) before avg
                visual_feature = visual_feature / visual_feature.norm(dim=-1, keepdim=True)
            visual_feature = visual_feature.mean(0).mean(0, keepdim=True) # avg across time and num_clips
        else:
            raise NotImplementedError

        if args.sim == 'cos':
            visual_feature = visual_feature / visual_feature.norm(dim=-1, keepdim=True)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

        all_visual_feature.append(visual_feature)
        all_text_feature.append(text_feature)
        all_text_string.append(text_str)
        all_info.append(vid)
        all_duration.append(duration)

    all_visual_feature = torch.cat(all_visual_feature, dim=0)
    all_text_feature = torch.cat(all_text_feature, dim=0)

    save_dict = {'visual': all_visual_feature.tolist(),
                 'text': all_text_feature.tolist(),
                 'str': all_text_string,
                 'info': all_info}
    # Optional: save features somewhere if needed
    
    all_visual_feature = all_visual_feature.cpu().numpy()
    all_text_feature = all_text_feature.cpu().numpy()
    sim = np.dot(all_text_feature, all_visual_feature.T)
    metrics = compute_metrics(sim)
    print(metrics)

    # try centering the feature (mean=0)
    all_visual_feature_ = all_visual_feature - all_visual_feature.mean(0, keepdims=True)
    all_text_feature_ = all_text_feature - all_text_feature.mean(0, keepdims=True)
    print('after centering:')
    metrics_center = compute_metrics(np.dot(all_text_feature_, all_visual_feature_.T))
    print(metrics_center)

    # try standarizing the feature (mean=0, std=1)
    all_visual_feature_standard = all_visual_feature_ / all_visual_feature_.std(0, keepdims=True)
    all_text_feature_standard = all_text_feature_ / all_text_feature_.std(0, keepdims=True)
    print('after standardize:')
    metrics_standard = compute_metrics(np.dot(all_text_feature_standard, all_visual_feature_standard.T))
    print(metrics_standard)

    metrics['C-R1'] = metrics_center['R1']
    metrics['C-R5'] = metrics_center['R5']
    metrics['C-R10'] = metrics_center['R10']
    metrics['C-MR'] = metrics_center['MR']

    metrics['S-R1'] = metrics_standard['R1']
    metrics['S-R5'] = metrics_standard['R5']
    metrics['S-R10'] = metrics_standard['R10']
    metrics['S-MR'] = metrics_standard['MR']

    return metrics


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    # S3D_PATH should point to https://github.com/antoine77340/MIL-NCE_HowTo100M/ folder 
    S3D_PATH = '/work/htd/Desktop_tmp/VideoMetricLearning/process_data/feature_milnce/'
    sys.path.append(S3D_PATH)
    import s3dg as milnce
    def get_word2vec_pre_projection():
        model = milnce.S3D(os.path.join(S3D_PATH, 's3d_dict.npy'), 512)
        model.load_state_dict(
            torch.load(os.path.join(S3D_PATH, 's3d_howto100m.pth')))
        return model.fc

    sys.path.append('../model/')
    from word2vec_model import Word2VecTokenizer, Word2VecModel

    class DummyArgs():
        def __init__(self):
            self.tokenizer = Word2VecTokenizer()
            self.num_workers = 4
            self.model = 'align'
            self.sim = 'cos'
            self.sentence_mode = 'cls'
            self.num_encoder_layers = 0
            self.seq_len = 64
            self.use_alignability_head = False
            self.test = False

    # test MILNCE raw features, with dot product
    args = DummyArgs()
    device = torch.device('cuda')
    lang_model = Word2VecModel()
    lang_model.to(device)
    lang_model.eval()
    milnce_fc = get_word2vec_pre_projection()
    milnce_fc.to(device)
    milnce_fc.eval()

    get_visual_feature = lambda x, *args, **kwargs: milnce_fc(x)
    get_text_feature = lambda x: x
    metrics = test_retrieval_yc2(lang_model, get_visual_feature, get_text_feature, device, args)

    sys.exit(0)