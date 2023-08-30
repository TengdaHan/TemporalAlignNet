"""For future reference: this is our feature extraction script for HowTo100M
by Tengda Han, Aug 30, 2023."""

import os
import numpy as np
import torch
import torch.nn as nn
import random
from tqdm import tqdm
import json 
import argparse 
import ffmpeg  # https://github.com/kkroening/ffmpeg-python
from torch.utils.data import Dataset
import pandas as pd
from einops import rearrange
import sys
import hashlib
from joblib import Parallel, delayed
import glob
import gc
import clip

if os.environ['LOGNAME'] == 'htd':  # local
    sys.path.insert(0, os.path.expanduser('~/work/Desktop_tmp/Align2/InternVideo/Pretrain/Multi-Modalities-Pretraining'))
else:  # AWS
    sys.path.insert(0, os.path.expanduser('~/Align2/InternVideo/Pretrain/Multi-Modalities-Pretraining'))
import InternVideo
from InternVideo.clip_utils.utils.attention import MultiheadAttention


try:
    os.system('module load apps/ffmpeg-4.2.1')
except:
    print('failed to load ffmpeg from module')
    out = os.system('which ffmpeg')
    print(f'using ffmpeg from: {out}')


device = "cuda"
if os.environ['LOGNAME'] == 'htd':  # local
    PREFIX = os.path.expanduser("/scratch/shared/beegfs/shared-datasets/HowTo100M")
else:
    PREFIX = os.path.expanduser("~/s3mount")


def check_existence(video_list, video_root, vid_to_path_dict, tmpdir='tmp'):
    os.makedirs(os.path.join(os.path.dirname(__file__), tmpdir), exist_ok=True)
    hash_tag = hashlib.sha256((video_root+json.dumps(video_list)).encode()).hexdigest()
    hash_file = os.path.join(os.path.dirname(__file__), tmpdir, f'{hash_tag}.check_existence.json')
    if os.path.exists(hash_file):
        print(f'load from TMP file: {hash_file}')
        existed_video = json.load(open(hash_file))
    else:
        check_fn = lambda x: os.path.exists(os.path.join(video_root, vid_to_path_dict.get(x, x)))
        result = Parallel(n_jobs=8, prefer='threads')(delayed(check_fn)(i) for i in tqdm(
            video_list, total=len(video_list), desc="Check Existence", leave=False,
            disable=('/srun' in os.environ['_'])))
        existed_video = []
        for res, vid in zip(result, video_list):
            if res:
                existed_video.append(vid)
        with open(hash_file, 'w') as fobj:
            json.dump(existed_video, fobj)
    return existed_video


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    

def replace_layer(module, target_class, new_class):
    """designed for nn.LayerNorm"""
    for name, child in module.named_children():
        if isinstance(child, target_class):
            original_state_dict = child.state_dict()
            original_device = child.weight.device
            setattr(module, name, new_class(child.normalized_shape, child.eps, child.elementwise_affine).to(original_device))
            new_attr = getattr(module, name)
            new_attr.load_state_dict(original_state_dict)
        else:
            replace_layer(child, target_class, new_class)


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention) or isinstance(l, MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def get_vid_to_path_dict():
    """output: dict: vid --> video_path"""
    with open('all_vid_to_path.json') as fobj:
        content = json.load(fobj)
    import ipdb; ipdb.set_trace()
    return content


def get_htm_vlen_df():
    htm_vlen_df = pd.read_csv(
        os.path.join('htm_vlen.csv'),
        names=['vid','vlen']
    )
    return htm_vlen_df



class HTM_LongLoader(Dataset):
    def __init__(self,
                 vid_list,
                 fps=8,
                 center_crop_only=True,
                 use_internvideo_backend=False,
                 return_half=True):
        """A dataloader for a list of videos provided in vid_list"""
        self.center_crop_only = center_crop_only
        self.video_root = PREFIX
        self.fps = fps

        # internvideo_backend uses decord, 
        # which provides similar results as ffmpeg-python, but slightly slower
        assert not use_internvideo_backend
        self.use_internvideo_backend = use_internvideo_backend

        self.return_half = return_half
        self.htm_vlen_df = get_htm_vlen_df()

        # Optional: check existence
        # self.vid_to_path_dict = get_vid_to_path_dict()
        # vid_with_video = check_existence(tuple(vid_list), self.video_root, self.vid_to_path_dict)
        vid_with_video = vid_list
        print(f'only have {len(vid_with_video)} videos existed in {self.video_root}')
        self.video_info = vid_with_video


    def __len__(self):
        return len(self.video_info)

    def get_vlen_float(self, path):
        try:
            probe = ffmpeg.probe(path)
        except Exception as e:
            print(f"ffprobe error on: {path}")
            raise e
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream:
            # Get the duration of the video in seconds
            try:
                if 'duration' in video_stream:
                    duration = float(video_stream['duration'])
                else:
                    duration = float(probe['format']['duration'])
            except Exception as e:
                print(f"{path} failed to get duration from ffprobe: {video_stream}")
                raise e
            return duration
        else:
            print(f"No video stream found in the file {path}")
            return None

    def __getitem__(self, index):
        video_path = self.video_info[index]
        vid = os.path.basename(video_path).split('.')[0]
        # vlen = self.htm_vlen_df[self.htm_vlen_df['vid'] == vid]['vlen'].values[0]  # BUG: some videos not in vlen_df because vlen_df is from MIL-NCE features
        vlen = int(self.get_vlen_float(video_path))
        if vlen == 0:
            print(f'Empty video detected: {video_path} has length {vlen}')
            raise ValueError
        if vlen > 10 * 60:
            print(f"Long video detected: {video_path} has length {vlen}")
        video, s, e = self._get_video_frame(video_path, vlen)
        return video, vid

    def _get_video_frame(self, video_path, vlen):
        assert os.path.exists(video_path)
        total_num_frames = vlen * self.fps
        start = 0
        end = vlen

        cmd = (ffmpeg.input(video_path, ss=start, t=end-start)
                     .filter('fps', fps=self.fps))
        if self.center_crop_only:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0,1), random.uniform(0,1)
        # cmd = (cmd.crop('(iw - {})*{}'.format(224, aw),
        #                 '(ih - {})*{}'.format(224, ah),
        #                  str(224), str(224)))
        cmd = (cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                        '(ih - min(iw,ih))*{}'.format(ah),
                        'min(iw,ih)',
                        'min(iw,ih)').filter('scale', 224, 224))
        try:
            out, _ = (cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                        .run(capture_stdout=True, quiet=True))
            frames = np.frombuffer(out, np.uint8).reshape([-1, 224, 224, 3]) / 255.
            frames = torch.from_numpy(frames).float()
            del out 
        except:
            print(f'failed to load video: {video_path}, replace with zeros')
            frames = torch.ones(total_num_frames, 224, 224, 3, dtype=torch.float) * 0.5

        num_frames = frames.size(0)
        if num_frames < total_num_frames:
            print(f"WARNING: {video_path} get {num_frames} features but has {total_num_frames} seconds duration")
            zeros = torch.zeros((total_num_frames - num_frames, 224, 224, 3), dtype=torch.float)
            frames = torch.cat((frames, zeros), axis=0)
        frames = frames[0:int(total_num_frames), :]  # T,H,W,C
        frames = rearrange(frames, ' t h w c -> t c h w')
        if self.return_half:
            frames = frames.half()
        return frames, start, end



@torch.no_grad()
def extract_feature(loader, model, device, epoch, args):
    model.eval()
    if args.half:
        replace_layer(model, nn.LayerNorm, LayerNorm)
        convert_weights(model)
    
    # mean/std copied from InternVideo and CLIP repo
    internvideo_mean = torch.FloatTensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    internvideo_std = torch.FloatTensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    clip_mean = torch.FloatTensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    clip_std = torch.FloatTensor([0.26862954, 0.26130258, 0.27577711]).to(device)

    if args.half:
        internvideo_mean = internvideo_mean.half()
        internvideo_std = internvideo_std.half()

    for idx, input_data in tqdm(enumerate(loader), total=len(loader)):
        video_seq_ = input_data[0].to(device, non_blocking=True)
        vid = input_data[1][0]
        B,T,C,H,W = video_seq_.shape
        assert B == 1
        video_seq_ = video_seq_[0]
        video_seq_list = torch.split(video_seq_, split_size_or_sections=int(args.fps*args.batch_size), dim=0)

        all_v_features = []
        for video_seq in tqdm(video_seq_list, total=len(video_seq_list), leave=False):
            if args.model.startswith('clip'):  # designed for fps==1
                video_seq = (video_seq - clip_mean[None,:,None,None]).div(clip_std[None,:,None,None])
                v_features = model.encode_image(video_seq)
            elif args.model == 'timesformer':  # designed for fps==8
                video_seq = rearrange(video_seq, '(b t) c h w -> b c t h w', t=8)
                video_seq = (video_seq - model.pixel_mean[None,:,None,None,None]).div(model.pixel_std[None,:,None,None,None])
                v_features = model.timesformer(video_seq)  # this extracts backbone output feature
            elif args.model.startswith('internvideo'): # designed for fps==8
                if loader.dataset.use_internvideo_backend:
                    video_seq = rearrange(video_seq, '(b t) c h w -> b c t h w', t=8)
                    v_features = model.encode_video(video_seq)
                else: # need norm
                    video_seq = (video_seq - internvideo_mean[None,:,None,None]) / internvideo_std[None,:,None,None]
                    video_seq = rearrange(video_seq, '(b t) c h w -> b c t h w', t=8)
                    if args.half:
                        video_seq = video_seq.half()
                    v_features = model.encode_video(video_seq)
            all_v_features.append(v_features.cpu())
        all_v_features = torch.cat(all_v_features, dim=0)
        torch.save(all_v_features.cpu(), os.path.join(args.output_dir, args.partname, f'{vid}.pth.tar'))
        
        # clear memory / cache
        del all_v_features, video_seq_, video_seq_list, input_data
        gc.collect()
        torch.cuda.empty_cache()

    print(f'extraction to {args.output_dir} finished')
    return


def main(args, root):
    device = torch.device('cuda')

    if args.model in ['internvideo-ffmpeg']:
        model = InternVideo.load_model(os.path.expanduser("~/Align2/InternVideo/ckpt/InternVideo-MM-L-14.ckpt")).cuda()
        if args.half:
            state_dict = model.state_dict()
            convert_weights(model)
            model.load_state_dict(state_dict)
        fps = 8
        feature_tag = 'internvideo_feature'
    elif args.model == 'clip-L14':
        model, preprocess = clip.load('ViT-L/14', device=device)
        fps = 1
        feature_tag = 'CLIP_ViT_L14_feature'

    ### Optional: process short video first, for easy debugging ###
    htm_vlen_df = get_htm_vlen_df()
    htm_vid_to_vlen = dict(zip(htm_vlen_df['vid'], htm_vlen_df['vlen']))
    vid_list = sorted(glob.glob(os.path.join(root, '*')), key=lambda x: htm_vid_to_vlen.get(os.path.basename(x).split('.')[0], 10000))
    
    vid_list = [os.path.join(root, i.strip()) for i in vid_list]
    if len(vid_list) == 0:
        print(f'no videos found in {root}, skip')
    ori_num_vid = len(vid_list)
    
    args.partname = os.path.basename(root)
    
    if os.environ['LOGNAME'] == 'htd':  # local
        args.output_dir = os.path.expanduser(f'/scratch/shared/beegfs/htd/htm_feature/internvideo_feature')
    else:
        args.output_dir = os.path.expanduser(f'~/{feature_tag}_fps{fps}/')

    output_path = os.path.join(args.output_dir, args.partname)
    os.makedirs(output_path, exist_ok=True)

    drop_list = glob.glob(os.path.join(output_path, '*.pth.tar'))
    drop_list = set([os.path.basename(i).split('.')[0] for i in drop_list])

    ### TODO: drop long videos -- need to extract them later
    drop_vid_extra = ['lWhqORImND0',  # video_nonFE_part36/lWhqORImND0.mp4
                      'pFzXM08Ah0E',  # video_nonFE_part31/pFzXM08Ah0E.mp4.part download failed
                      'g3mvqjCSQN8',  # video_FE_part24/g3mvqjCSQN8.mp4 vlen=1520s
                      'iElp8HaMhZg',  # video_nonFE_part32/iElp8HaMhZg.mp4.part download failed
                      'bLw3NKUw2qw',  # video_nonFE_part27/bLw3NKUw2qw.mp4 has length 3711
                      '4P4-QIj0B2o',  # video_nonFE_part10/4P4-QIj0B2o.mp4 fail to load
                      '0pAZq7VHA2I',  # video_nonFE_part9/0pAZq7VHA2I.mp4 has length 4210
                      'xNI6ARjBCx0',  # video_nonFE_part5/xNI6ARjBCx0.mp4 ffprobe error
                      'otZups3dY4M',  # video_nonFE_part16/otZups3dY4M.mp4 has length 3994
                      '8SSOKppwpAE',  # video_nonFE_part27/8SSOKppwpAE.mp4 ffprobe error
                      'J5c2q3JPQgs',  # video_nonFE_part34/J5c2q3JPQgs.mp4 has length 4927
                      '1X6skAO2tUk',  # video_nonFE_part9/1X6skAO2tUk.mp4 ffprobe error
                      'didF9Y96y8o',  # video_nonFE_part13/didF9Y96y8o.mp4 has length 3646
                      'kPAz0eoe8A0',  # video_nonFE_part16/kPAz0eoe8A0.mp4 has length 5055
                      'nsumE3hdj1I',  # video_nonFE_part28/nsumE3hdj1I.mp4 has length 3624
                      'uUoQuaVcXu4',  # video_FE_part13/uUoQuaVcXu4.mp4 has length 3842
                      'jZS4PmaJBw8',  # video_FE_part6/jZS4PmaJBw8.mp4.parts download failed
                      'X5bMz5k1_-k',  # video_nonFE_part29/X5bMz5k1_-k.mp4 has length 3747
                      'L2Kut1P1kRE',  # video_nonFE_part32/L2Kut1P1kRE.mp4 has length 6209
                      'PyZq_Jm0aqQ',  # video_nonFE_part33/PyZq_Jm0aqQ.mp4 has length 3662
                      'qvvrMk3XlAk',  # video_nonFE_part33/qvvrMk3XlAk.mp4 has length 3888
                      '75eje3XKa7w',  # video_FE_part19/75eje3XKa7w.mp4 has length 3679
                      'M-3Elxy46tw',  # video_nonFE_part23/M-3Elxy46tw.mp4 has length 4283
                      'eICDhfkX6uo',  # video_FE_part13/eICDhfkX6uo.mp4 has length 6275
                      'qPncsf9Pk6I',  # video_nonFE_part3/qPncsf9Pk6I.mp4 ffprobe error
                      'a2ZaPZCmH4A',  # video_nonFE_part22/a2ZaPZCmH4A.mp4 has length 4547
                      'GBWi8QJjbyM',  # video_nonFE_part22/GBWi8QJjbyM.mp4 has length 4663
                      'dMBjHHS00Yw',  # video_FE_part16/dMBjHHS00Yw.mp4 has length 9099
                      'yoQeJVpy4kw',  # video_FE_part14/yoQeJVpy4kw.mp4 ffprobe error
                      'gvAT7pEUwIM',  # video_FE_part2/gvAT7pEUwIM.mp4 has length 3669
                      'OIKChki07YQ',  # video_FE_part17/OIKChki07YQ.mp4 has length 3655
                      'DkkiRtbFQI0',  # video_FE_part20/DkkiRtbFQI0.mp4 has length 4000
                      'XqTQbSga-D0',  # video_FE_part20/XqTQbSga-D0.mp4 has length 5793
                      'HoEcEbgJuik',  # video_nonFE_part15/HoEcEbgJuik.mp4 has length 3672
                      'yvcrrmeBwlo',  # video_nonFE_part19/yvcrrmeBwlo.mp4 has length 3900
                      'uXcqw0Gjq4E',  # video_nonFE_part15/uXcqw0Gjq4E.mp4 ffprobr error
                      'igrF5pD7Tbo',  # video_FE_part18/igrF5pD7Tbo.mp4 has length 3670
                      'cmUDEPWlvqI',  # video_FE_part15/cmUDEPWlvqI.mp4 has length 3797
                      'XaJSlYMFHyQ',  # video_FE_part15/XaJSlYMFHyQ.mp4 has length 4098
                      'yJ8UOFB2Y7g',  # video_FE_part1/yJ8UOFB2Y7g.mp4.ytdl ffprobe error
                      'EBXkTF4PBMI',  # video_nonFE_part17/EBXkTF4PBMI.mp4 has length 0
                      'v2455kM6_VE',  # video_FE_part1/v2455kM6_VE.mp4.ytdl ffprobe error
                      ]
    drop_list = drop_list.union(set(drop_vid_extra))
    print(f'find {len(drop_list)} existed videos in {output_path}')
    vid_list = [i for i in vid_list if os.path.basename(i).split('.')[0] not in drop_list]
    vid_list = sorted(vid_list, key=lambda x: htm_vid_to_vlen.get(x, 10000))

    if len(vid_list) == 0:
        print(f'no videos need to do in {root}, skip')
    else:
        print(f'Need to work on {ori_num_vid} --> {len(vid_list)} videos for {args.partname}')
    
    args.fps = fps
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = HTM_LongLoader(vid_list=vid_list, 
                             fps=fps,
                             use_internvideo_backend=False,
                             return_half=args.half
                             )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)
    extract_feature(loader, model, device, 0, args)

    print(root, 'finishes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='internvideo-ffmpeg', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('-j', '--num_workers', default=2, type=int)
    parser.add_argument('--half', default=1, type=int)
    parser.add_argument('--part', default=1, nargs='*', type=int)
    args = parser.parse_args()

    print(f"{PREFIX=}")

    rootlist = []
    for part in args.part:
        if part > 100:
            assert part // 100 == 1
            rootlist.append(os.path.join(PREFIX, f'videos_360p/video_FE_part{part-100}'))
            rootlist.append(os.path.join(PREFIX, f'videos/video_FE_part{part-100}'))
        else:
            rootlist.append(os.path.join(PREFIX, f'videos_360p/video_nonFE_part{part}'))
            rootlist.append(os.path.join(PREFIX, f'videos/video_nonFE_part{part}'))
    
    print(f"{rootlist=}")

    for idx, root in tqdm(enumerate(rootlist)):
        main(args, root)
