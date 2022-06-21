import argparse
import os
import json
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='s3d', type=str)
    parser.add_argument('--seed', default=888,type=int)
    parser.add_argument('--language_model', default='word2vec', type=str)
    parser.add_argument('--dataset', default='htm', type=str)
    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--fps', default=5, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-7, type=float)
    parser.add_argument('--loss', default='nce', type=str)
    parser.add_argument('--wd', default=1e-5, type=float)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--test', default='', type=str)
    parser.add_argument('--pretrain', default='', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--clip_grad', default=0, type=float)
    parser.add_argument('--prefix', default='tmp', type=str)
    parser.add_argument('--name_prefix', default='', type=str)
    parser.add_argument('--gpu', default=None, type=str)
    parser.add_argument('--sim', default='cos', type=str)
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--runtime_save_iter', default=1000, type=int)
    parser.add_argument('--optim_policy', default='default', type=str)
    parser.add_argument('--optim', default='adamw', type=str)
    parser.add_argument('--pt_backbone', default=True, type=bool)
    parser.add_argument('--backprop_freq', default=1, type=int)
    parser.add_argument('--freezeBN', action='store_true')
    parser.add_argument('--convert_from_frozen_bn', action='store_true')
    parser.add_argument('--auto_align_tag', default='htm_aa_v1', type=str)
    parser.add_argument('-j', '--num_workers', default=12, type=int)

    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')

    args = parser.parse_args()
    return args


def set_path(args):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    args.launch_timestamp = dt_string

    if args.resume: 
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    elif args.test: 
        if os.path.dirname(args.test).endswith('model'):
            exp_path = os.path.dirname(os.path.dirname(args.test))
        else:
            exp_path = os.path.dirname(args.test)
    else:
        name_prefix = f"{args.name_prefix}_" if args.name_prefix else ""
        freeze_prefix = 'FreezeBN_' if args.freezeBN else ''
        init_tag = '' if args.pt_backbone else '_initBackbone'
        exp_path = (f"log-{args.prefix}/{name_prefix}{dt_string}_{freeze_prefix}"
            f"{args.model}_{args.auto_align_tag}_{args.loss}_{args.language_model}_sim-{args.sim}_{args.dataset}_frames{args.num_frames}_"
            f"fps{args.fps}_policy-{args.optim_policy}_bs{args.batch_size}_lr{args.lr}{init_tag}")

    log_path = os.path.join(exp_path, 'log')
    model_path = os.path.join(exp_path, 'model')

    if os.environ.get('SLURM_PROCID', "0") == "0":
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        with open(f'{log_path}/running_command.txt', 'a') as f:
            json.dump({'command_time_stamp':dt_string, **args.__dict__}, f, indent=2)
            f.write('\n')

    return log_path, model_path, exp_path