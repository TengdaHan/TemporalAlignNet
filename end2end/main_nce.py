import os
import sys
import numpy as np
import torch
import random
import builtins
from transformers import AdamW
from tensorboardX import SummaryWriter
import time
import math
import functools
from einops import rearrange
import torch.cuda.amp as amp 
import torch.nn.functional as F
import torch.distributed as dist

# local modules
from model_e2e import MyS3D
from config import parse_args, set_path
from video_loader import HTM_ClipLoader
sys.path.append('../model')
from word2vec_model import Word2VecTokenizer
sys.path.append('../')
import utils.tensorboard_utils as TB
from utils.train_utils import clip_gradients
from utils.data_utils import DataLoaderBG
from utils.utils import AverageMeter, ProgressMeter, save_checkpoint, save_runtime_checkpoint


def get_loss(v_features, t_features, device, args):
    NUM = v_features.shape[0]
    target = torch.arange(NUM, device=device)
    if args.sim == 'cos':
        sim = v_features.matmul(t_features.t()).div(0.07)
    else:
        sim = v_features.matmul(t_features.t())
    loss_per_t = F.cross_entropy(sim, target)
    loss_per_v = F.cross_entropy(sim.transpose(0,1), target)
    loss = loss_per_t + loss_per_v
    top1_per_t = (sim.argmax(-1) == target).float().mean()
    top1_per_v = (sim.argmax(0) == target).float().mean()

    return {'loss': loss, 
            'loss-per-text': loss_per_t.detach(),
            'loss-per-video': loss_per_v.detach(),
            'top1-per-text': top1_per_t.detach(),
            'top1-per-video': top1_per_v.detach()}


def get_text_feature(model, token, args):
    if args.distributed:
        model = model.module
    if args.language_model == 'word2vec':
        t_features = model.s3d.text_module(token)['text_embedding']
    else:
        raise NotImplementedError
    return t_features


def train(loader, model, optimizer, lr_scheduler, grad_scaler, device, epoch, args):
    batch_time = AverageMeter('Time',':.2f')
    data_time = AverageMeter('Data',':.2f')
    losses = AverageMeter('Loss',':.4f')
    progress = ProgressMeter(
        len(loader), [batch_time, data_time, losses],
        prefix='Epoch:[{}]'.format(epoch))
    end = time.time()
    tic = time.time()

    model.train()
    optimizer.zero_grad()

    for idx, input_data in enumerate(loader):
        data_time.update(time.time() - end)
        video_seq = input_data['video'].to(device, non_blocking=True)
        B = video_seq.shape[0]
        video_seq = rearrange(video_seq, 'b n t c h w -> (b n) c t h w')
        with amp.autocast():
            v_features = model(video_seq)
            token = input_data['token'].to(device, non_blocking=True)
            token = rearrange(token, 'b n l -> (b n) l')
            t_features = get_text_feature(model, token, args)
            if args.sim == 'cos':
                v_features = v_features.div(v_features.norm(dim=-1, keepdim=True))
                t_features = t_features.div(t_features.norm(dim=-1, keepdim=True))
            loss_dict = get_loss(v_features, t_features, device, args)
            loss = loss_dict['loss']
            if (not torch.isinf(loss)) and (not torch.isnan(loss)):
                losses.update(loss.item(), B)

        grad_scaler.scale(loss).backward()
        if idx % args.backprop_freq == 0:
            grad_scaler.unscale_(optimizer)
            if args.clip_grad:
                _ = clip_gradients(model, clip_grad=args.clip_grad)  # 3.0
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()

        if args.prof is not None:
            args.prof.step()

        batch_time.update(time.time() - end)
        progress.display(idx)
        print('\t' + ' '.join([f"{k}:{v.item():.3f}" for k,v in loss_dict.items()]))
        lr_scheduler.step(args.iteration)

        if args.iteration % 5 == 0:
            for k, v in loss_dict.items():
                args.train_plotter.add_data(f'local/{k}', v.item(), args.iteration)
            args.train_plotter.add_data('local/lr', lr_scheduler.get_last_lr()[0], args.iteration)
            args.train_plotter.add_data('device/sps', 1/(time.time()-end), args.iteration)
            args.train_plotter.log_gpustat(step=args.iteration)
            args.train_plotter.writer.flush()

        end = time.time()
        args.iteration += 1

        if (args.iteration % args.runtime_save_iter == 0) and is_master():
            print('saving runtime checkpoint ...')
            if hasattr(model, 'module'):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best_acc': 1e5,
                'optimizer': optimizer.state_dict(),
                'iteration': args.iteration}
            save_runtime_checkpoint(save_dict, 
                filename=os.path.join(args.model_path, 'runtime.pth.tar'))


    print(f'epoch {epoch} finished, takes {time.time() - tic} seconds')
    args.train_plotter.add_data('global/loss', losses.avg, epoch)
    return losses.avg


def setup(args):
    # DDP setting
    args.distributed = int(os.environ.get('SLURM_JOB_NUM_NODES', "1")) > 1
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        # suppress printing if not on master gpu
        if args.rank!=0:
            def print_pass(*args):
                pass
            builtins.print = print_pass

    # CUDA setting and batch size conversion
    if torch.cuda.is_available():
        if args.distributed:
            torch.cuda.set_device(args.gpu)
            num_gpu = args.world_size
        else:
            if args.gpu is None:
                args.gpu = str(os.environ["CUDA_VISIBLE_DEVICES"])
            else:
                os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
            num_gpu = len(str(args.gpu).split(','))

        device = torch.device('cuda')
        args.num_gpu = num_gpu
        args.batch_size = num_gpu * args.batch_size
        print('=> Effective BatchSize = %d' % args.batch_size)
    else:
        args.num_gpu = 0
        device = torch.device('cpu')
        print('=> Run with CPU')

    # general setting
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    args.iteration = 1
    args.log_path, args.model_path, args.exp_path = set_path(args)
    
    # tensorboard monitor in background thread
    class DummyWriter():
        pass
    if is_master():
        writer_train = SummaryWriter(logdir=os.path.join(args.log_path, 'train'),
                                flush_secs=60)
        writer_val = SummaryWriter(logdir=os.path.join(args.log_path, 'val'),
                                flush_secs=60)
    else:
        writer_train = DummyWriter()
        writer_val = DummyWriter()
    args.train_plotter = TB.PlotterThread(writer_train)
    args.val_plotter = TB.PlotterThread(writer_val)

    # language tokenizer
    if args.language_model == 'word2vec':
        args.tokenizer = Word2VecTokenizer()
    else:
        raise NotImplementedError

    # log args if in sbatch job
    if ('/srun' in os.environ['_']) and is_master():
        print('running command: {')
        for key, item in args.__dict__.items():
            print(f'  "{key}": {item}')
        print('}')

    return device


def get_dataset(args):
    D = HTM_ClipLoader
    train_dataset = D(
        auto_align_tag=args.auto_align_tag,
        tokenizer=args.tokenizer,
        mode='train',
        num_frames=args.num_frames,
        fps=args.fps)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = DataLoaderBG(train_dataset,
        batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn, pin_memory=True, drop_last=True,
        shuffle=(train_sampler is None), sampler=train_sampler, 
        worker_init_fn=set_worker_sharing_strategy
    )
    return train_dataset, train_loader


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy('file_system')


def get_model_card(tag):
    model_card_dict = {}
    return model_card_dict.get(tag, tag)


def optim_policy(model, args, policy='default'):
    params = []
    no_decay = ['.ln_', '.bn', '.bias', '.logit_scale', '.entropy_scale']
    param_group_no_decay = []
    param_group_with_decay = []

    if policy == 'default':
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f'Param not requires_grad: {name}')
                continue
            if any([i in name for i in no_decay]):
                param_group_no_decay.append(param)
            else:
                param_group_with_decay.append(param)
        params.append({'params': param_group_no_decay, 'lr': args.lr, 'weight_decay': 0.0})
        params.append({'params': param_group_with_decay, 'lr': args.lr, 'weight_decay': args.wd})
    else:
        raise NotImplementedError

    return params


def main(args):
    device = setup(args)

    ### model ###
    if args.model in ['s3d',]:
        model = MyS3D(language_model=args.language_model, freezeBN=args.freezeBN, pretrained_s3d=args.pt_backbone)
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
            find_unused_parameters=True,
            static_graph=True)
        model_without_dp = model.module
    else:
        model_without_dp = model

    _, train_loader = get_dataset(args)

    ### optimizer ###
    params = optim_policy(model, args, args.optim_policy)
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.wd, correct_bias=True)
    best_loss = 1e5

    ### restart ###
    if args.resume:
        print(f"resume from checkpoint {args.resume}")
        args.resume = get_model_card(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['state_dict']
        args.start_epoch = checkpoint['epoch']+1
        args.iteration = checkpoint['iteration']
        best_acc = checkpoint['best_acc']
        if args.convert_from_frozen_bn:
            tmp_state_dict = {}
            for k,v in state_dict.items():
                if '.bn' in k:
                    tmp_state_dict[k.replace('.scale', '.weight')] = v
                else:
                    tmp_state_dict[k] = v
            state_dict = tmp_state_dict
        try:
            model_without_dp.load_state_dict(state_dict)
        except:
            missing_keys, unexpected_keys = model_without_dp.load_state_dict(state_dict, strict=False)
            if len(missing_keys):
                print(f'[Missing keys]:{"="*12}\n{chr(10).join(missing_keys)}\n{"="*20}')
            if len(unexpected_keys):
                print(f'[Unexpected keys]:{"="*12}\n{chr(10).join(unexpected_keys)}\n{"="*20}')
            user_input = input('[WARNING] Non-Equal load for resuming training, continue? [y/n]')
            if user_input.lower() == 'n':
                sys.exit()
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(f'Not resuming optimizer states due to Error: {e}\nInitialized the optimizer instead...')

    if args.pretrain:
        print(f"pretrain from checkpoint {args.pretrain}")
        args.pretrain = get_model_card(args.pretrain)
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        state_dict = checkpoint['state_dict']
        best_acc = checkpoint['best_acc']
        if args.convert_from_frozen_bn:
            tmp_state_dict = {}
            for k,v in state_dict.items():
                if '.bn' in k:
                    tmp_state_dict[k.replace('.scale', '.weight')] = v
                else:
                    tmp_state_dict[k] = v
            state_dict = tmp_state_dict
        try:
            model_without_dp.load_state_dict(state_dict)
        except:
            missing_keys, unexpected_keys = model_without_dp.load_state_dict(state_dict, strict=False)
            if len(missing_keys):
                print(f'[Missing keys]:{"="*12}\n{chr(10).join(missing_keys)}\n{"="*20}')
            if len(unexpected_keys):
                print(f'[Unexpected keys]:{"="*12}\n{chr(10).join(unexpected_keys)}\n{"="*20}')
            user_input = input('[WARNING] Non-Equal load for resuming training, continue? [y/n]')
            if user_input.lower() == 'n':
                sys.exit()
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(f'Not resuming optimizer states due to Error: {e}\nInitialized the optimizer instead...')

    args.decay_steps = args.epochs * len(train_loader)
    args.warmup_iterations = 1000
    def lr_schedule_fn(iteration, iter_per_epoch, args):
        if iteration < args.warmup_iterations:
            lr_multiplier = iteration / (args.warmup_iterations)
        else:
            lr_multiplier = 0.5 * \
                (1. + math.cos(math.pi * (iteration - args.warmup_iterations) / (args.epochs*iter_per_epoch - args.warmup_iterations)))
        return lr_multiplier

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, functools.partial(lr_schedule_fn, iter_per_epoch=len(train_loader)*args.resample, args=args)
    )
    lr_scheduler.step(args.iteration)  # for resume mode
    grad_scaler = amp.GradScaler()

    # profiler, optional
    args.prof = None

    # main loop
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)
        if args.distributed: 
            train_loader.sampler.set_epoch(epoch)
            
        train_loss = train(train_loader, model, optimizer, lr_scheduler, grad_scaler, device, epoch, args)

        if ((epoch % args.eval_freq == 0) or (epoch == args.epochs - 1)) and is_master():
            is_best = train_loss < best_loss  # temporary use val loss
            best_loss = min(train_loss, best_loss)
            state_dict = model_without_dp.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best_acc': best_loss,
                'optimizer': optimizer.state_dict(),
                'iteration': args.iteration}
            save_checkpoint(save_dict, is_best, args.eval_freq, 
                filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch), 
                keep_all=False,)

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))
    sys.exit(0)


def is_master():
    return os.environ.get('SLURM_PROCID', "0") == "0"


if __name__ == '__main__':
    args = parse_args()
    print(f'Using GPU id={os.environ.get("SLURM_PROCID", "0")}')
    main(args)


"""
Three ways to run:

1. single GPU training:
    CUDA_VISIBLE_DEVICES=0 python main_nce.py --freezeBN --sim cos --auto_align_tag htm_aa_v1 \
        --epochs 40 --batch_size 16 --num_frames 16 --fps 5

2. multi-GPU training with distributed.launch:
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
       --nproc_per_node=2 main_nce.py --freezeBN --sim cos --auto_align_tag htm_aa_v1 \
       --epochs 40 --batch_size 16 --num_frames 16 --fps 5

3. using SLURM sbatch script
"""