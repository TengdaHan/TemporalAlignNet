import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange
import os
import sys
sys.path.append('..')
from utils.utils import get_youtube_link, second_to_time
import copy
from tqdm import tqdm
import ffmpeg
from torch.nn.utils.rnn import pad_sequence


def circulant(tensor, dim):
    """get a circulant version of the tensor along the {dim} dimension.
    The additional axis is appended as the last dimension.
    E.g. 
    circulant(tensor([0,1,2]), dim=0) --> [[0,1,2],[2,0,1],[1,2,0]]"""
    S = tensor.shape[dim]
    tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], dim=dim)
    return tmp.unfold(dim, S, 1).flip((-1,))


def get_mask_from_time(start_list, end_list, num_timestamp, num_text, device='cuda'):
    """get a binary mask of shape [Batchsize, Num_text, Time].
    For the n-th sentence in the b-th video, 
    the vector [1x1xTime] has value 1 if the text corresponds this time segment."""
    B = len(start_list)
    steps = torch.arange(num_timestamp, device=device)[None,None,:].repeat(B, num_text, 1)
    start_list = pad_sequence(
        [torch.FloatTensor(i) for i in start_list],
        batch_first=True, 
        padding_value=num_timestamp+1e2).to(device, non_blocking=True)
    end_list = pad_sequence(
        [torch.FloatTensor(i) for i in end_list],
        batch_first=True, 
        padding_value=-1e2).to(device, non_blocking=True)
    mask = (start_list[:,:,None] <= steps) * (steps < end_list[:,:,None]) 
    return mask, start_list, end_list


def get_text_pos(start_list, end_list, device='cuda'):
    B = len(start_list)
    start_list = pad_sequence(
        [torch.FloatTensor(i) for i in start_list],
        batch_first=True, padding_value=0).to(device, non_blocking=True)
    end_list = pad_sequence(
        [torch.FloatTensor(i) for i in end_list],
        batch_first=True, padding_value=0).to(device, non_blocking=True)
    return torch.stack((start_list, end_list), dim=-1)


def get_loss(input_data, 
             video_seq, text_embed, video_padding_mask, text_padding_mask,
             logits, args, abs_text_pos):
    if args.model in ['init', 'cotrain']:
        logits_dual = logits['logits_dual']
        logits_joint = logits['logits_joint']
    if args.model in ['cotrain']:
        ema_logits_dual = logits['ema-logits_dual']
        ema_logits_joint = logits['ema-logits_joint']

    if args.sim == 'cos':
        logits_dual = logits_dual / 0.07
        logits_joint = logits_joint / 0.07
        if args.model in ['cotrain']:
            ema_logits_dual = ema_logits_dual / 0.07
            ema_logits_joint = ema_logits_joint / 0.07

    device = logits_dual.device
    B, T, _ = video_seq.shape
    N = text_embed.shape[1]
    num_enc_layers = logits_dual.shape[1]
    num_joint_layers = logits_joint.shape[1]

    loss_dict = {}

    # binary tgt: B,T,B,N
    binary_tgt_raw, _, _ = get_mask_from_time(
        input_data['start'], input_data['end'],
        num_timestamp=T, num_text=N, device=device)  # B,N,T
    binary_tgt = rearrange(binary_tgt_raw, 'b n t -> b t n').unsqueeze(2).repeat(1,1,B,1) * torch.eye(
        B, device=device)[:,None,:,None]
    flatten_text = np.array([item for sublist in input_data['text'] for item in sublist])

    if args.learn_agreement:
        with torch.no_grad():
            ### get prob mask for joint model ###
            if args.model in ['cotrain']:
                logits_joint_diag = torch.diagonal(
                    ema_logits_joint, dim1=0, dim2=3).permute(3,0,1,2)
            else:
                logits_joint_diag = torch.diagonal(
                    logits_joint, dim1=0, dim2=3).permute(3,0,1,2)
            tmp = logits_joint_diag.permute(0,2,1,3)
            tmp.masked_fill_(video_padding_mask[:,:,None,None].bool(), -6e4)
            tmp = tmp.permute(0,3,2,1)
            tmp.masked_fill_(text_padding_mask[:,:,None,None].bool(), -6e4)
            logits_joint_diag = tmp.permute(0,2,3,1)

            # 2-way softmax to approximate exclusion principle: each sentence corresponds to only one clip
            prob_per_text = logits_joint_diag.softmax(-1).div(0.07).softmax(-2)
            last_layer_prob_per_text = prob_per_text[:,-1,]
            last_layer_logits_per_text = logits_joint_diag[:,-1,]

            joint_self_tgt = torch.zeros(B,T,B,N, device=device)
            joint_max_prob_per_text = torch.zeros(B,N, device=device)
            joint_max_logits_per_text = torch.zeros(B,N, device=device)

            # vectorize
            old_durations = binary_tgt_raw.sum(-1)
            old_durations = torch.maximum(old_durations, torch.ones(1,device=device))
            old_durations.masked_fill_(text_padding_mask.bool(), 0)
            # create 1d avgpool kernel, roll it over all temporal positions
            k_avgpool = (torch.arange(T, device=device)[None,None,:].repeat(
                B,N,1) < old_durations[:,:,None])
            k_avgpool_circulant = circulant(k_avgpool, dim=-1)  # binary

            # avoid the last few cyclic rows by masking the lower diagonal
            tril_mask = torch.tril(torch.ones(T,T,device=device,dtype=torch.bool), diagonal=-1)
            k_avgpool_circulant.masked_fill_(tril_mask[None,None,:], 0)
            k_avgpool_circulant.masked_fill_((k_avgpool_circulant.sum(-1) < old_durations[:,:,None])[...,None], 0)

            # to avoid collapse towards the boundary
            k_avgpool_circulant[:,:,:,0] = 0  # never choose temp-index 0
            k_avgpool_circulant[:,:,:,-1] = 0  # never choose temp-index -1

            k_avgpool_circulant = k_avgpool_circulant.div(
                torch.clip(k_avgpool_circulant.sum(-1, keepdim=True).float(), min=1e-3)
                )
            prob_scan = last_layer_prob_per_text.permute(0,2,1)[:,:,None,:].mul(
                k_avgpool_circulant).sum(-1)

            max_prob, max_position = prob_scan.max(-1)
            joint_max_prob_per_text = max_prob

            max_position_k_avgpool = torch.gather(k_avgpool_circulant, dim=2, 
                index=max_position[:,:,None,None].repeat(1,1,1,T))
            joint_max_logits_per_text = last_layer_logits_per_text.permute(0,2,1).mul(
                max_position_k_avgpool.squeeze(2)).sum(-1)
            joint_self_tgt.masked_fill_(max_position_k_avgpool.permute(0,3,2,1).repeat(1,1,B,1).mul(
                torch.eye(B, device=device)[:,None,:,None]).bool(), 1)

            ### get prob mask for dual model ###
            if args.model in ['cotrain']:
                logits_dual_diag = torch.diagonal(
                        ema_logits_dual, dim1=0, dim2=3).permute(3,0,1,2)
            else:
                logits_dual_diag = torch.diagonal(
                        logits_dual, dim1=0, dim2=3).permute(3,0,1,2)
            tmp = logits_dual_diag.permute(0,2,1,3)
            tmp.masked_fill_(video_padding_mask[:,:,None,None].bool(), - 6e4)
            tmp = tmp.permute(0,3,2,1)
            tmp.masked_fill_(text_padding_mask[:,:,None,None].bool(), - 6e4)
            logits_dual_diag = tmp.permute(0,2,3,1)

            # 2-way softmax to approximate exclusion principle: each sentence corresponds to only one clip
            dual_prob_per_text = logits_dual_diag.softmax(-1).div(0.07).softmax(-2)
            dual_last_layer_prob_per_text = dual_prob_per_text[:,-1,]
            dual_last_layer_logits_per_text = logits_dual_diag[:,-1,]

            dual_self_tgt = torch.zeros(B,T,B,N, device=device)
            dual_max_prob_per_text = torch.zeros(B,N, device=device)
            dual_max_logits_per_text = torch.zeros(B,N, device=device)

            # vectorize
            prob_scan = dual_last_layer_prob_per_text.permute(0,2,1)[:,:,None,:].mul(
                k_avgpool_circulant).sum(-1)
            max_prob, max_position = prob_scan.max(-1)
            dual_max_prob_per_text = max_prob

            max_position_k_avgpool = torch.gather(k_avgpool_circulant, dim=2, 
                index=max_position[:,:,None,None].repeat(1,1,1,T))
            dual_max_logits_per_text = dual_last_layer_logits_per_text.permute(0,2,1).mul(
                max_position_k_avgpool.squeeze(2)).sum(-1)
            dual_self_tgt.masked_fill_(max_position_k_avgpool.permute(0,3,2,1).repeat(1,1,B,1).mul(
                torch.eye(B, device=device)[:,None,:,None]).bool(), 1)

            ### check agreement between dual and joint ###
            joint_self_tgt_diag = torch.diagonal(joint_self_tgt, dim1=0, dim2=2).permute(2,0,1)
            dual_self_tgt_diag = torch.diagonal(dual_self_tgt, dim1=0, dim2=2).permute(2,0,1)
            self_tgt_iou = torch.logical_and(joint_self_tgt_diag, dual_self_tgt_diag).sum(1).div(
                torch.clamp(torch.logical_or(joint_self_tgt_diag, dual_self_tgt_diag).sum(1).float(), min=1e-5)
            )

            intersection_self_tgt = torch.logical_and(joint_self_tgt, dual_self_tgt)
            union_self_tgt = torch.logical_or(joint_self_tgt, dual_self_tgt)

            dual_confidence_per_text = dual_max_logits_per_text >= torch.quantile(
                dual_max_logits_per_text[~text_padding_mask.bool()].float(),0.3)
            joint_confidence_per_text = joint_max_logits_per_text >= torch.quantile(
                joint_max_logits_per_text[~text_padding_mask.bool()].float(),0.3)
            confidence_per_text = torch.logical_and(dual_confidence_per_text, joint_confidence_per_text)

            iou_th = torch.tensor(0.5, device=device)
            confidence_iou = self_tgt_iou >= iou_th
            confidence_mask = torch.logical_and(confidence_per_text, confidence_iou)

            if args.temporal_agreement_type == 'i':
                agreement_self_tgt = intersection_self_tgt.clone().float()
                agreement_self_tgt[:,:,~confidence_mask.bool()] = 0
            elif args.temporal_agreement_type == 'u':
                agreement_self_tgt = union_self_tgt.clone().float()
                agreement_self_tgt[:,:,~confidence_mask.bool()] = 0
            elif args.temporal_agreement_type == 'keep':
                # keep youtube timestamp, if iou>th, replace by self-labelling
                agreement_self_tgt = binary_tgt.clone()
                agreement_self_tgt[:,:,confidence_iou.bool()] = union_self_tgt[:,:,confidence_iou.bool()].to(agreement_self_tgt.dtype)
            elif args.temporal_agreement_type == 'keep-joint':
                # keep youtube timestamp, if iou>th, replace by self-labelling from joint encoder
                agreement_self_tgt = binary_tgt.clone()
                agreement_self_tgt[:,:,confidence_iou.bool()] = joint_self_tgt[:,:,confidence_iou.bool()].to(agreement_self_tgt.dtype)

            # exclusive principle: remove duplicate 1s for the same timestamps, only keep the first
            agreement_self_tgt_diag = torch.diagonal(agreement_self_tgt, dim1=0, dim2=2)
            agreement_self_tgt_diag_dedup = torch.zeros_like(agreement_self_tgt_diag)
            first_pos_each_time = agreement_self_tgt_diag.argmax(1, keepdim=True)
            agreement_self_tgt_diag_dedup.scatter_(dim=1, index=first_pos_each_time, value=1)
            agreement_self_tgt_diag_dedup[:,0,:] = agreement_self_tgt_diag[:,0,:]
            # for those totally omitted text, fill them back with original tgt
            no_pos_mask = agreement_self_tgt_diag_dedup.sum(0) == 0
            agreement_self_tgt_diag_dedup[:,no_pos_mask] = torch.diagonal(binary_tgt,dim1=0,dim2=2)[:,no_pos_mask]
            agreement_self_tgt_dedup = agreement_self_tgt_diag_dedup.permute(2,0,1)[:,:,None,:].repeat(1,1,B,1) * torch.eye(B,B,device=device)[:,None,:,None]
            agreement_self_tgt = agreement_self_tgt_dedup

            loss_dict['confidence-ratio'] = confidence_mask[~text_padding_mask.bool()].float().mean()
            loss_dict['iou-threshold'] = iou_th

    ### prepare tgt ###
    if args.learn_agreement:
        no_padding_binary_tgt = agreement_self_tgt[:,:,~text_padding_mask.bool()]
    else:
        no_padding_binary_tgt = binary_tgt[:,:,~text_padding_mask.bool()]
    no_padding_binary_tgt = no_padding_binary_tgt.view(B*T,-1)
    video_mask_with_pos = no_padding_binary_tgt.sum(-1) > 0
    text_mask_with_pos = no_padding_binary_tgt.sum(-2) > 0

    ### get logits for dual model ###
    no_padding_logits_dual = logits_dual[:,:,:,~text_padding_mask.bool()]
    no_padding_logits_dual = no_padding_logits_dual.permute(1,0,2,3).reshape(num_enc_layers, B*T, -1)
    
    no_padding_logits_dual_pos = no_padding_logits_dual.clone()
    no_padding_logits_dual_pos[:,~no_padding_binary_tgt.bool()] = - 6e4
    no_padding_logits_dual_neg = no_padding_logits_dual

    v_numerator_dual = torch.logsumexp(no_padding_logits_dual_pos, dim=-1)
    v_denomenator_dual = torch.logsumexp(no_padding_logits_dual_neg, dim=-1)
    v_loss_milnce_dual = (v_denomenator_dual - v_numerator_dual)[:,video_mask_with_pos.bool()]
    
    t_numerator_dual = torch.logsumexp(no_padding_logits_dual_pos, dim=-2)
    t_denomenator_dual = torch.logsumexp(no_padding_logits_dual_neg, dim=-2)
    t_loss_milnce_dual = (t_denomenator_dual - t_numerator_dual)[:,text_mask_with_pos.bool()]

    loss_dual = (v_loss_milnce_dual.mean() + t_loss_milnce_dual.mean()) / 2
    loss_dict['loss-dual'] = loss_dual.detach()
    

    ### get logits for joint model ###
    no_padding_logits_joint = logits_joint[:,:,:,~text_padding_mask.bool()]
    no_padding_logits_joint = no_padding_logits_joint.permute(1,0,2,3).reshape(num_joint_layers, B*T, -1)
    no_padding_logits_joint_pos = no_padding_logits_joint.clone()
    no_padding_logits_joint_pos[:,~no_padding_binary_tgt.bool()] = - 6e4

    v_numerator_joint = torch.logsumexp(no_padding_logits_joint_pos, dim=-1)
    v_denomenator_joint = torch.logsumexp(no_padding_logits_joint, dim=-1)
    v_loss_milnce_joint = (v_denomenator_joint - v_numerator_joint)[:,video_mask_with_pos.bool()]
    
    t_numerator_joint = torch.logsumexp(no_padding_logits_joint_pos, dim=-2)
    t_denomenator_joint = torch.logsumexp(no_padding_logits_joint, dim=-2)
    t_loss_milnce_joint = (t_denomenator_joint - t_numerator_joint)[:,text_mask_with_pos.bool()]

    loss_joint = (v_loss_milnce_joint.mean() + t_loss_milnce_joint.mean()) / 2
    loss_dict['loss-joint'] = loss_joint.detach()

    if (args.loss_threshold > 0) or args.use_alignability_head:
        # threshold on per-text losss => only keep alignable text in the tgt
        with torch.no_grad():
            max_logits_dual_per_text = rearrange(torch.diagonal(logits_dual, dim1=0, dim2=3), 'l t n b -> l t b n')[-1,:,~text_padding_mask.bool()].max(0).values
            max_logits_dual_per_text_standard = (max_logits_dual_per_text - max_logits_dual_per_text.mean(0, keepdim=True)) / max_logits_dual_per_text.std(0, keepdim=True)
            max_logits_joint_per_text = rearrange(torch.diagonal(logits_joint, dim1=0, dim2=3), 'l t n b -> l t b n')[-1,:,~text_padding_mask.bool()].max(0).values
            max_logits_joint_per_text_standard = (max_logits_joint_per_text - max_logits_joint_per_text.mean(0, keepdim=True)) / max_logits_joint_per_text.std(0, keepdim=True)
            max_logits_combined = max_logits_dual_per_text_standard + max_logits_joint_per_text_standard
            t_th_metric = - max_logits_combined
            t_th_mask = t_th_metric <= torch.quantile(t_th_metric.float(), args.loss_threshold, -1, keepdim=True)

            no_padding_binary_tgt_th = no_padding_binary_tgt.clone()
            no_padding_binary_tgt_th[:, ~t_th_mask.bool()] = 0
            video_mask_with_pos_th = no_padding_binary_tgt_th.sum(-1) > 0

        if args.loss_threshold > 0:
            loss_dict['loss-dual-all'] = loss_dual.detach()
            loss_dict['loss-joint-all'] = loss_joint.detach()

            t_loss_milnce_th = t_loss_milnce_dual[:, t_th_mask].mean()
            v_loss_milnce_th = (v_denomenator_dual - v_numerator_dual)[:,video_mask_with_pos_th.bool()].mean()
            loss_dual_th = (v_loss_milnce_th + t_loss_milnce_th) / 2
            loss_dict[f'loss-dual'] = loss_dual_th.detach()

            t_loss_milnce_joint_th = t_loss_milnce_joint[:, t_th_mask].mean()
            v_loss_milnce_joint_th = (v_denomenator_joint - v_numerator_joint)[:,video_mask_with_pos_th.bool()].mean()
            loss_joint_th = (v_loss_milnce_joint_th + t_loss_milnce_joint_th) / 2
            loss_dict[f'loss-joint'] = loss_joint_th.detach()

        if args.use_alignability_head:
            with torch.no_grad():
                # 2=ignore, 0:neg, 1:pos
                t_align_th_mask = torch.ones_like(t_th_metric,) * 2.0
                # t_th_top = torch.quantile(t_th_metric.float(), 0.3, -1, keepdim=True)
                # t_th_bot = torch.quantile(t_th_metric.float(), 0.7, -1, keepdim=True)
                # t_align_th_mask.masked_fill_(t_th_metric <= t_th_top, 1.0)
                # t_align_th_mask.masked_fill_(t_th_metric >= t_th_bot, 0.0)
                t_th_top_mask = torch.logical_and(
                    max_logits_dual_per_text > torch.quantile(max_logits_dual_per_text.float(), 0.5, keepdim=True),
                    max_logits_joint_per_text > torch.quantile(max_logits_joint_per_text.float(), 0.5, keepdim=True),
                )
                t_th_bot_mask = torch.logical_and(
                    max_logits_dual_per_text < torch.quantile(max_logits_dual_per_text.float(), 0.5, keepdim=True),
                    max_logits_joint_per_text < torch.quantile(max_logits_joint_per_text.float(), 0.5, keepdim=True),
                )
                t_align_th_mask.masked_fill_(t_th_top_mask, 1.0)
                t_align_th_mask.masked_fill_(t_th_bot_mask, 0.0)

                if abs_text_pos is not None:
                    abs_text_center_no_pad = abs_text_pos[~text_padding_mask.bool(), :].mean(-1)
                    trim_mask = torch.logical_or(abs_text_center_no_pad < 0.2,  abs_text_center_no_pad > 0.8)
                    t_align_th_mask.masked_fill_(trim_mask, 0.0)

            logits_alignability_dual = logits['dual_logits_alignability']
            logits_alignability_joint = logits['joint_logits_alignability']

            logits_alignability_dual = logits_alignability_dual[...,0][
                ~text_padding_mask.bool()][text_mask_with_pos.bool()]
            
            # compute loss for each layer
            # logits_alignability_joint = logits_alignability_joint.permute(1,0,2,3)[...,0][:,
            #     ~text_padding_mask.bool()][:,text_mask_with_pos.bool()]

            # or compute loss for specific layer
            logits_alignability_joint = logits_alignability_joint[:,2,:,0][
                ~text_padding_mask.bool()][text_mask_with_pos.bool()]
            
            t_align_th_mask_binary = t_align_th_mask[t_align_th_mask!=2]
            pos_weight = torch.ones_like(t_align_th_mask_binary) * (1/torch.mean(t_align_th_mask_binary) - 1.0)
            # loss_bce_joint = F.binary_cross_entropy_with_logits(logits_alignability_joint[:,t_align_th_mask!=2], 
            #     t_align_th_mask[t_align_th_mask!=2][None,:].repeat(num_joint_layers,1), pos_weight=pos_weight)
            loss_bce_joint = F.binary_cross_entropy_with_logits(logits_alignability_joint[t_align_th_mask!=2], 
                t_align_th_mask[t_align_th_mask!=2], pos_weight=pos_weight)
            loss_bce_dual = F.binary_cross_entropy_with_logits(logits_alignability_dual[t_align_th_mask!=2], 
                t_align_th_mask[t_align_th_mask!=2], pos_weight=pos_weight)
            # alignability_top1 = ((logits_alignability_joint[-1,t_align_th_mask!=2]>0) == t_align_th_mask_binary).detach().float().mean()
            alignability_top1 = ((logits_alignability_joint[t_align_th_mask!=2]>0) == t_align_th_mask_binary).detach().float().mean()

            # loss_dict['loss-dual-bce'] = loss_bce_dual.detach()
            loss_dict['loss-joint-bce'] = loss_bce_joint.detach()
            loss_dict['alignability_top1'] = alignability_top1
    
    ### compute the final loss ###
    bce_weight = 1
    nce_weight = 0 if args.optim_policy == 'bce' else 1 

    if args.loss_threshold > 0:
        loss_total = (loss_dual + loss_joint) / 2  # only for monitoring
        loss = (loss_dual_th + loss_joint_th) / 2
        if args.use_alignability_head:
            loss = loss * nce_weight + bce_weight * loss_bce_joint
        loss_dict['loss-total'] = loss_total.detach()
    else:
        loss = (loss_dual + loss_joint) / 2
        if args.use_alignability_head:
            loss = loss * nce_weight + bce_weight * loss_bce_joint
    loss_dict['loss'] = loss

    ### visualization (optional) ###
    if False: # args.pretrain: # temporary, for debug
        if args.model in ['cotrain']:
            logits_dual_vis = logits_dual[:,-1,:]
            logits_joint_vis = logits_joint[:,-1,:]

        idx = 0

        visualize(logits_dual_vis * 0.07, binary_tgt, 
            input_data['text'], input_data['vid'],
            input_data['start'], input_data['end'], 
            'dual', idx, args)

        visualize(logits_joint_vis * 0.07, binary_tgt, 
            input_data['text'], input_data['vid'],
            input_data['start'], input_data['end'], 
            'joint', idx, args)

        # print(f"Youtube-URL: {get_youtube_link(input_data['cut_start'], input_data['vid'])}")

        # visualize(logits_joint_vis * 0.07, shift_timestamp.transpose(1,2), 
        #     input_data['text'], input_data['vid'],
        #     input_data['start'], input_data['end'], 
        #     'joint-shift-tgt', idx, args)

        if args.learn_agreement:
            visualize(agreement_self_tgt, binary_tgt, 
                input_data['text'], input_data['vid'],
                input_data['start'], input_data['end'], 
                'agreement_tgt', idx, args)
            visualize(last_layer_prob_per_text, binary_tgt, 
                input_data['text'], input_data['vid'],
                input_data['start'], input_data['end'], 
                'last_layer_prob_joint', idx, args)

            visualize(dual_last_layer_prob_per_text, binary_tgt, 
                input_data['text'], input_data['vid'],
                input_data['start'], input_data['end'], 
                'last_layer_prob_dual', idx, args)

            visualize(ema_logits_joint[:, -1], binary_tgt, 
                input_data['text'], input_data['vid'],
                input_data['start'], input_data['end'], 
                'ema_logits_joint', idx, args)

        import ipdb; ipdb.set_trace()

    return loss_dict



def visualize(raw_logits, binary_tgt, sentences, vids, starts, ends, name_tag, idx, args, 
              num_vis_sample=2, start_ts=0, alignability_gt=None, alignability_pred=None):
    # except cos similarity
    raw_logits = raw_logits.float().detach().cpu()
    binary_tgt = binary_tgt.detach().cpu()
    if 'shift' in name_tag:
        title = 'Shifted-GT'
    else:
        title = 'GT'

    figsize = (16,12)
    fig, axes = plt.subplots(num_vis_sample*2,1,figsize=figsize)  # 16,12
    with torch.no_grad():
        for b_idx in range(num_vis_sample):
            start_ = starts[b_idx]
            end_ = ends[b_idx]
            vid_ = vids[b_idx]
            sent_ = sentences[b_idx]
            num_sent = len(sent_)
            if raw_logits.dim() == 4:
                logits_ = raw_logits[b_idx, :, b_idx, :][:, 0:num_sent].transpose(0,1)
            else:
                logits_ = raw_logits[b_idx, :, 0:num_sent].transpose(0,1)
            if binary_tgt.dim() == 4:
                tgt_ = binary_tgt[b_idx, :, b_idx, :][:, 0:num_sent].transpose(0,1)
            elif binary_tgt.dim() == 3:
                tgt_ = binary_tgt[b_idx, :, :][:, 0:num_sent].transpose(0,1)
            else:
                raise NotImplementedError(f"dim:{binary_tgt.dims()} is not supported")
            ratio = 3
            height_ = num_sent * ratio
            logits_interpolate = F.interpolate(logits_[None,None,:,],
                size=(height_, logits_.shape[1]), mode='nearest')[0,0]
            tgt_interpolate = F.interpolate(tgt_[None,None,:,],
                size=(height_, logits_.shape[1]), mode='nearest')[0,0]
            
            tmp = []
            for s in sent_:
                if len(s) < 48:
                    tmp.append(s)
                else:
                    tmp.append(s[0:48]+'...')
            sent_ = tmp
            if alignability_gt is not None:
                sent_suffix_ = []
                for s, a in zip(sent_, alignability_gt):
                    if a:
                        sent_suffix_.append(s+"[{}]".format('\u2714'))
                    else:
                        sent_suffix_.append(s+"[{}]".format('\u2718'))
            else:
                sent_suffix_ = sent_

            if alignability_pred is not None:
                sent_suffix_pred_ = []
                for s, a in zip(sent_, alignability_pred):
                    if a:
                        sent_suffix_pred_.append(s+"[{}]".format('\u2714'))
                    else:
                        sent_suffix_pred_.append(s+"[{}]".format('\u2718'))
            else:
                sent_suffix_pred_ = sent_

            sent_ticks = np.arange(num_sent) * ratio + ratio/2 - 0.5

            time_ticks = np.arange(0,64+1,8) + start_ts
            time_ticks = second_to_time(time_ticks)

            axes[b_idx * 2].imshow(tgt_interpolate.numpy())
            axes[b_idx * 2].set_yticks(sent_ticks)
            axes[b_idx * 2].set_yticklabels(sent_suffix_)
            # axes[b_idx * 2].set_title(f'{title} for {vid_} from {start_}s to {end_}s')
            axes[b_idx * 2].set_xticks(np.arange(0,64+1,8)-0.5); axes[b_idx * 2].set_xticklabels(time_ticks)
            axes[b_idx * 2].grid(which='major', axis='x', linestyle='--')
            # axp = axes[b_idx * 2 + 1].imshow((logits_interpolate.numpy() + 1) / 2,)
            axp = axes[b_idx * 2 + 1].imshow(logits_interpolate.numpy(),)
            arg_max = logits_.argmax(-1)
            # axes[b_idx * 2 + 1].set_title(f'Pred for {vid_} from {start_}s to {end_}s\n'
            #     # f'Max at {arg_max}'
            #     )
            axes[b_idx * 2 + 1].set_yticks(sent_ticks)
            axes[b_idx * 2 + 1].set_yticklabels(sent_suffix_pred_)
            axes[b_idx * 2 + 1].set_xticks(np.arange(0,64+1,8)-0.5); axes[b_idx * 2 + 1].set_xticklabels(time_ticks)
            axes[b_idx * 2 + 1].grid(which='major', axis='x', linestyle='--')
            # cb = plt.colorbar(axp, ax=[axes[b_idx * 2 + 1]])
    
    plt.savefig(os.path.join(args.log_path, f'iter-{idx:02d}_{vid_}_{name_tag}.jpg'), dpi=300, bbox_inches='tight')
    plt.close()
    return 