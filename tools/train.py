#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import argparse
import numpy as np
from tabulate import tabulate
import tqdm

import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.models import model_factory
from configs import cfg_factory
from lib.cityscapes_cv2 import get_data_loader
from tools.evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg

# apex
has_apex = True
try:
    from apex import amp, parallel
except ImportError:
    has_apex = False


## fix all random seeds
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')




def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
    parse.add_argument('--port', dest='port', type=int, default=44554,)
    parse.add_argument('--cfg-file', dest='cfg_file', type=str, default='bisenetv2', help="specify the name without suffix of config file",)
    parse.add_argument('--finetune-from', type=str, default=None,)
    return parse.parse_args()

args = parse_args()
cfg = cfg_factory[args.cfg_file] #bisenetv2_combined
if "val_bs" not in cfg.__dict__:
    cfg.val_bs = cfg.ims_per_gpu

def set_model():
    net = model_factory[cfg.model_type](cfg.class_num)
    # net = model_factory[cfg.model_type](19)
    if not args.finetune_from is None:
        logger = logging.getLogger()
        logger.info('finetune from ', args.finetune_from)
        state_all = torch.load(args.finetune_from, map_location='cpu')
        model_dict = net.state_dict()
        state_clip = {}
        for k,v in state_all.items():
            if not k in model_dict or v.shape != model_dict[k].shape:
                logger.info(k)
                continue
            state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)

    if cfg.use_sync_bn: net = set_syncbn(net)
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7)
    criteria_aux = [OhemCELoss(0.7) for _ in range(cfg.num_aux_heads)]
    return net, criteria_pre, criteria_aux

def set_syncbn(net):
    if has_apex:
        net = parallel.convert_syncbn_model(net)
    else:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    return net


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    return optim


def set_model_dist(net):
    if has_apex:
        net = parallel.DistributedDataParallel(net, delay_allreduce=True)
    else:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank, ],
            output_device=local_rank)
    return net


def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters


def train():
    is_dist = dist.is_initialized()
    rank = dist.get_rank()
    logger = logging.getLogger()
    logger.info(cfg.__dict__)
    if is_dist and rank == 0:
        writer = SummaryWriter(cfg.respth)

    ## training dataset
    dl = get_data_loader(cfg.dataset,
            cfg.im_root, cfg.train_im_anns,
            cfg.ims_per_gpu, cfg.scales, cfg.cropsize,
            cfg.max_iter, mode='train', distributed=is_dist)

    # validation dataset
    val_dl = get_data_loader(cfg.dataset,
        cfg.im_root, cfg.val_im_anns,
        cfg.val_bs, None, cfg.cropsize,
        None, mode='val', distributed=is_dist)

    ## model
    net, criteria_pre, criteria_aux = set_model()

    ## optimizer
    optim = set_optimizer(net)

    ## fp16
    if has_apex:
        opt_level = 'O1' if cfg.use_fp16 else 'O0'
        net, optim = amp.initialize(net, optim, opt_level=opt_level)

    ## ddp training
    net = set_model_dist(net)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    if rank == 0:
        pbar = tqdm.tqdm(total=len(dl),desc="train")
    
    best_iou = 0
    
    ## train loop
    for it, (im, lb) in enumerate(dl):
        net.train()
        im = im.cuda()
        lb = lb.cuda()
        lb = torch.squeeze(lb, 1)
        
        optim.zero_grad()
        logits, *logits_aux = net(im)
        loss_pre = criteria_pre(logits, lb)
        loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
        loss = loss_pre + sum(loss_aux)
        if has_apex:
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optim.step()
        torch.cuda.synchronize()
        lr_schdr.step()

        time_meter.update()
        loss_meter.update(loss.item())
        loss_pre_meter.update(loss_pre.item())
        _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]

        if rank == 0:
            pbar.update()
            pbar.set_postfix({"loss_pred":"{:.4f}".format(loss_pre.detach().item()), "loss":"{:.4f}".format(loss.detach().item())})

        ## print training log message
        if (it + 1) % 20 == 0:
            if rank == 0:
                grid = torchvision.utils.make_grid(im)
                writer.add_scalar("Loss/train_pre_loss", loss_pre.detach().item(), it)
                writer.add_scalar("Loss/train_total_loss", loss.detach().item(), it)
                for i,aux_loss in enumerate(loss_aux):
                    writer.add_scalar(f"Loss/train_aux_loss{i}", aux_loss.detach().item(), it)
                writer.add_image('images', grid, 0)
                # label_img = deocde_label(label)
                # writer.add_image('labels', grid, 0)
                # lr = lr_schdr.get_lr()
                # lr = sum(lr) / len(lr)
                # print_log_msg(
                #     it, cfg.max_iter, lr, time_meter, loss_meter,
                #     loss_pre_meter, loss_aux_meters)
        if rank == 0:
            writer.add_scalar("meta/lr", lr_schdr.get_lr()[0], it)

        ## validation
        if (it + 1) % 1000 == 0:
            net.eval()
            # single_scale = MscEvalV0((1., ), False)
            # mIOU = single_scale(net, val_dl, cfg.class_num)
            
            hist = torch.zeros(cfg.class_num, cfg.class_num, requires_grad=False).cuda()
            if dist.is_initialized() and dist.get_rank() != 0:
                diter = enumerate(val_dl)
            else:
                diter = enumerate(tqdm.tqdm(val_dl))
            for i, (imgs, label) in diter:
                val_loss_meter = AvgMeter('val_loss')
                label = label.squeeze(1).cuda()
                image = imgs.cuda()
                with torch.no_grad():
                    val_logits = net(image)[0]
                val_loss_pre = criteria_pre(val_logits, label)
                # loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
                val_loss = val_loss_pre # + sum(loss_aux)
                val_loss_meter.update(val_loss.item())
                probs = torch.softmax(val_logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                keep = label != 255
                hist += torch.bincount(
                    label[keep] * cfg.class_num + preds[keep],
                    minlength=cfg.class_num ** 2
                    ).view(cfg.class_num, cfg.class_num)
            if dist.is_initialized():
                dist.all_reduce(hist, dist.ReduceOp.SUM)
            ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
            miou = ious.mean()
            logger.info('single mIOU is: %s\n', miou)
            logger.info('class iou:{}'.format(ious))
            if rank == 0:
                writer.add_scalar("Metric/val_miou", miou, it)
                writer.add_scalar("Loss/val_pre_loss", val_loss_meter.get()[0], it)
            if miou > best_iou:
                logger.info("previous best iou: {:.4f}, new best mIOU:{:.4f} is got at {}".format(best_iou, miou, it))
                best_iou = miou
                if dist.get_rank() == 0:
                    state = net.module.state_dict()
                    save_pth = osp.join(cfg.respth, 'iter_{}_model.pth'.format(it))
                    torch.save(state, save_pth)

    ## dump the final model and evaluate the result
    # save_pth = osp.join(cfg.respth, 'model_final.pth')
    save_pth = osp.join(cfg.respth, '{}_model_final.pth'.format(time.strftime('%Y-%m-%d-%H-%M-%S')))
    logger.info('\nsave models to {}'.format(save_pth))
    state = net.module.state_dict()
    if dist.get_rank() == 0: torch.save(state, save_pth)

    logger.info('\nevaluating the final model')
    torch.cuda.empty_cache()
    heads, mious = eval_model(net, cfg.dataset, cfg.val_bs, cfg.im_root, cfg.val_im_anns, cfg.cropsize, cfg.class_num)
    logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))

    if rank == 0:
        pbar.close()
        writer.close()

    return


def main():
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:{}'.format(args.port),
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )
    cfg.respth = osp.join(cfg.respth, time.strftime('%Y-%m-%d-%H-%M'))
    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger('{}-train'.format(cfg.model_type), cfg.respth)
    # logger.info(cfg)
    train()


if __name__ == "__main__":
    main()
