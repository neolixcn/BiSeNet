#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

import lib.transform_cv2 as T
from lib.sampler import RepeatedDistSampler
import albumentations as A


class BaseDataset(Dataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train',convert_label=False):
        super(BaseDataset, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.trans_func = trans_func

        self.lb_map = None

        with open(annpath, 'r') as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths = [], []
        for pair in pairs:
            imgpth, lbpth = pair.split(',')
            self.img_paths.append(osp.join(dataroot, imgpth))
            self.lb_paths.append(osp.join(dataroot, lbpth))

        assert len(self.img_paths) == len(self.lb_paths)
        self.len = len(self.img_paths)
        self.convert_5cls_to_4=convert_label
        # self.extra_transform=

    def __getitem__(self, idx):
        impth, lbpth = self.img_paths[idx], self.lb_paths[idx]
        img, label = cv2.imread(impth)[:, :, ::-1], cv2.imread(lbpth, 0)
        if not self.lb_map is None:
            label = self.lb_map[label]
        im_lb = dict(im=img, lb=label)
        if not self.trans_func is None:
            im_lb = self.trans_func(im_lb)
        im_lb = self.to_tensor(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        # Pyten-CombineCermentAndOtherRoad
        if self.convert_5cls_to_4:
            label[label==3] =  2
            label[label==4] =  3

        return img.detach(), label.detach() # .unsqueeze(0)

    def __len__(self):
        return self.len


class TransformationTrain(object):

    def __init__(self, scales, cropsize):
        self.trans_func = T.Compose([
            T.RandomResizedCrop(scales, cropsize),
            T.RandomHorizontalFlip(),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
            T.AlbumAug(
                        A.Compose([
                        A.ToGray(p=0.3),
                        A.Blur(blur_limit=3, p=0.3),
                        ])
            ),
        ])

    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)
        return im_lb


class TransformationVal(object):

    def __init__(self, width, height):
        self.w = width
        self.h = height

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        im = cv2.resize(im, (self.w, self.h))
        lb = cv2.resize(lb, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        # for crop
        # im = im[208:, :, :]
        # lb = lb[208:, :]

        return dict(im=im, lb=lb)


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    ds = CityScapes('./data/', mode='val')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
