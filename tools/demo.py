
import sys
sys.path.insert(0, '.')
import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import time

import lib.transform_cv2 as T
from lib.models import model_factory
from configs import cfg_factory
from lib.cityscapes_labels import trainId2color

torch.set_grad_enabled(False)
np.random.seed(123)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--cfg-file', dest='cfg_file', type=str, default='bisenetv2', help="specify the name without suffix of config file",)
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png',)
parse.add_argument('--list-file', dest='list_file', type=str, default=None,)
parse.add_argument('--save-path', dest='save_path', type=str, default='./res.jpg',)
args = parse.parse_args()
cfg = cfg_factory[args.cfg_file]

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

# define model
net = model_factory[cfg.model_type](cfg.class_num)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)

if args.list_file:
    with open(args.list_file, "r") as f:
        images = f.readlines()
    image_list = [img.strip() for img in images]
else:
    image_list = [img for img in os.listdir(args.img_path) if not os.path.isdir(img)]

sum_time = 0
img_num = 0

for img in image_list:
    img_num += 1
    # import pdb; pdb.set_trace()
    org_im = cv2.imread(os.path.join(args.img_path,img))
    # for crop
    # resized_im = org_im[208:, :, ::-1]
    resized_im = cv2.resize(org_im, (1280, 512))[:, :, ::-1]
    im = to_tensor(dict(im=resized_im, lb=None))['im'].unsqueeze(0).cuda()

    # inference
    start = time.time()
    out = net(im)[0].argmax(dim=1).squeeze().detach().cpu() # .numpy()
    end = time.time()
    # pred = palette[out]
    color_map = torch.ones((out.shape[0], out.shape[1], 3))* 255
    for id in trainId2color:
        color_map[out == id] = torch.tensor(trainId2color[id]).float()
    color_map = color_map.numpy()
    combined_img = cv2.addWeighted(resized_im[:,:,::-1], 0.5, color_map[:,:,::-1], 0.5,0,dtype = cv2.CV_32F)
    if not os.path.exists(os.path.join(args.save_path, img.split("/")[0])):
        os.makedirs(os.path.join(args.save_path, img.split("/")[0]))
    cv2.imwrite(os.path.join(args.save_path, img), combined_img)
    sum_time += end-start
print("total time:", sum_time/img_num)