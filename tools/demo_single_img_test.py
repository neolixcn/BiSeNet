
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
parse.add_argument('--save-path', dest='save_path', type=str, default='./res.jpg',)
args = parse.parse_args()
cfg = cfg_factory[args.cfg_file]

if os.path.exists(args.save_path):
    os.path.makedirs(args.save_path)

palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

# define model
net = model_factory[cfg.model_type](19)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)
# im = cv2.imread(args.img_path)[:, :, ::-1]
im = cv2.imread(args.img_path)
im = cv2.resize(im, (1024,512))[:, :, ::-1]
# im = cv2.resize(im, (1024,1024))[:, :, ::-1]
#im = cv2.resize(im, (1920,1024))[:, :, ::-1]
im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

# inference
start = time.time()
out = net(im)[0].argmax(dim=1).squeeze().detach().cpu() # .numpy()
end = time.time()
# pred = palette[out]
color_map = torch.ones((out.shape[0], out.shape[1], 3))* 255
for id in trainId2color:
    color_map[out == id] = torch.tensor(trainId2color[id]).float()
color_map = color_map.numpy()
cv2.imwrite(args.save_path, color_map)
print("total time:", end - start)