# BiSeNetV1 & BiSeNetV2

My implementation of [BiSeNetV1](https://arxiv.org/abs/1808.00897) and [BiSeNetV2](https://arxiv.org/abs/1808.00897).


The mIOU evaluation result of the models trained and evaluated on cityscapes train/val set is:
| none | ss | ssc | msf | mscf | fps(fp16/fp32) | link |
|------|:--:|:---:|:---:|:----:|:---:|:----:|
| bisenetv1 | 75.55 | 76.90 | 77.40 | 78.91 | 60/19 | [download](https://drive.google.com/file/d/140MBBAt49N1z1wsKueoFA6HB_QuYud8i/view?usp=sharing) |
| bisenetv2 | 74.12 | 74.18 | 75.89 | 75.87 | 50/16 | [download](https://drive.google.com/file/d/1qq38u9JT4pp1ubecGLTCHHtqwntH0FCY/view?usp=sharing) |

> Where **ss** means single scale evaluation, **ssc** means single scale crop evaluation, **msf** means multi-scale evaluation with flip augment, and **mscf** means multi-scale crop evaluation with flip evaluation. The eval scales of multi-scales evaluation are `[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]`, and the crop size of crop evaluation is `[1024, 1024]`.

> The fps is tested in different way from the paper. For more information, please see [here](./tensorrt).

Note that the model has a big variance, which means that the results of training for many times would vary within a relatively big margin. For example, if you train bisenetv2 for many times, you will observe that the result of **ss** evaluation of bisenetv2 varies between 72.1-74.4. 


## platform
My platform is like this: 
* ubuntu 18.04
* nvidia Tesla T4 gpu, driver 450.51.05
* cuda 10.2
* cudnn 7
* miniconda python 3.6.9
* pytorch 1.6.0


## get start
With a pretrained weight, you can run inference on an single image like this: 
```
$ python tools/demo_single_img.py --cfg-file bisenetv2 --weight-path /path/to/your/weights.pth --img-path ./example.png --save-path ./res/example_res.jpg
```

## dataset

Currently, support trainging with cityscapes and neolix dataset, just specify the data root, image list file for training and validation in config file, which are im_root, train_im_anns and val_im_anns.
you can directly modify configs/bisenetv2_cityscapes.py or bisentv2_neolix_fisheye.py, or create new config file and add them in configs/init.py
## train
In order to train the model, you can run command like this: 
```
$ export CUDA_VISIBLE_DEVICES=0,1

# if you want to train with apex
$ python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg-file bisenetv2 # or bisenetv1
or just run the script launch_training.sh

# if you want to train with pytorch fp16 feature from torch 1.6(pay attention that this command is not tested with new code)
$ python -m torch.distributed.launch --nproc_per_node=2 tools/train_amp.py --model bisenetv2 # or bisenetv1
```

Note that though `bisenetv2` has fewer flops, it requires much more training iterations. The the training time of `bisenetv1` is shorter.


## finetune from trained model
You can also load the trained model weights and finetune from it:
```
$ export CUDA_VISIBLE_DEVICES=0,1
$ python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --finetune-from ./res/model_final.pth --cfg-file bisenetv2 # or bisenetv1

# same with pytorch fp16 feature(pay attention that this command is not tested with new code)
$ python -m torch.distributed.launch --nproc_per_node=2 tools/train_amp.py --finetune-from ./res/model_final.pth --model bisenetv2 # or bisenetv1
```

## eval pretrained models

You can evaluate a trained model like this: 
```
$python tools/evaluate.py --weight-path /home/pantengteng/Programs/BiSeNet/res/model_final.pth --cfg-file bisenetv2_neolix_fisheye  
```
If you need to evaluate trained model in different dataset, just change the im_root and val_im_anns in config file.

# visualize
support two ways to specify images.
## specify image folder
Only the img-path parameter need to be specify in this way.
```
$python tools/demo.py --cfg-file bisenetv1 --weight-path /home/pantengteng/Programs/BiSeNet/res/2021-03-02-07-17-30_model_final.pth --img-path /data/pantengteng/lane_detection_test/snow_lane_detection --save-path /home/pantengteng/Programs/BiSeNet/results/hengtong_snow_test_v1
```
## specify image list
The img-path parameter and list-file parameter should be specified in this way.
```
$CUDA_VISIBLE_DEVICES=3 python tools/demo.py --cfg-file bisenetv2_neolix_fisheye --img-path /nfs/neolix_data1/neolix_dataset/test_dataset/freespace_segmentation/neolix_freespace_fisheye/images/ --list-file /nfs/neolix_data1/neolix_dataset/test_dataset/freespace_segmentation/neolix_freespace_fisheye/test.txt --save-path ./results/testtest --weight-path /data/pantengteng/bisenet/2021-06-21-19-36/iter_47999_model.pth
```
## Infer with tensorrt
You can go to [tensorrt](./tensorrt) For details.


### Be aware that this is the refactored version of the original codebase. You can go to the `old` directory for original implementation.


# 多尺度评估
https://github.com/CoinCheung/BiSeNet/issues/55
https://github.com/CoinCheung/BiSeNet/issues/85
