
## bisenetv2
cfg = dict(
    dataset="cityscapes",
    model_type='bisenetv2',
    class_num= 19,
    num_aux_heads=4,
    lr_start = 5e-2, # 0.1# 
    weight_decay= 5e-4,
    warmup_iters = 1000,
    max_iter = 150000,
    im_root='/nfs/neolix_data1/OpenSource_dataset/freespace_segmentation/cityscapes',
    train_im_anns='/nfs/neolix_data1/OpenSource_dataset/freespace_segmentation/cityscapes/trainImages.txt',
    val_im_anns='/nfs/neolix_data1/OpenSource_dataset/freespace_segmentation/cityscapes/valImages.txt',
    scales=[0.25, 2.],
    cropsize=[512,1280], # 736,
    ims_per_gpu=6, #4
    use_fp16=True, #False
    use_sync_bn=True, # False
    respth="/data/pantengteng/bisenet", # '/nfs/neolix_data1/models/panteng/bisenetv2',
)

