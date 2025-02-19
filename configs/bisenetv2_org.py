
## bisenetv2
cfg = dict(
    dataset="cityscapes",
    model_type='bisenetv2',
    class_num= 19,
    num_aux_heads=4,
    lr_start = 5e-2,
    weight_decay=5e-4,
    warmup_iters = 1000,
    max_iter = 150000,
    im_root='/nfs/neolix_data1/OpenSource_dataset/freespace_segmentation/cityscapes',
    train_im_anns='/nfs/neolix_data1/OpenSource_dataset/freespace_segmentation/cityscapes/trainImages.txt',
    val_im_anns='/nfs/neolix_data1/OpenSource_dataset/freespace_segmentation/cityscapes/valImages.txt',
    scales=[0.25, 2.],
    cropsize=[512, 1280], #512
    ims_per_gpu=8,
    use_fp16=True,
    use_sync_bn=False,
    respth='/nfs/neolix_data1/models/panteng/bisenetv2',
)
