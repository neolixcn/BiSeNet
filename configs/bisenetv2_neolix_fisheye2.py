
## bisenetv2
cfg = dict(
    dataset='neolix',
    model_type='bisenetv2',
    class_num=5,
    num_aux_heads=4,
    lr_start = 0.001,#1e-3, # 5e-2,
    weight_decay=5e-4,
    warmup_iters = 1000,
    max_iter = 50000, #100000,# 50000, #150000,
    im_root='/nfs/neolix_data1/neolix_dataset/develop_dataset/freespace_segmentation/neolix_freespace_fisheye',
    train_im_anns='/nfs/neolix_data1/neolix_dataset/develop_dataset/freespace_segmentation/neolix_freespace_fisheye/train_val.txt',
    val_im_anns='/nfs/neolix_data1/neolix_dataset/test_dataset/freespace_segmentation/neolix_freespace_fisheye/test.txt',
    # train_im_anns='/nfs/neolix_data1/neolix_dataset/develop_dataset/freespace_segmentation/neolix_freespace_fisheye/train.txt',
    # val_im_anns='/nfs/neolix_data1/neolix_dataset/develop_dataset/freespace_segmentation/neolix_freespace_fisheye/val.txt',
    scales= [0.75,1.25], #[0.25, 2.], #
    cropsize=[512,1280], #[736, 1024],
    ims_per_gpu= 8, #4
    use_fp16=False, #True,
    use_sync_bn=True,
    respth="/data/pantengteng/bisenet",#'/nfs/neolix_data1/models/panteng/bisenetv2',
)
