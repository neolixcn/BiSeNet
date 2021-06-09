
## bisenetv2
cfg = dict(
    model_type='bisenetv2',
    class_num= 19,
    num_aux_heads=4,
    lr_start = 5e-2,
    weight_decay=5e-4,
    warmup_iters = 1000,
    max_iter = 50000, #150000,
    im_root='/nfs/nas/dataset/combined',
    train_im_anns='/nfs/nas/dataset/combined/train_new.txt',
    val_im_anns='/nfs/nas/dataset/combined/val_new.txt',
    scales=[0.25, 2.],
    cropsize= [736, 1280], #[512, 1024],
    ims_per_gpu=4, #8
    use_fp16=True,
    use_sync_bn=False,
    respth='./res',
)
