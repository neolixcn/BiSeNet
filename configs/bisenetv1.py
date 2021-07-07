
cfg = dict(
    model_type='bisenetv1',
    class_num=5,
    num_aux_heads=2,
    lr_start=1e-2,
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=80000,
    im_root='/data/pantengteng/neolix_freespace',
    train_im_anns='/data/pantengteng/neolix_freespace/train.txt',
    val_im_anns='/data/pantengteng/neolix_freespace/val.txt',
    scales=[0.75, 2.],
    cropsize=[720, 1280],
    ims_per_gpu=8,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res',
)
