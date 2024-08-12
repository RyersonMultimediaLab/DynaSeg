_base_ = ['../default_runtime.py']

model = dict(
    type='MyCustomArchitectureOptimalClustersResNet18',
    input_dim = 3,
    T=10,
    mu=50,  # Weight for spatial continuity loss
    update_factor = 1,
    num_layers=3,
    qdy=64,  # Number of output classes
)

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
out_pipeline = [dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)]


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='CustomReplayDataset',
        data_source=dict(
            type='CocoImageList',
            root='./data/coco',
            img_prefix='val2017',
            seg_prefix='stuffthingmaps/val2017',
            img_postfix='.jpg',
            seg_postfix='.png',
            list_file='data/curated/val2017/Coco164kFull_Stuff_Coarse_7.txt',
            file_client_args=dict(backend='disk'),
            return_label=False),
        inv_pipelines=[],
        eqv_pipeline=[],
        shared_pipelines=[dict(type='ResizeCenterCrop', res=640)],
        out_pipeline=out_pipeline,
        prefetch=False,
        return_label=False,
        mode='train',
        res1=320,
        res2=640,
    ),
    val=dict(
        type='CustomCocoEvalDataset',
        samples_per_gpu=2,
        data_source=dict(
            type='CocoImageList',
            root='./data/coco',
            img_prefix='val2017',
            seg_prefix='stuffthingmaps/val2017',
            img_postfix='.jpg',
            seg_postfix='.png',
            list_file='data/curated/val2017/Coco164kFull_Stuff_Coarse_7.txt',
            file_client_args=dict(backend='disk'),
            return_label=True),
        img_out_pipeline=out_pipeline,
        res=320,
    ),
    test=dict(
        type='CustomCocoEvalDataset',
        samples_per_gpu=2,
        data_source=dict(
            type='CocoImageList',
            root='./data/coco',
            img_prefix='val2017',
            seg_prefix='stuffthingmaps/val2017',
            img_postfix='.jpg',
            seg_postfix='.png',
            list_file='data/curated/val2017/Coco164kFull_Stuff_Coarse_7.txt',
            file_client_args=dict(backend='disk'),
            return_label=True),
        img_out_pipeline=out_pipeline,
        res=320,
    ),
)

optimizer = dict(type='SGD', lr=0.1, weight_decay=0.0001, momentum=0.9)
optimizer_config = dict(grad_clip=None)
log_config = dict(interval=5)
# Learning rate scheduler
# lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
lr_config = dict(policy='Fixed')
runner = dict(type='EpochBasedRunner',
              max_epochs=5)
evaluator = [
    dict(
        type='ClusterIoUEvaluator',
        distributed=True,
        num_classes=27,
        num_thing_classes=12,
        num_stuff_classes=15)
]

custom_hooks = [
    dict(type='ValidateHook', initial=True, interval=1, trial=-1),
    dict(type='ReshuffleDatasetHook'),
    dict(
        type='LossWeightStepUpdateHook',
        interval=1,
        steps=[8, 10],
        gammas=[0, 1.0],
        key_names=['loss_kernel_cross_weight'])
]
custom_imports = dict(
    imports=[
        'Dynamicsegnet.models.architectures.my_custom_architecture_optimal_clusters_Resnet18',
        'Dynamicsegnet.engine.hooks.validate_hook',
        'Dynamicsegnet.engine.hooks.reshuffle_hook',
        'Dynamicsegnet.datasets.custom_coco_eval_dataset',
        'Dynamicsegnet.engine.hooks.wandblogger_hook',
    ],
    allow_failed_imports=False)

