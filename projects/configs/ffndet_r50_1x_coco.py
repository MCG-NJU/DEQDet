_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.deqdet'], allow_failed_imports=False)

# train_dataloader = dict(
#     dataset=dict(
#         ann_file='annotations/instances_val2017.json',
#         data_prefix=dict(img='val2017/'),
# ))

num_proposals = 100
num_things_classes = 80
num_stuff_classes = 53


num_query = 100
QUERY_DIM = 256
FEAT_DIM = 256
FF_DIM = 2048
# P_in for spatial mixing in the paper.
in_points = 64
# P_out for spatial mixing in the paper. Also named as `out_points` in this codebase.
out_patterns = 128
# G for the mixer grouping in the paper. Please distinguishe it from num_heads in MHSA in this codebase.
n_group = 4
num_stages = 6

model = dict(
    type='DEQDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapperWithGN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    rpn_head=dict(
        type='FixPositionEmbeddingRPNHead',
        num_proposals=num_proposals,
        proposal_feature_channel=256),
    roi_head=dict(
        type='FFNDetRoIHead',
        featmap_strides=[4, 8, 16, 32],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[]), # simple place-holder for adamixer-base
        num_stages=num_stages,
        bbox_head=[
            dict(
                type='AdaMixerHead',
                num_classes=80,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=1,
                feedforward_channels=FF_DIM,
                content_dim=QUERY_DIM,
                feat_channels=FEAT_DIM,
                dropout=0.0,
                in_points=in_points,
                out_points=out_patterns,
                n_groups=n_group,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                # NOTE: The following argument is a placeholder to hack the code. No real effects for decoding or updating bounding boxes.
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])),
        ]*num_stages,
    ),

    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ]),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1)
        ]*num_stages),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals)))
find_unused_parameters=True
# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.000025,
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
            custom_keys={
                'bbox_head': dict(lr_mult=4, decay_mult=10),
            }
    ),
    clip_grad=dict(max_norm=1, norm_type=2)
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
)
log_processor = dict(type='LogProcessor', window_size=100, by_epoch=True)
