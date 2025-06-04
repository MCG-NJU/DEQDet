_base_ = ['./eqdet_swinl_1x_coco.py']
pretrained = './swin_large_patch4_window12_384_22k.pth'  # noqa
depths = [2, 2, 18, 2]

num_proposals = 300
num_things_classes = 80
num_stuff_classes = 53


refinement_steps = 20
num_query = 100
QUERY_DIM = 512
FEAT_DIM = 256
FF_DIM = 2048
# P_in for spatial mixing in the paper.
in_points = 64
# P_out for spatial mixing in the paper. Also named as `out_points` in this codebase.
out_patterns = 128
# G for the mixer grouping in the paper. Please distinguishe it from num_heads in MHSA in this codebase.
n_group = 4

model = dict(
    rpn_head=dict(
        type='FixPositionEmbeddingRPNHead',
        num_proposals=num_proposals,
        proposal_feature_channel=QUERY_DIM),
    roi_head=dict(
        type='DEQDetRoIHead',
        featmap_strides=[4, 8, 16, 32],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[]),  # simple place-holder for adamixer-base
        use_init_head=True,
        supervision_position=[1, 3, 6, 9, 12, refinement_steps],
        perturb_content_prob=0.2,
        perturb_position_prob=0.2,
        perturb_content_intensity=0.1,
        perturb_position_intensity=25,
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
        ],
    init_cfg=None)
)





# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
custom_keys.update({'bbox_head': dict(lr_mult=4, decay_mult=10)})
# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))
resume=True
load_from = None
