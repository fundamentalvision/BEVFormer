import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from mmcv.runner.base_module import BaseModule
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from mmcv.cnn import build_norm_layer, build_conv_layer
import torch.utils.checkpoint as checkpoint
from mmdet.models.backbones.resnet import Bottleneck, BasicBlock


class ResNetFusion(BaseModule):
    def __init__(self, in_channels, out_channels, inter_channels, num_layer, norm_cfg=dict(type='SyncBN'),
                 with_cp=False):
        super(ResNetFusion, self).__init__()
        layers = []
        self.inter_channels = inter_channels
        for i in range(num_layer):
            if i == 0:
                if inter_channels == in_channels:
                    layers.append(BasicBlock(in_channels, inter_channels, stride=1, norm_cfg=norm_cfg))
                else:
                    downsample = nn.Sequential(
                        build_conv_layer(None, in_channels, inter_channels, 3, stride=1, padding=1, dilation=1,
                                         bias=False),
                        build_norm_layer(norm_cfg, inter_channels)[1])
                    layers.append(
                        BasicBlock(in_channels, inter_channels, stride=1, norm_cfg=norm_cfg, downsample=downsample))
            else:
                layers.append(BasicBlock(inter_channels, inter_channels, stride=1, norm_cfg=norm_cfg))
        self.layers = nn.Sequential(*layers)
        self.layer_norm = nn.Sequential(
                nn.Linear(inter_channels, out_channels),
                nn.LayerNorm(out_channels))
        self.with_cp = with_cp

    def forward(self, x):
        x = torch.cat(x, 1).contiguous()
        # x should be [1, in_channels, bev_h, bev_w]
        for lid, layer in enumerate(self.layers):
            if self.with_cp and x.requires_grad:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # nchw -> n(hw)c
        x = self.layer_norm(x)
        return x


@TRANSFORMER.register_module()
class PerceptionTransformerBEVEncoder(BaseModule):
    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 embed_dims=256,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformerBEVEncoder, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.rotate_center = rotate_center
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        if self.use_cams_embeds:
            self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        if self.use_cams_embeds:
            normal_(self.cams_embeds)

    def forward(self,
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                prev_bev=None,
                **kwargs):
        """
        obtain bev features.
        """
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(bev_queries,
                                 feat_flatten,
                                 feat_flatten,
                                 bev_h=bev_h,
                                 bev_w=bev_w,
                                 bev_pos=bev_pos,
                                 spatial_shapes=spatial_shapes,
                                 level_start_index=level_start_index,
                                 prev_bev=None,
                                 shift=bev_queries.new_tensor([0, 0]).unsqueeze(0),
                                 **kwargs)
        # rotate current bev to final aligned
        prev_bev = bev_embed
        if 'aug_param' in kwargs['img_metas'][0] and 'GlobalRotScaleTransImage_param' in kwargs['img_metas'][0][
            'aug_param']:
            rot_angle, scale_ratio, flip_dx, flip_dy, bda_mat, only_gt = kwargs['img_metas'][0]['aug_param'][
                'GlobalRotScaleTransImage_param']
            prev_bev = prev_bev.reshape(bs, bev_h, bev_w, -1).permute(0, 3, 1, 2)  # bchw
            if only_gt:
                # rot angle
                # prev_bev = torchvision.transforms.functional.rotate(prev_bev, -30, InterpolationMode.BILINEAR)
                ref_y, ref_x = torch.meshgrid(
                    torch.linspace(0.5, bev_h - 0.5, bev_h, dtype=bev_queries.dtype, device=bev_queries.device),
                    torch.linspace(0.5, bev_w - 0.5, bev_w, dtype=bev_queries.dtype, device=bev_queries.device))
                ref_y = (ref_y / bev_h)
                ref_x = (ref_x / bev_w)
                grid = torch.stack((ref_x, ref_y), -1)
                grid_shift = grid * 2.0 - 1.0
                grid_shift = grid_shift.unsqueeze(0).unsqueeze(-1)
                # bda_mat = ( bda_mat[:2, :2] / scale_ratio).to(grid_shift).view(1, 1, 1, 2,2).repeat(grid_shift.shape[0], grid_shift.shape[1], grid_shift.shape[2], 1, 1)
                bda_mat = bda_mat[:2, :2].to(grid_shift).view(1, 1, 1, 2, 2).repeat(grid_shift.shape[0],
                                                                                    grid_shift.shape[1],
                                                                                    grid_shift.shape[2], 1, 1)
                grid_shift = torch.matmul(bda_mat, grid_shift).squeeze(-1)
                # grid_shift = grid_shift / scale_ratio
                prev_bev = torch.nn.functional.grid_sample(prev_bev, grid_shift, align_corners=False)
                # if flip_dx:
                #     prev_bev = torch.flip(prev_bev, dims=[-1])
                # if flip_dy:
                #     prev_bev = torch.flip(prev_bev, dims=[-2])
            prev_bev = prev_bev.reshape(bs, -1, bev_h * bev_w)
            prev_bev = prev_bev.permute(0, 2, 1)
        return prev_bev


@TRANSFORMER.register_module()
class PerceptionTransformerV2(PerceptionTransformerBEVEncoder):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 embed_dims=256,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 frames=(0,),
                 decoder=None,
                 num_fusion=3,
                 inter_channels=None,
                 **kwargs):
        super(PerceptionTransformerV2, self).__init__(num_feature_levels, num_cams, two_stage_num_proposals, encoder,
                                                      embed_dims, use_cams_embeds, rotate_center,
                                                      **kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        """Initialize layers of the Detr3DTransformer."""
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.frames = frames
        if len(self.frames) > 1:
            self.fusion = ResNetFusion(len(self.frames) * self.embed_dims, self.embed_dims,
                                       inter_channels if inter_channels is not None else len(
                                           self.frames) * self.embed_dims,
                                       num_fusion)

    def init_weights(self):
        """Initialize the transformer weights."""
        super().init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        return super().forward(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length,
            bev_pos,
            prev_bev,
            **kwargs
        )

    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=None,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        if len(self.frames) > 1:
            cur_ind = list(self.frames).index(0)
            assert prev_bev[cur_ind] is None and len(prev_bev) == len(self.frames)
            prev_bev[cur_ind] = bev_embed

            # fill prev frame feature 
            for i in range(1, cur_ind + 1):
                if prev_bev[cur_ind - i] is None:
                    prev_bev[cur_ind - i] = prev_bev[cur_ind - i + 1].detach()

            # fill next frame feature 
            for i in range(cur_ind + 1, len(self.frames)):
                if prev_bev[i] is None:
                    prev_bev[i] = prev_bev[i - 1].detach()
            bev_embed = [x.reshape(x.shape[0], bev_h, bev_w, x.shape[-1]).permute(0, 3, 1, 2).contiguous() for x in
                         prev_bev]
            bev_embed = self.fusion(bev_embed)

        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out
