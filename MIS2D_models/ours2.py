import os
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import numpy as np
import matplotlib.pyplot as plt
import kornia.filters.sobel as sobel_filter


from MIS2D_models import structure_loss
from MIS2D_models.gcn_lib import Grapher as GCB
from MIS2D_models.gcn_lib import act_layer
from MIS2D_models.backbone.transformer.p2t import Block
from MIS2D_models.backbone.transformer import load_transformer_backbone_model
from MIS2D_models.backbone.cnn import load_cnn_backbone_model
from MIS2D_models.backbone.transformer.vit import ViT

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 skip_channels: int,
                 img_size: int=224,
                 patch_size: int=3,
                 stride: int=1,
                 pool_ratio: List[int]=[12, 16, 20, 24],
                 num_heads: int=1) -> None:
        super(Decoder, self).__init__()

        self.in_channels = in_channels + skip_channels
        self.patch_embedding = OverlapPatchEmbed(img_size=img_size,
                                                 patch_size=patch_size,
                                                 stride=stride,
                                                 in_chans=self.in_channels,
                                                 embed_dim=out_channels)
        self.block = Block(dim=out_channels, num_heads=num_heads)
        self.d_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels) for _ in pool_ratio])
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, skip):
        B, _, _, _ = x.size()
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)
        x_global, H, W = self.patch_embedding(x)
        x_global = self.block(x_global, H, W, self.d_convs)
        x_global = self.norm(x_global)
        x_global = x_global.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x_global, x

class SegmentationHead(nn.Module):
    def __init__(self,
                 in_channel_list: List[int],
                 num_classes: int,
                 scale_factor_list: List[int]) -> None:
        super(SegmentationHead, self).__init__()

        self.output_region_conv1 = nn.Sequential(nn.Conv2d(in_channel_list[0], num_classes, kernel_size=(1, 1), stride=(1, 1)),
                                                  nn.Upsample(scale_factor=scale_factor_list[0], mode='bilinear', align_corners=True))
        self.output_region_conv2 = nn.Sequential(nn.Conv2d(in_channel_list[1], num_classes, kernel_size=(1, 1), stride=(1, 1)),
                                                  nn.Upsample(scale_factor=scale_factor_list[1], mode='bilinear', align_corners=True))
        self.output_region_conv3 = nn.Sequential(nn.Conv2d(in_channel_list[2], num_classes, kernel_size=(1, 1), stride=(1, 1)),
                                                  nn.Upsample(scale_factor=scale_factor_list[2], mode='bilinear', align_corners=True))
        self.output_region_conv4 = nn.Sequential(nn.Conv2d(in_channel_list[3], num_classes, kernel_size=(1, 1), stride=(1, 1)),
                                                  nn.Upsample(scale_factor=scale_factor_list[3], mode='bilinear', align_corners=True))

        self.output_boundary_conv1 = nn.Sequential(nn.Conv2d(in_channel_list[0], 1, kernel_size=(1, 1), stride=(1, 1)),
                                                  nn.Upsample(scale_factor=scale_factor_list[0], mode='bilinear', align_corners=True))
        self.output_boundary_conv2 = nn.Sequential(nn.Conv2d(in_channel_list[1], 1, kernel_size=(1, 1), stride=(1, 1)),
                                                  nn.Upsample(scale_factor=scale_factor_list[1], mode='bilinear', align_corners=True))
        self.output_boundary_conv3 = nn.Sequential(nn.Conv2d(in_channel_list[2], 1, kernel_size=(1, 1), stride=(1, 1)),
                                                  nn.Upsample(scale_factor=scale_factor_list[2], mode='bilinear', align_corners=True))
        self.output_boundary_conv4 = nn.Sequential(nn.Conv2d(in_channel_list[3], 1, kernel_size=(1, 1), stride=(1, 1)),
                                                  nn.Upsample(scale_factor=scale_factor_list[3], mode='bilinear', align_corners=True))

    def forward(self, decoder_feature_list):
        region_stage1_output = self.output_region_conv1(decoder_feature_list[0])
        region_stage2_output = self.output_region_conv2(decoder_feature_list[1])
        region_stage3_output = self.output_region_conv3(decoder_feature_list[2])
        region_stage4_output = self.output_region_conv4(decoder_feature_list[3])

        boundary_stage1_output = self.output_boundary_conv1(decoder_feature_list[0])
        boundary_stage2_output = self.output_boundary_conv2(decoder_feature_list[1])
        boundary_stage3_output = self.output_boundary_conv3(decoder_feature_list[2])
        boundary_stage4_output = self.output_boundary_conv4(decoder_feature_list[3])

        region_output_list = [region_stage1_output, region_stage2_output, region_stage3_output, region_stage4_output]
        boundary_output_list = [boundary_stage1_output, boundary_stage2_output, boundary_stage3_output, boundary_stage4_output]

        return region_output_list, boundary_output_list

class ContextAttentionModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_selection_features: int=64) -> None:
        super(ContextAttentionModule, self).__init__()

        self.in_channels = in_channels
        self.num_selection_features = num_selection_features

        self.spatial_attention_map2 = nn.Sequential(
            nn.Conv2d(num_selection_features, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())

        self.smoothness = 1e-6

    def forward(self, x):
        B, C, H, W = x.size()

        # Uncertainty-based Filtering
        uncertainty_map = -1.0 * torch.sigmoid(x) * torch.log(torch.sigmoid(x) + self.smoothness)
        uncertainty_score = torch.mean(uncertainty_map, dim=(2, 3))

        # Choose M Feature Maps which have the lowest uncertainty score
        _, indices = torch.topk(uncertainty_score, self.num_selection_features, dim=1, largest=False)

        indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        selected_x = torch.gather(x, 1, indices)

        spatial_attention_map2 = self.spatial_attention_map2(selected_x)

        x = x * spatial_attention_map2

        return x

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut

        return x

class SkipConnectionModule(nn.Module):
    def __init__(self,
                 in_channel_list: List[int],
                 skip_channel_list: List[int],
                 num_graph_layers: int=1,
                 top_k_channels: int=32,
                 reduce_ratio: int=1,
                 img_size: int=224,
                 num_knn: int=11,
                 conv_type: str = 'mr',
                 gcb_act: str = 'gelu',
                 gcb_norm: str = 'batch',
                 bias: bool = True,
                 use_stochastic: bool = False,
                 epsilon: float = 0.2,
                 drop_out_rate: float = 0.0,
                 padding: int = 5) -> None:
        super(SkipConnectionModule, self).__init__()

        self.num_graph_layers = num_graph_layers
        self.top_k_channels = top_k_channels
        max_dilation = 18 // num_knn

        self.skip_feature_reduction1 = nn.Sequential(nn.Conv2d(in_channel_list[0], skip_channel_list[0], kernel_size=1, stride=1, padding=0),
                                                     nn.BatchNorm2d(skip_channel_list[0]), nn.ReLU(inplace=True))
        self.skip_feature_reduction2 = nn.Sequential(nn.Conv2d(in_channel_list[1], skip_channel_list[1], kernel_size=1, stride=1, padding=0),
                                                     nn.BatchNorm2d(skip_channel_list[1]), nn.ReLU(inplace=True))
        self.skip_feature_reduction3 = nn.Sequential(nn.Conv2d(in_channel_list[2], skip_channel_list[2], kernel_size=1, stride=1, padding=0),
                                                     nn.BatchNorm2d(skip_channel_list[2]), nn.ReLU(inplace=True))
        self.skip_feature_reduction4 = nn.Sequential(nn.Conv2d(in_channel_list[3], skip_channel_list[3], kernel_size=1, stride=1, padding=0),
                                                     nn.BatchNorm2d(skip_channel_list[3]), nn.ReLU(inplace=True))

        for layer_idx in range(num_graph_layers):
            graph_layer = GCB(sum(skip_channel_list), num_knn, min(0 // 4 + 1, max_dilation), conv_type, gcb_act, gcb_norm, bias, use_stochastic, epsilon, reduce_ratio,
                              n=img_size // 4 * img_size // 4, drop_path=drop_out_rate, relative_pos=True, padding=padding)
            ffn = FFN(sum(skip_channel_list), hidden_features=sum(skip_channel_list) * 4, out_features=sum(skip_channel_list), act='relu', drop_path=drop_out_rate)

            context_attention_layer = ContextAttentionModule(in_channels=sum(skip_channel_list), num_selection_features=top_k_channels)

            setattr(self, 'graph_convolution_{}'.format(layer_idx), graph_layer)
            setattr(self, 'ffn_{}'.format(layer_idx), ffn)
            setattr(self, 'context_attention_layer_{}'.format(layer_idx), context_attention_layer)

    def forward(self, encoder_features):
        x1, x2, x3, x4 = encoder_features

        x1 = self.skip_feature_reduction1(x1)
        x2 = self.skip_feature_reduction2(x2)
        x3 = self.skip_feature_reduction3(x3)
        x4 = self.skip_feature_reduction4(x4)

        x1_ori, x2_ori, x3_ori, x4_ori = x1, x2, x3, x4

        _, C1, H1, W1 = x1.shape
        _, C2, H2, W2 = x2.shape
        _, C3, H3, W3 = x3.shape
        _, C4, H4, W4 = x4.shape

        x1 = F.interpolate(x1, size=(H2, W2), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=(H2, W2), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=(H2, W2), mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        for layer_idx in range(self.num_graph_layers):
            graph_layer = getattr(self, 'graph_convolution_{}'.format(layer_idx))
            ffn = getattr(self, 'ffn_{}'.format(layer_idx))
            context_attention_layer = getattr(self, 'context_attention_layer_{}'.format(layer_idx))

            x = graph_layer(x)
            x = ffn(x)
            x = context_attention_layer(x)

        x1, x2, x3, x4 = torch.split(x, [C1, C2, C3, C4], dim=1)

        x1 = F.interpolate(x1, size=(H1, W1), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=(H3, W3), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=(H4, W4), mode='bilinear', align_corners=True)

        x1 = x1 + x1_ori
        x2 = x2 + x2_ori
        x3 = x3 + x3_ori
        x4 = x4 + x4_ori

        return [x1, x2, x3, x4]

class Ours2(nn.Module):
    def __init__(self,
                 args,
                 num_channels: int=3,
                 num_classes: int=1,
                 num_graph_layers: int=1,
                 skip_channels: int=32,
                 top_k_channels: int=32,
                 transformer_backbone: str='p2t_small',
                 pretrained: bool=True,
                 num_heads: int=1,
                 boundary_threshold: float=0.5) -> None:
        super(Ours2, self).__init__()

        assert num_channels == 3, "Input image must have 3 channels"

        self.args = args
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.transformer_backbone = transformer_backbone
        self.num_graph_layers = num_graph_layers
        self.skip_channels = skip_channels
        self.top_k_channels = top_k_channels
        self.num_heads = num_heads
        self.boundary_threshold = boundary_threshold

        # pyramid pooling ratios for each stage
        pool_ratios = [[12, 16, 20, 24], [6, 8, 10, 12], [3, 4, 5, 6], [1, 2, 3, 4]]

        self.feature_encoding = load_transformer_backbone_model(backbone_name=transformer_backbone, pretrained=pretrained)
        self.feature_encoding.head = nn.Identity()
        self.in_channels = 512
        self.encoder_channel_list = [64, 128, 320] + [self.in_channels]
        self.skip_channel_list = [skip_channels for _ in range(len(self.encoder_channel_list))]
        self.decoder_filters = [256, 128, 64]
        self.scale_factor_list = [32, 16, 8, 4]

        self.skip_connection_module = SkipConnectionModule(self.encoder_channel_list, self.skip_channel_list, self.num_graph_layers, self.top_k_channels)

        self.decoder_stage1 = Decoder(self.skip_channel_list[0], self.decoder_filters[0], self.skip_channel_list[0], pool_ratio=pool_ratios[2], num_heads=num_heads)
        self.decoder_stage2 = Decoder(self.decoder_filters[0], self.decoder_filters[1], self.skip_channel_list[1], pool_ratio=pool_ratios[1], num_heads=num_heads)
        self.decoder_stage3 = Decoder(self.decoder_filters[1], self.decoder_filters[2], self.skip_channel_list[2], pool_ratio=pool_ratios[0], num_heads=num_heads)

        self.segmentation_head = SegmentationHead([self.skip_channel_list[0]] + self.decoder_filters, num_classes, self.scale_factor_list)

    def forward(self, data):
        x = data['image']
        if x.size()[1] == 1: x = x.repeat(1, 3, 1, 1)
        _, _, H, W = x.shape

        features = self.feature_encoding.forward_feature(x)

        skip_features = self.skip_connection_module(features)

        decoder_stage1, _ = self.decoder_stage1(skip_features[3], skip_features[2])
        decoder_stage2, _ = self.decoder_stage2(decoder_stage1, skip_features[1])
        decoder_stage3, pre_features = self.decoder_stage3(decoder_stage2, skip_features[0])

        output = self.segmentation_head([skip_features[3], decoder_stage1, decoder_stage2, decoder_stage3])

        output_dict = {
            'prediction': output[0][-1],
            'target': data['target']}

        output_dict = self._calculate_criterion(output_dict, data)

        return output_dict

    def _calculate_criterion(self, output_dict, data):
        if self.num_classes > 1: loss = F.cross_entropy(output_dict['prediction'], data['target'])
        else: loss = structure_loss(output_dict['prediction'], data['target'])

        output_dict['loss'] = loss

        return output_dict

def _training_config(args):
    args.model_name = 'TransGUNet'

    # Dataset Argument
    args.num_channels = 3
    args.num_classes = 9 if args.train_data_type in ['Synapse'] else 1
    args.image_size = 224 if args.train_data_type in ['Synapse'] else 352
    args.metric_list = ['DSC', 'HD95', 'IoU'] if args.train_data_type in ['Synapse'] else ['DSC', 'IoU', 'WeightedF-Measure', 'S-Measure', 'E-Measure', 'MAE']
    if args.train_data_type in ['PolypSegData', 'ISIC2018']:
        args.mean = [0.485, 0.456, 0.406]
        args.std = [0.229, 0.224, 0.225]
    elif args.train_data_type in ['COVID19', 'BUSI', 'DSB2018', 'Synapse']:
        args.mean = [0.5]
        args.std = [0.5]

    # Training Argument
    args.multi_scale_train = False if args.train_data_type in ['Synapse'] else True
    args.train_batch_size = 24 if args.train_data_type in ['Synapse'] else 16
    args.test_batch_size = 1 if args.train_data_type in ['Synapse'] else 16

    if args.train_data_type in ['PolypSegData']: args.final_epoch = 50
    elif args.train_data_type in ['ISIC2018', 'BUSI']: args.final_epoch = 100
    elif args.train_data_type in ['Synapse']: args.final_epoch = 150
    elif args.train_data_type in ['COVID19']: args.final_epoch = 200


    # Optimizer Argument
    args.optimizer_name = 'AdamW'
    args.lr = 1e-4
    args.weight_decay = 1e-4

    return args