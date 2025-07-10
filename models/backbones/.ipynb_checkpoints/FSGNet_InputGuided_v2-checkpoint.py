import torch
import torch.nn as nn
import torch.nn.functional as F

from .FSGNet import (
    M_Conv, ConvNext, ResidualConv, SelfAttentionBlock, 
    CrossAttentionBlock, FastGuidedFilter_attention
)


class SemanticEdgeExtractor(nn.Module):
    """语义边缘提取器 - 学习任务相关的边缘信息"""
    def __init__(self, in_channels=3):
        super(SemanticEdgeExtractor, self).__init__()
        
        # 多尺度边缘检测网络
        self.edge_extractor = nn.Sequential(
            # 第一层：大感受野捕获整体结构
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第二层：中等感受野细化边缘
            nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第三层：小感受野精确定位
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 输出层：生成单通道边缘图
            nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        
        # 边缘增强融合层
        self.enhancement_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 提取语义边缘
        semantic_edge = self.edge_extractor(x)  # (B, 1, H, W)
        
        # 边缘增强
        enhanced_input = torch.cat([x, semantic_edge], dim=1)  # (B, 4, H, W)
        enhanced_input = self.enhancement_conv(enhanced_input)  # (B, 3, H, W)
        
        return enhanced_input, semantic_edge


class AdaptiveGuideFusion(nn.Module):
    """自适应引导融合模块"""
    def __init__(self, channels):
        super(AdaptiveGuideFusion, self).__init__()
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        x_spatial = x_channel * spatial_att
        
        return x_spatial


class FSGNet_InputGuided_v2(nn.Module):
    """改进的FSGNet - 使用语义边缘增强的输入引导"""
    def __init__(self, channel, n_classes, base_c, depths, kernel_size):
        super(FSGNet_InputGuided_v2, self).__init__()

        # 语义边缘提取器
        self.semantic_edge_extractor = SemanticEdgeExtractor(in_channels=channel)

        self.input_layer = nn.Sequential(
            M_Conv(channel, base_c * 1, kernel_size=kernel_size),
            *[ConvNext(base_c * 1, kernel_size=kernel_size) for _ in range(depths[0])]
        )
        self.input_skip = nn.Sequential(
            M_Conv(channel, base_c * 1, kernel_size=kernel_size),
        )
        self.conv1 = M_Conv(channel, base_c * 1, kernel_size=3)

        self.down_conv_2 = nn.Sequential(*[
            nn.Conv2d(base_c * 2, base_c * 2, kernel_size=2, stride=2),
            *[ConvNext(base_c * 2, kernel_size=kernel_size) for _ in range(depths[1])]
            ])
        self.conv2 = M_Conv(channel, base_c * 2, kernel_size=3)

        self.down_conv_3 = nn.Sequential(*[
            nn.Conv2d(base_c * 4, base_c * 4, kernel_size=2, stride=2),
            *[ConvNext(base_c * 4, kernel_size=kernel_size) for _ in range(depths[2])]
            ])
        self.conv3 = M_Conv(channel, base_c * 4, kernel_size=3)

        self.down_conv_4 = nn.Sequential(*[
            nn.Conv2d(base_c * 8, base_c * 8, kernel_size=2, stride=2),
            *[ConvNext(base_c * 8, kernel_size=kernel_size) for _ in range(depths[3])]
            ])
        self.attn = SelfAttentionBlock()

        self.up_residual_conv3 = ResidualConv(base_c * 8, base_c * 4, 1, 1)
        self.up_residual_conv2 = ResidualConv(base_c * 4, base_c * 2, 1, 1)
        self.up_residual_conv1 = ResidualConv(base_c * 2, base_c * 1, 1, 1)

        self.output_layer3 = nn.Sequential(
            nn.Conv2d(base_c * 4, n_classes, 1, 1),
            nn.Sigmoid(),
        )
        self.output_layer2 = nn.Sequential(
            nn.Conv2d(base_c * 2, n_classes, 1, 1),
            nn.Sigmoid(),
        )
        self.output_layer1 = nn.Sequential(
            nn.Conv2d(base_c * 1, n_classes, 1, 1),
            nn.Sigmoid(),
        )

        self.fgf = FastGuidedFilter_attention(r=2, eps=1e-2)
        self.attention_block3 = CrossAttentionBlock(in_channels=base_c * 8)
        self.attention_block2 = CrossAttentionBlock(in_channels=base_c * 4)
        self.attention_block1 = CrossAttentionBlock(in_channels=base_c * 2)

        self.conv_cat_3 = M_Conv(base_c * 8 + base_c * 8, base_c * 8, kernel_size=1)
        self.conv_cat_2 = M_Conv(base_c * 8 + base_c * 4, base_c * 4, kernel_size=1)
        self.conv_cat_1 = M_Conv(base_c * 4 + base_c * 2, base_c * 2, kernel_size=1)

        # 改进的输入引导卷积层，集成自适应融合
        self.input_guide_conv3 = nn.Sequential(
            nn.Conv2d(channel, base_c * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_c * 8),
            nn.ReLU(inplace=True),
            AdaptiveGuideFusion(base_c * 8)
        )
        self.input_guide_conv2 = nn.Sequential(
            nn.Conv2d(channel, base_c * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_c * 4),
            nn.ReLU(inplace=True),
            AdaptiveGuideFusion(base_c * 4)
        )
        self.input_guide_conv1 = nn.Sequential(
            nn.Conv2d(channel, base_c * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_c * 2),
            nn.ReLU(inplace=True),
            AdaptiveGuideFusion(base_c * 2)
        )

        # 边缘信息融合层
        self.edge_fusion_conv3 = nn.Conv2d(1, base_c * 8, kernel_size=1, bias=False)
        self.edge_fusion_conv2 = nn.Conv2d(1, base_c * 4, kernel_size=1, bias=False)
        self.edge_fusion_conv1 = nn.Conv2d(1, base_c * 2, kernel_size=1, bias=False)

    def forward(self, x):
        # 语义边缘增强
        enhanced_input, semantic_edge = self.semantic_edge_extractor(x)
        
        # 获取多尺度输入
        _, _, h, w = enhanced_input.size()
        x_scale_2 = F.interpolate(enhanced_input, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        x_scale_3 = F.interpolate(enhanced_input, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        # 编码器（使用增强后的输入）
        x1 = self.input_layer(enhanced_input) + self.input_skip(enhanced_input)
        x1_conv = self.conv1(enhanced_input)
        x1_down = torch.cat([x1_conv, x1], dim=1)

        x2 = self.down_conv_2(x1_down)
        x2_conv = self.conv2(x_scale_2)
        x2_down = torch.cat([x2_conv, x2], dim=1)

        x3 = self.down_conv_3(x2_down)
        x3_conv = self.conv3(x_scale_3)
        x3_down = torch.cat([x3_conv, x3], dim=1)

        x4 = self.down_conv_4(x3_down)
        x4 = self.attn(x4)

        # 解码器 - 使用语义增强的引导
        # 第3层解码
        _, _, h, w = x3_down.size()
        x3_gf = torch.cat([x3_down, F.interpolate(x4, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x3_gf_conv = self.conv_cat_3(x3_gf)
        
        # 语义增强的引导图
        x3_scale_input = F.interpolate(enhanced_input, size=(h, w), mode='bilinear', align_corners=True)
        edge_3 = F.interpolate(semantic_edge, size=(h, w), mode='bilinear', align_corners=True)
        
        input_guide_3 = self.input_guide_conv3(x3_scale_input)
        edge_guide_3 = self.edge_fusion_conv3(edge_3)
        enhanced_guide_3 = input_guide_3 + edge_guide_3  # 融合边缘信息
        
        enhanced_guide_3_small = F.interpolate(enhanced_guide_3, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        
        fgf_out = self.fgf(enhanced_guide_3_small, x4, enhanced_guide_3, 
                          self.attention_block3(enhanced_guide_3_small, x4))
        x3_up = self.up_residual_conv3(fgf_out)

        # 第2层解码
        _, _, h, w = x2_down.size()
        x2_gf = torch.cat([x2_down, F.interpolate(x3_gf_conv, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x2_gf_conv = self.conv_cat_2(x2_gf)
        
        x2_scale_input = F.interpolate(enhanced_input, size=(h, w), mode='bilinear', align_corners=True)
        edge_2 = F.interpolate(semantic_edge, size=(h, w), mode='bilinear', align_corners=True)
        
        input_guide_2 = self.input_guide_conv2(x2_scale_input)
        edge_guide_2 = self.edge_fusion_conv2(edge_2)
        enhanced_guide_2 = input_guide_2 + edge_guide_2
        
        enhanced_guide_2_small = F.interpolate(enhanced_guide_2, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        
        fgf_out = self.fgf(enhanced_guide_2_small, x3_up, enhanced_guide_2, 
                          self.attention_block2(enhanced_guide_2_small, x3_up))
        x2_up = self.up_residual_conv2(fgf_out)

        # 第1层解码
        _, _, h, w = x1_down.size()
        x1_gf = torch.cat([x1_down, F.interpolate(x2_gf_conv, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x1_gf_conv = self.conv_cat_1(x1_gf)
        
        x1_scale_input = F.interpolate(enhanced_input, size=(h, w), mode='bilinear', align_corners=True)
        edge_1 = F.interpolate(semantic_edge, size=(h, w), mode='bilinear', align_corners=True)
        
        input_guide_1 = self.input_guide_conv1(x1_scale_input)
        edge_guide_1 = self.edge_fusion_conv1(edge_1)
        enhanced_guide_1 = input_guide_1 + edge_guide_1
        
        enhanced_guide_1_small = F.interpolate(enhanced_guide_1, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        
        fgf_out = self.fgf(enhanced_guide_1_small, x2_up, enhanced_guide_1, 
                          self.attention_block1(enhanced_guide_1_small, x2_up))
        x1_up = self.up_residual_conv1(fgf_out)

        # 输出层
        _, _, h, w = x.size()
        out_3 = F.interpolate(x3_up, size=(h, w), mode='bilinear', align_corners=True)
        out_2 = F.interpolate(x2_up, size=(h, w), mode='bilinear', align_corners=True)
        out_3 = self.output_layer3(out_3)
        out_2 = self.output_layer2(out_2)
        out_1 = self.output_layer1(x1_up)

        return out_1, out_2, out_3