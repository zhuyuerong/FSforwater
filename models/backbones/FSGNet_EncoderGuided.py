import torch
import torch.nn as nn
import torch.nn.functional as F

from .FSGNet import (
    M_Conv, ConvNext, ResidualConv, SelfAttentionBlock, 
    CrossAttentionBlock, FastGuidedFilter_attention
)


class FSGNet_EncoderGuided(nn.Module):
    """FSGNet using encoder features as guide (like x3_down)"""
    def __init__(self, channel, n_classes, base_c, depths, kernel_size):
        super(FSGNet_EncoderGuided, self).__init__()

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

        # 添加缺失的融合卷积层（参考原始FSGNet）
        self.conv_cat_3 = M_Conv(base_c * 8 + base_c * 8, base_c * 8, kernel_size=1)  # 1024→512
        self.conv_cat_2 = M_Conv(base_c * 8 + base_c * 4, base_c * 4, kernel_size=1)  # 768→256
        self.conv_cat_1 = M_Conv(base_c * 4 + base_c * 2, base_c * 2, kernel_size=1)  # 384→128

        # 用于处理编码器特征作为引导的卷积层 - 匹配目标特征通道数
        self.encoder_guide_conv3 = M_Conv(base_c * 8, base_c * 8, kernel_size=1)  # 512→512，匹配x4
        self.encoder_guide_conv2 = M_Conv(base_c * 4, base_c * 4, kernel_size=1)  # 256→256，匹配x3_up
        self.encoder_guide_conv1 = M_Conv(base_c * 2, base_c * 2, kernel_size=1)  # 128→128，匹配x2_up

    def forward(self, x):
        # Get multi-scale from input
        _, _, h, w = x.size()
        x_scale_2 = F.interpolate(x, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        x_scale_3 = F.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        # Encoder
        x1 = self.input_layer(x) + self.input_skip(x)
        x1_conv = self.conv1(x)
        x1_down = torch.cat([x1_conv, x1], dim=1)

        x2 = self.down_conv_2(x1_down)
        x2_conv = self.conv2(x_scale_2)
        x2_down = torch.cat([x2_conv, x2], dim=1)

        x3 = self.down_conv_3(x2_down)
        x3_conv = self.conv3(x_scale_3)
        x3_down = torch.cat([x3_conv, x3], dim=1)

        x4 = self.down_conv_4(x3_down)
        x4 = self.attn(x4)

        # Decoder with encoder-guided filtering
        # 第3层解码 - 按照原始FSGNet的融合方式，但使用编码器特征作为引导
        _, _, h, w = x3_down.size()
        
        # 构建融合特征（模仿原始FSGNet的x3_gf）
        x3_gf = torch.cat([x3_down, F.interpolate(x4, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x3_gf_conv = self.conv_cat_3(x3_gf)  # 融合并降维到512通道
        
        # 使用编码器特征作为引导
        encoder_guide_3 = self.encoder_guide_conv3(x3_down)  # 保持512通道
        encoder_guide_3_small = F.interpolate(encoder_guide_3, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        
        fgf_out = self.fgf(encoder_guide_3_small, x4, encoder_guide_3, self.attention_block3(encoder_guide_3_small, x4))
        x3_up = self.up_residual_conv3(fgf_out)

        # 第2层解码
        _, _, h, w = x2_down.size()
        x2_gf = torch.cat([x2_down, F.interpolate(x3_gf_conv, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x2_gf_conv = self.conv_cat_2(x2_gf)
        
        encoder_guide_2 = self.encoder_guide_conv2(x2_down)
        encoder_guide_2_small = F.interpolate(encoder_guide_2, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        
        fgf_out = self.fgf(encoder_guide_2_small, x3_up, encoder_guide_2, self.attention_block2(encoder_guide_2_small, x3_up))
        x2_up = self.up_residual_conv2(fgf_out)

        # 第1层解码
        _, _, h, w = x1_down.size()
        x1_gf = torch.cat([x1_down, F.interpolate(x2_gf_conv, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x1_gf_conv = self.conv_cat_1(x1_gf)
        
        encoder_guide_1 = self.encoder_guide_conv1(x1_down)
        encoder_guide_1_small = F.interpolate(encoder_guide_1, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        
        fgf_out = self.fgf(encoder_guide_1_small, x2_up, encoder_guide_1, self.attention_block1(encoder_guide_1_small, x2_up))
        x1_up = self.up_residual_conv1(fgf_out)

        # Output layers
        _, _, h, w = x.size()
        out_3 = F.interpolate(x3_up, size=(h, w), mode='bilinear', align_corners=True)
        out_2 = F.interpolate(x2_up, size=(h, w), mode='bilinear', align_corners=True)
        out_3 = self.output_layer3(out_3)
        out_2 = self.output_layer2(out_2)
        out_1 = self.output_layer1(x1_up)

        return out_1, out_2, out_3