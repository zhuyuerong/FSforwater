import torch
import torch.nn as nn
import torch.nn.functional as F

from .FSGNet import (
    M_Conv, ConvNext, ResidualConv, SelfAttentionBlock, 
    CrossAttentionBlock, FastGuidedFilter_attention
)


class FSGNet_InputGuided(nn.Module):
    """FSGNet using input image downsampling as guide"""
    def __init__(self, channel, n_classes, base_c, depths, kernel_size):
        super(FSGNet_InputGuided, self).__init__()

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

        # Input processing for guidance - 将原始输入转换为合适的引导特征
        self.input_guide_conv3 = nn.Sequential(
            nn.Conv2d(channel, base_c * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c * 4),
            nn.ReLU(inplace=True)
        )
        self.input_guide_conv2 = nn.Sequential(
            nn.Conv2d(channel, base_c * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c * 2),
            nn.ReLU(inplace=True)
        )
        self.input_guide_conv1 = nn.Sequential(
            nn.Conv2d(channel, base_c * 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c * 1),
            nn.ReLU(inplace=True)
        )

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

        # Decoder with input-guided filtering
        # 第3层解码 - 使用输入图像的下采样作为引导
        _, _, h, w = x3_down.size()
        x3_scale_input = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        input_guide_3 = self.input_guide_conv3(x3_scale_input)  # 将输入转为合适的通道数
        input_guide_3_small = F.interpolate(input_guide_3, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        
        fgf_out = self.fgf(input_guide_3_small, x4, input_guide_3, self.attention_block3(input_guide_3_small, x4))
        x3_up = self.up_residual_conv3(fgf_out)

        # 第2层解码
        _, _, h, w = x2_down.size()
        x2_scale_input = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        input_guide_2 = self.input_guide_conv2(x2_scale_input)
        input_guide_2_small = F.interpolate(input_guide_2, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        
        fgf_out = self.fgf(input_guide_2_small, x3_up, input_guide_2, self.attention_block2(input_guide_2_small, x3_up))
        x2_up = self.up_residual_conv2(fgf_out)

        # 第1层解码
        _, _, h, w = x1_down.size()
        x1_scale_input = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        input_guide_1 = self.input_guide_conv1(x1_scale_input)
        input_guide_1_small = F.interpolate(input_guide_1, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        
        fgf_out = self.fgf(input_guide_1_small, x2_up, input_guide_1, self.attention_block1(input_guide_1_small, x2_up))
        x1_up = self.up_residual_conv1(fgf_out)

        # Output layers
        _, _, h, w = x.size()
        out_3 = F.interpolate(x3_up, size=(h, w), mode='bilinear', align_corners=True)
        out_2 = F.interpolate(x2_up, size=(h, w), mode='bilinear', align_corners=True)
        out_3 = self.output_layer3(out_3)
        out_2 = self.output_layer2(out_2)
        out_1 = self.output_layer1(x1_up)

        return out_1, out_2, out_3