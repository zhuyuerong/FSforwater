import torch
import torch.nn as nn
import torch.nn.functional as F

from .FSGNet import (
    M_Conv, ConvNext, ResidualConv, SelfAttentionBlock, 
    CrossAttentionBlock, FastGuidedFilter_attention
)


class SemanticEdgeExtractor(nn.Module):
    """语义边缘提取器 - 从输入图像中提取多尺度语义边缘特征"""
    def __init__(self, in_channels=3, edge_channels=64):
        super(SemanticEdgeExtractor, self).__init__()
        
        # 多尺度边缘检测卷积
        self.edge_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, edge_channels//4, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(edge_channels//4),
            nn.ReLU(inplace=True)
        )
        
        self.edge_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, edge_channels//4, kernel_size=5, padding=2, groups=1),
            nn.BatchNorm2d(edge_channels//4),
            nn.ReLU(inplace=True)
        )
        
        self.edge_conv3 = nn.Sequential(
            nn.Conv2d(in_channels, edge_channels//4, kernel_size=7, padding=3, groups=1),
            nn.BatchNorm2d(edge_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # Sobel边缘检测 - 修复尺寸问题
        self.sobel_conv = nn.Conv2d(in_channels, edge_channels//4, kernel_size=3, padding=1, bias=False)
        
        # 初始化Sobel算子
        self._init_sobel_kernel()
        
        # 边缘特征融合
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(edge_channels, edge_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(edge_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(edge_channels, edge_channels, kernel_size=1),
            nn.Sigmoid()  # 生成边缘权重图
        )
        
    def _init_sobel_kernel(self):
        """初始化Sobel边缘检测核"""
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # 获取卷积层的实际形状
        out_channels = self.sobel_conv.weight.shape[0]  # edge_channels//4
        in_channels = self.sobel_conv.weight.shape[1]   # 3 (RGB)
        
        # 正确的初始化方式
        with torch.no_grad():
            for i in range(out_channels):
                for j in range(in_channels):
                    if i < out_channels // 2:
                        # 前半部分输出通道使用sobel_x
                        self.sobel_conv.weight[i, j] = sobel_x
                    else:
                        # 后半部分输出通道使用sobel_y
                        self.sobel_conv.weight[i, j] = sobel_y
    
    def forward(self, x):
        # 多尺度边缘特征
        edge1 = self.edge_conv1(x)
        edge2 = self.edge_conv2(x)
        edge3 = self.edge_conv3(x)
        edge_sobel = self.sobel_conv(x)
        
        # 融合所有边缘特征
        edge_features = torch.cat([edge1, edge2, edge3, edge_sobel], dim=1)
        edge_weights = self.edge_fusion(edge_features)
        
        return edge_weights, edge_features


class AdaptiveGuidanceModule(nn.Module):
    """自适应引导模块 - 根据解码层级自适应调整引导强度"""
    def __init__(self, guide_channels, decoder_channels):
        super(AdaptiveGuidanceModule, self).__init__()
        
        self.guide_channels = guide_channels
        self.decoder_channels = decoder_channels
        
        # 通道对齐
        self.guide_align = nn.Conv2d(guide_channels, decoder_channels, kernel_size=1)
        
        # 自适应权重生成
        self.adaptive_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(decoder_channels, decoder_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, guide_feature, decoder_feature):
        # 对齐引导特征通道
        guide_aligned = self.guide_align(guide_feature)
        
        # 生成自适应权重
        adaptive_w = self.adaptive_weight(decoder_feature)
        
        # 空间注意力
        decoder_mean = torch.mean(decoder_feature, dim=1, keepdim=True)
        decoder_max, _ = torch.max(decoder_feature, dim=1, keepdim=True)
        spatial_input = torch.cat([decoder_mean, decoder_max], dim=1)
        spatial_w = self.spatial_attention(spatial_input)
        
        # 自适应融合
        enhanced_guide = guide_aligned * adaptive_w * spatial_w
        
        return enhanced_guide


class FSGNet_InputGuided_v3(nn.Module):
    """FSGNet使用语义边缘增强的输入引导 - 第三版优化"""
    def __init__(self, channel, n_classes, base_c, depths, kernel_size):
        super(FSGNet_InputGuided_v3, self).__init__()

        # 原始编码器部分
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

        # 解码器部分
        self.up_residual_conv3 = ResidualConv(base_c * 8, base_c * 4, 1, 1)
        self.up_residual_conv2 = ResidualConv(base_c * 4, base_c * 2, 1, 1)
        self.up_residual_conv1 = ResidualConv(base_c * 2, base_c * 1, 1, 1)

        # 输出层
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

        # FastGuidedFilter和注意力机制
        self.fgf = FastGuidedFilter_attention(r=2, eps=1e-2)
        self.attention_block3 = CrossAttentionBlock(in_channels=base_c * 8)
        self.attention_block2 = CrossAttentionBlock(in_channels=base_c * 4)
        self.attention_block1 = CrossAttentionBlock(in_channels=base_c * 2)

        # 新增：语义边缘提取器
        self.edge_extractor = SemanticEdgeExtractor(in_channels=channel, edge_channels=base_c)
        
        # 修正的自适应引导模块 - 通道数匹配
        self.adaptive_guide3 = AdaptiveGuidanceModule(base_c * 4, base_c * 8)  # 256->512通道
        self.adaptive_guide2 = AdaptiveGuidanceModule(base_c * 2, base_c * 4)  # 128->256通道
        self.adaptive_guide1 = AdaptiveGuidanceModule(base_c * 1, base_c * 2)  # 64->128通道
        
        # 改进的输入引导特征处理
        self.input_guide_conv3 = nn.Sequential(
            nn.Conv2d(channel, base_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True)
        )
        self.input_guide_conv2 = nn.Sequential(
            nn.Conv2d(channel, base_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True)
        )
        self.input_guide_conv1 = nn.Sequential(
            nn.Conv2d(channel, base_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True)
        )
        
        # 边缘增强融合模块 - 将基础引导和边缘特征融合为最终的引导特征
        self.edge_enhance_conv3 = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c * 4, kernel_size=1),  # 融合后转换为256通道
            nn.BatchNorm2d(base_c * 4),
            nn.ReLU(inplace=True)
        )
        self.edge_enhance_conv2 = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c * 2, kernel_size=1),  # 融合后转换为128通道
            nn.BatchNorm2d(base_c * 2),
            nn.ReLU(inplace=True)
        )
        self.edge_enhance_conv1 = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c * 1, kernel_size=1),  # 融合后转换为64通道
            nn.BatchNorm2d(base_c * 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 提取语义边缘特征
        edge_weights, edge_features = self.edge_extractor(x)
        
        # 获取多尺度输入
        _, _, h, w = x.size()
        x_scale_2 = F.interpolate(x, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        x_scale_3 = F.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        # 编码器
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

        # 解码器 - 第3层（最深层）
        _, _, h, w = x3_down.size()
        x3_scale_input = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        
        # 生成基础引导特征
        base_guide_3 = self.input_guide_conv3(x3_scale_input)  # 64通道
        
        # 添加边缘增强
        edge_weights_3 = F.interpolate(edge_weights, size=(h, w), mode='bilinear', align_corners=True)  # 64通道
        
        # 融合边缘特征和基础引导特征
        enhanced_guide_3 = torch.cat([base_guide_3, edge_weights_3], dim=1)  # 128通道
        enhanced_guide_3 = self.edge_enhance_conv3(enhanced_guide_3)  # 转换为256通道
        
        # 自适应引导
        adaptive_guide_3 = self.adaptive_guide3(enhanced_guide_3, x3_down)  # 256->512通道
        guide_3_small = F.interpolate(adaptive_guide_3, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        
        fgf_out = self.fgf(guide_3_small, x4, adaptive_guide_3, self.attention_block3(guide_3_small, x4))
        x3_up = self.up_residual_conv3(fgf_out)

        # 解码器 - 第2层
        _, _, h, w = x2_down.size()
        x2_scale_input = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        
        base_guide_2 = self.input_guide_conv2(x2_scale_input)  # 64通道
        edge_weights_2 = F.interpolate(edge_weights, size=(h, w), mode='bilinear', align_corners=True)  # 64通道
        
        enhanced_guide_2 = torch.cat([base_guide_2, edge_weights_2], dim=1)  # 128通道
        enhanced_guide_2 = self.edge_enhance_conv2(enhanced_guide_2)  # 保持128通道
        
        adaptive_guide_2 = self.adaptive_guide2(enhanced_guide_2, x2_down)  # 128->256通道
        guide_2_small = F.interpolate(adaptive_guide_2, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        
        fgf_out = self.fgf(guide_2_small, x3_up, adaptive_guide_2, self.attention_block2(guide_2_small, x3_up))
        x2_up = self.up_residual_conv2(fgf_out)

        # 解码器 - 第1层
        _, _, h, w = x1_down.size()
        x1_scale_input = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        
        base_guide_1 = self.input_guide_conv1(x1_scale_input)  # 64通道
        edge_weights_1 = F.interpolate(edge_weights, size=(h, w), mode='bilinear', align_corners=True)  # 64通道
        
        enhanced_guide_1 = torch.cat([base_guide_1, edge_weights_1], dim=1)  # 128通道
        enhanced_guide_1 = self.edge_enhance_conv1(enhanced_guide_1)  # 转换为64通道
        
        adaptive_guide_1 = self.adaptive_guide1(enhanced_guide_1, x1_down)  # 64->128通道
        guide_1_small = F.interpolate(adaptive_guide_1, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        
        fgf_out = self.fgf(guide_1_small, x2_up, adaptive_guide_1, self.attention_block1(guide_1_small, x2_up))
        x1_up = self.up_residual_conv1(fgf_out)

        # 输出层
        _, _, h, w = x.size()
        out_3 = F.interpolate(x3_up, size=(h, w), mode='bilinear', align_corners=True)
        out_2 = F.interpolate(x2_up, size=(h, w), mode='bilinear', align_corners=True)
        out_3 = self.output_layer3(out_3)
        out_2 = self.output_layer2(out_2)
        out_1 = self.output_layer1(x1_up)

        return out_1, out_2, out_3