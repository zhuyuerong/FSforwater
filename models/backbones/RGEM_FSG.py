import torch
import torch.nn as nn
import torch.nn.functional as F
from .FSGNet import *  # 复用FSG-Net的所有模块


class ReverseGEM(nn.Module):
    """Reverse Graph Edge Module - 反向图边缘模块
    专门用于增强背景连通性，保护微小血管
    """
    def __init__(self, bg_enhancement_factor=0.5, vessel_protection_factor=0.1):
        super(ReverseGEM, self).__init__()
        self.bg_enhancement_factor = bg_enhancement_factor
        self.vessel_protection_factor = vessel_protection_factor
        
        # 边聚合函数（用于背景响应）
        self.edge_aggregation_func = nn.Sequential(
            nn.Linear(4, 1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, classification_response):
        """
        Args:
            classification_response: [B, 2, H, W] - [batch, class, height, width]
                                   class 0: 背景概率
                                   class 1: 血管概率
        Returns:
            processed_response: [B, 2, H, W] - 处理后的分类响应
        """
        # 提取背景和血管响应
        bg_response = classification_response[:, 0:1, :, :]  # [B, 1, H, W]
        vessel_response = classification_response[:, 1:2, :, :]  # [B, 1, H, W]
        
        # 对背景响应应用GEM增强
        enhanced_bg = self._apply_gem_to_background(bg_response)
        
        # 背景增强影响血管：背景越强，血管机会相对减少
        # 但同时保护原有的血管响应
        vessel_suppression = enhanced_bg * self.bg_enhancement_factor
        protected_vessel = vessel_response * (1 - vessel_suppression) + \
                          vessel_response * self.vessel_protection_factor
        
        # 重新组合输出
        processed_response = torch.cat([enhanced_bg, protected_vessel], dim=1)
        
        return processed_response
    
    def _apply_gem_to_background(self, bg_response):
        """对背景响应应用GEM处理"""
        B, C, H, W = bg_response.size()
        
        # 构建4邻域边连接（基于背景响应）
        edge = torch.stack(
            (
                torch.cat((bg_response[:, :, -1:], bg_response[:, :, :-1]), dim=2),  # 上移
                torch.cat((bg_response[:, :, 1:], bg_response[:, :, :1]), dim=2),   # 下移
                torch.cat((bg_response[:, :, :, -1:], bg_response[:, :, :, :-1]), dim=3),  # 左移
                torch.cat((bg_response[:, :, :, 1:], bg_response[:, :, :, :1]), dim=3)    # 右移
            ), dim=-1
        ) * bg_response.unsqueeze(dim=-1)  # [B, C, H, W, 4]
        
        # 边聚合：增强背景连通性
        edge_reshaped = edge.reshape(-1, 4)
        aggregated_edge = self.edge_aggregation_func(edge_reshaped).reshape((B, C, H, W))
        
        # 增强背景响应：原始响应 + 连通性增强
        enhanced_bg = bg_response + aggregated_edge * self.bg_enhancement_factor
        
        # 确保输出在合理范围内
        enhanced_bg = torch.clamp(enhanced_bg, 0.0, 1.0)
        
        return enhanced_bg


class RGEM_FSGNet(FSGNet):
    """FSG-Net + Reverse GEM模块（实验1：仅在最后分类层应用）"""
    def __init__(self, in_channels=3, n_classes=2, base_c=16, depths=[2, 2, 2, 2], kernel_size=7,
                 use_reverse_gem=True,
                 bg_enhancement_factor=0.5,
                 vessel_protection_factor=0.1,
                 **kwargs):
        super(RGEM_FSGNet, self).__init__(in_channels, n_classes, base_c, depths, kernel_size)
        
        self.use_reverse_gem = use_reverse_gem
        
        # 反向GEM模块（仅用于最后的分类层）
        if self.use_reverse_gem:
            self.reverse_gem = ReverseGEM(
                bg_enhancement_factor=bg_enhancement_factor,
                vessel_protection_factor=vessel_protection_factor
            )

    def forward(self, x):
        # 复用FSG-Net的完整前向传播，直到最后的输出层
        _, _, h, w = x.size()
        x_scale_2 = F.interpolate(x, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        x_scale_3 = F.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        # 编码器部分（与原FSGNet完全相同）
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

        # 解码器部分（与原FSGNet完全相同）
        _, _, h, w = x3_down.size()
        x3_gf = torch.cat([x3_down, F.interpolate(x4, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x3_gf_conv = self.conv_cat_3(x3_gf)
        x3_small = F.interpolate(x3_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        fgf_out = self.fgf(x3_small, x4, x3_gf_conv, self.attention_block3(x3_small, x4))
        x3_up = self.up_residual_conv3(fgf_out)

        _, _, h, w = x2_down.size()
        x2_gf = torch.cat([x2_down, F.interpolate(x3_gf_conv, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x2_gf_conv = self.conv_cat_2(x2_gf)
        x2_small = F.interpolate(x2_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        fgf_out = self.fgf(x2_small, x3_up, x2_gf_conv, self.attention_block2(x2_small, x3_up))
        x2_up = self.up_residual_conv2(fgf_out)

        _, _, h, w = x1_down.size()
        x1_gf = torch.cat([x1_down, F.interpolate(x2_gf_conv, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x1_gf_conv = self.conv_cat_1(x1_gf)
        x1_small = F.interpolate(x1_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        fgf_out = self.fgf(x1_small, x2_up, x1_gf_conv, self.attention_block1(x1_small, x2_up))
        x1_up = self.up_residual_conv1(fgf_out)

        # 输出层（与原FSGNet相同）
        _, _, h, w = x.size()
        out_3 = F.interpolate(x3_up, size=(h, w), mode='bilinear', align_corners=True)
        out_2 = F.interpolate(x2_up, size=(h, w), mode='bilinear', align_corners=True)
        out_3 = self.output_layer3(out_3)
        out_2 = self.output_layer2(out_2)
        out_1 = self.output_layer1(x1_up)

        # 应用sigmoid激活函数
        out_1 = torch.sigmoid(out_1)
        out_2 = torch.sigmoid(out_2) 
        out_3 = torch.sigmoid(out_3)
        
        # 🔥 关键修改：在最终输出应用反向GEM
        if self.use_reverse_gem:
            out_1 = self.reverse_gem(out_1)
            out_2 = self.reverse_gem(out_2)  # 对所有输出层都应用
            out_3 = self.reverse_gem(out_3)

        return out_1, out_2, out_3