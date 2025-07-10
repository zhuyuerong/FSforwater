import torch
import torch.nn as nn
import torch.nn.functional as F
from .FSGNet import *  # 导入所有FSGNet组件


class RegionGrowingModule(nn.Module):
    """可微分的区域生长模块，集成到上采样过程中"""
    
    def __init__(self, in_channels, mu_f=0.0789, sigma_f=0.0774, alpha=1.0):
        super(RegionGrowingModule, self).__init__()
        
        # 区域生长参数
        self.register_parameter('mu_f', nn.Parameter(torch.tensor(mu_f, dtype=torch.float32)))
        self.register_parameter('sigma_f', nn.Parameter(torch.tensor(sigma_f, dtype=torch.float32)))
        self.register_parameter('alpha', nn.Parameter(torch.tensor(alpha, dtype=torch.float32)))
        
        # Sobel卷积核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # 膨胀核
        self.register_buffer('dilation_kernel', torch.ones(1, 1, 3, 3, dtype=torch.float32))
        
        # 特征增强卷积
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.7))
    
    def extract_green_channel(self, x):
        """从特征图中提取类似绿色通道的信息"""
        if x.shape[1] == 3:  # 如果是RGB图像
            return x[:, 1:2, :, :]  # 直接取绿色通道
        else:  # 如果是特征图
            return torch.mean(x, dim=1, keepdim=True)
    
    def calculate_gradient(self, x):
        """计算梯度幅值"""
        green_like = self.extract_green_channel(x)
        grad_x = F.conv2d(green_like, self.sobel_x, padding=1)
        grad_y = F.conv2d(green_like, self.sobel_y, padding=1)
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return gradient_magnitude
    
    def soft_region_growing(self, features, gradient_magnitude):
        """软区域生长"""
        B, C, H, W = features.shape
        
        # 计算血管阈值
        vessel_threshold = self.mu_f + self.alpha * self.sigma_f
        vessel_candidates = torch.sigmoid((gradient_magnitude - vessel_threshold) * 10.0)
        
        # 使用特征图的激活作为种子
        feature_activation = torch.mean(features, dim=1, keepdim=True)
        seed_threshold = feature_activation.mean(dim=[2, 3], keepdim=True) + \
                        0.5 * feature_activation.std(dim=[2, 3], keepdim=True)
        seeds = torch.sigmoid((feature_activation - seed_threshold) * 10.0)
        
        # 迭代软膨胀
        grown_mask = seeds
        for _ in range(2):
            dilated = F.conv2d(grown_mask, self.dilation_kernel, padding=1)
            dilated = torch.clamp(dilated, 0, 1)
            new_grown = dilated * vessel_candidates
            grown_mask = torch.max(grown_mask, new_grown * 0.5)
        
        return grown_mask
    
    def forward(self, features, original_image=None):
        """
        Args:
            features: [B, C, H, W] 上采样的特征图
            original_image: [B, 3, H', W'] 原始输入图像（可选）
        Returns:
            enhanced_features: [B, C, H, W] 增强后的特征图
        """
        # 增强特征
        enhanced = self.enhance_conv(features)
        
        # 计算梯度
        if original_image is not None:
            upsampled_image = F.interpolate(original_image, size=features.shape[2:], mode='bilinear', align_corners=True)
            gradient_magnitude = self.calculate_gradient(upsampled_image)
        else:
            gradient_magnitude = self.calculate_gradient(features)
        
        # 软区域生长
        grown_mask = self.soft_region_growing(enhanced, gradient_magnitude)
        
        # 扩展到所有通道并融合
        grown_mask_expanded = grown_mask.expand(-1, features.shape[1], -1, -1)
        fusion_weight = torch.sigmoid(self.fusion_weight)
        final_features = fusion_weight * enhanced + (1 - fusion_weight) * (enhanced * grown_mask_expanded)
        
        return final_features


class RG_FSGNet(nn.Module):
    """集成区域生长的FSGNet - 在上采样过程中应用RG"""
    
    def __init__(self, channel, n_classes, base_c, depths, kernel_size, 
                 enable_rg=True, mu_f=0.0789, sigma_f=0.0774, alpha=1.0):
        super(RG_FSGNet, self).__init__()  # 🔥 修复：类名要一致！

        self.enable_rg = enable_rg
        
        # 复制原始FSGNet的所有组件
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

        # 【新增】区域生长模块到上采样路径
        if self.enable_rg:
            self.rg_module_3 = RegionGrowingModule(base_c * 4, mu_f, sigma_f, alpha)
            self.rg_module_2 = RegionGrowingModule(base_c * 2, mu_f, sigma_f, alpha)
            self.rg_module_1 = RegionGrowingModule(base_c * 1, mu_f, sigma_f, alpha)

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

    def forward(self, x):
        # 保存原始输入用于区域生长
        original_input = x
        
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

        # Decoder with integrated region growing
        # 第一轮上采样
        _, _, h, w = x3_down.size()
        x3_gf = torch.cat([x3_down, F.interpolate(x4, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x3_gf_conv = self.conv_cat_3(x3_gf)
        x3_small = F.interpolate(x3_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        fgf_out = self.fgf(x3_small, x4, x3_gf_conv, self.attention_block3(x3_small, x4))
        x3_up = self.up_residual_conv3(fgf_out)
        
        # 🔥 第一次区域生长：在第一轮上采样后应用
        if self.enable_rg:
            x3_up = self.rg_module_3(x3_up, original_input)

        # 第二轮上采样
        _, _, h, w = x2_down.size()
        x2_gf = torch.cat([x2_down, F.interpolate(x3_gf_conv, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x2_gf_conv = self.conv_cat_2(x2_gf)
        x2_small = F.interpolate(x2_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        fgf_out = self.fgf(x2_small, x3_up, x2_gf_conv, self.attention_block2(x2_small, x3_up))
        x2_up = self.up_residual_conv2(fgf_out)
        
        # 🔥 第二次区域生长：在第二轮上采样后应用（恢复！）
        if self.enable_rg:
            x2_up = self.rg_module_2(x2_up, original_input)

        # 第三轮上采样
        _, _, h, w = x1_down.size()
        x1_gf = torch.cat([x1_down, F.interpolate(x2_gf_conv, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x1_gf_conv = self.conv_cat_1(x1_gf)
        x1_small = F.interpolate(x1_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        fgf_out = self.fgf(x1_small, x2_up, x1_gf_conv, self.attention_block1(x1_small, x2_up))
        x1_up = self.up_residual_conv1(fgf_out)
        
        # 🔥 第三次区域生长：在第三轮上采样后应用（恢复！）
        if self.enable_rg:
            x1_up = self.rg_module_1(x1_up, original_input)

        # 输出层
        _, _, h, w = original_input.size()
        out_3 = F.interpolate(x3_up, size=(h, w), mode='bilinear', align_corners=True)
        out_2 = F.interpolate(x2_up, size=(h, w), mode='bilinear', align_corners=True)
        out_3 = self.output_layer3(out_3)
        out_2 = self.output_layer2(out_2)
        out_1 = self.output_layer1(x1_up)

        return out_1, out_2, out_3

