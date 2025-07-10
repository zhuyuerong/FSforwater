import torch
import torch.nn as nn
import torch.nn.functional as F
from .FSGNet import *  # 复用FSG-Net的所有模块


class GEM(nn.Module):
    """Graph Edge Module - 图边缘模块"""
    def __init__(self, input_channels=256):
        super(GEM, self).__init__()
        self.input_channels = input_channels
        
        self.edge_aggregation_func = nn.Sequential(
            nn.Linear(4, 1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )
        output_channels = max(1, input_channels // 2)  # 确保至少有1个输出通道
        self.output_channels = output_channels
        self.vertex_update_func = nn.Sequential(
            nn.Linear(2 * input_channels, output_channels),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.edge_update_func = nn.Sequential(
            nn.Linear(2 * input_channels, output_channels),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.update_edge_reduce_func = nn.Sequential(
            nn.Linear(4, 1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )
        self.final_aggregation_layer = nn.Sequential(
            nn.Conv2d(input_channels + output_channels, input_channels, 
                     kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_tensor):
        x = input_tensor
        B, C, H, W = x.size()
        vertex = input_tensor
        
        # Edge计算
        edge = torch.stack(
            (
                torch.cat((input_tensor[:, :, -1:], input_tensor[:, :, :-1]), dim=2),
                torch.cat((input_tensor[:, :, 1:], input_tensor[:, :, :1]), dim=2),
                torch.cat((input_tensor[:, :, :, -1:], input_tensor[:, :, :, :-1]), dim=3),
                torch.cat((input_tensor[:, :, :, 1:], input_tensor[:, :, :, :1]), dim=3)
            ), dim=-1
        ) * input_tensor.unsqueeze(dim=-1)
        
        # Edge聚合
        edge_reshaped = edge.reshape(-1, 4)
        aggregated_edge = self.edge_aggregation_func(edge_reshaped).reshape((B, C, H, W))
        
        # Vertex更新
        cat_feature_for_vertex = torch.cat((vertex, aggregated_edge), dim=1)
        cat_feature_reshaped = cat_feature_for_vertex.permute(0, 2, 3, 1).reshape((-1, 2 * self.input_channels))
        
        vertex_update_output = self.vertex_update_func(cat_feature_reshaped)
        
        update_vertex = vertex_update_output.reshape((B, H, W, self.output_channels)).permute(0, 3, 1, 2)
        
        # Edge更新
        cat_feature_for_edge = torch.cat(
            (
                torch.stack((vertex, vertex, vertex, vertex), dim=-1),
                edge
            ), dim=1
        ).permute(0, 2, 3, 4, 1).reshape((-1, 2 * self.input_channels))
        
        edge_update_output = self.edge_update_func(cat_feature_for_edge)
        
        update_edge = edge_update_output.reshape((B, H, W, 4, self.output_channels)).permute(0, 4, 1, 2, 3).reshape((-1, 4))
        
        update_edge_converted = self.update_edge_reduce_func(update_edge).reshape((B, self.output_channels, H, W))
        
        update_feature = update_vertex * update_edge_converted
        
        final_cat = torch.cat((x, update_feature), dim=1)
        
        output = self.final_aggregation_layer(final_cat)
        return output


class GEM_FSGNet(FSGNet):
    """FSG-Net + GEM模块"""
    def __init__(self, in_channels=3, n_classes=2, base_c=16, depths=[2, 2, 2, 2], kernel_size=7,
                 use_gem=True, 
                 use_gem_encoder=[True, True, True, True],  # 4层编码器
                 use_gem_bottleneck=True,                   # 中间层
                 use_gem_decoder=[True, True, True, True],  # 4层解码器
                 **kwargs):
        super(GEM_FSGNet, self).__init__(in_channels, n_classes, base_c, depths, kernel_size)
        
        self.use_gem = use_gem
        self.use_gem_encoder = use_gem_encoder
        self.use_gem_bottleneck = use_gem_bottleneck  
        self.use_gem_decoder = use_gem_decoder
        
        # 编码器GEM模块
        if self.use_gem and self.use_gem_encoder[0]:
            self.gem_enc1 = GEM(input_channels=base_c * 1)
        if self.use_gem and self.use_gem_encoder[1]:
            self.gem_enc2 = GEM(input_channels=base_c * 2)
        if self.use_gem and self.use_gem_encoder[2]:
            self.gem_enc3 = GEM(input_channels=base_c * 4)
        if self.use_gem and self.use_gem_encoder[3]:
            self.gem_enc4 = GEM(input_channels=base_c * 8)
            
        # 瓶颈层GEM模块
        if self.use_gem and self.use_gem_bottleneck:
            self.gem_bottleneck = GEM(input_channels=base_c * 8)
            
        # 解码器GEM模块  
        if self.use_gem and self.use_gem_decoder[0]:
            self.gem_dec1 = GEM(input_channels=base_c * 4)
        if self.use_gem and self.use_gem_decoder[1]:
            self.gem_dec2 = GEM(input_channels=base_c * 2)
        if self.use_gem and self.use_gem_decoder[2]:
            self.gem_dec3 = GEM(input_channels=base_c * 1)
        if self.use_gem and self.use_gem_decoder[3]:
            self.gem_dec4 = GEM(input_channels=n_classes)

    def forward(self, x):
        # 复用FSG-Net的编码器部分
        _, _, h, w = x.size()
        x_scale_2 = F.interpolate(x, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        x_scale_3 = F.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        # 编码器第1层
        x1 = self.input_layer(x) + self.input_skip(x)
        if self.use_gem and self.use_gem_encoder[0] and hasattr(self, 'gem_enc1'):
            x1 = self.gem_enc1(x1)
        x1_conv = self.conv1(x)
        x1_down = torch.cat([x1_conv, x1], dim=1)

        # 编码器第2层
        x2 = self.down_conv_2(x1_down)
        if self.use_gem and self.use_gem_encoder[1] and hasattr(self, 'gem_enc2'):
            x2 = self.gem_enc2(x2)
        x2_conv = self.conv2(x_scale_2)
        x2_down = torch.cat([x2_conv, x2], dim=1)

        # 编码器第3层
        x3 = self.down_conv_3(x2_down)
        if self.use_gem and self.use_gem_encoder[2] and hasattr(self, 'gem_enc3'):
            x3 = self.gem_enc3(x3)
        x3_conv = self.conv3(x_scale_3)
        x3_down = torch.cat([x3_conv, x3], dim=1)

        # 编码器第4层
        x4 = self.down_conv_4(x3_down)
        if self.use_gem and self.use_gem_encoder[3] and hasattr(self, 'gem_enc4'):
            x4 = self.gem_enc4(x4)
        x4 = self.attn(x4)

        # 瓶颈层GEM
        if self.use_gem and self.use_gem_bottleneck and hasattr(self, 'gem_bottleneck'):
            x4 = self.gem_bottleneck(x4)

        # 复用FSG-Net的解码器部分
        _, _, h, w = x3_down.size()
        x3_gf = torch.cat([x3_down, F.interpolate(x4, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x3_gf_conv = self.conv_cat_3(x3_gf)
        x3_small = F.interpolate(x3_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        fgf_out = self.fgf(x3_small, x4, x3_gf_conv, self.attention_block3(x3_small, x4))
        x3_up = self.up_residual_conv3(fgf_out)
        # 解码器第1层GEM
        if self.use_gem and self.use_gem_decoder[0] and hasattr(self, 'gem_dec1'):
            x3_up = self.gem_dec1(x3_up)

        _, _, h, w = x2_down.size()
        x2_gf = torch.cat([x2_down, F.interpolate(x3_gf_conv, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x2_gf_conv = self.conv_cat_2(x2_gf)
        x2_small = F.interpolate(x2_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        fgf_out = self.fgf(x2_small, x3_up, x2_gf_conv, self.attention_block2(x2_small, x3_up))
        x2_up = self.up_residual_conv2(fgf_out)
        # 解码器第2层GEM
        if self.use_gem and self.use_gem_decoder[1] and hasattr(self, 'gem_dec2'):
            x2_up = self.gem_dec2(x2_up)

        _, _, h, w = x1_down.size()
        x1_gf = torch.cat([x1_down, F.interpolate(x2_gf_conv, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x1_gf_conv = self.conv_cat_1(x1_gf)
        x1_small = F.interpolate(x1_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        fgf_out = self.fgf(x1_small, x2_up, x1_gf_conv, self.attention_block1(x1_small, x2_up))
        x1_up = self.up_residual_conv1(fgf_out)
        # 解码器第3层GEM
        if self.use_gem and self.use_gem_decoder[2] and hasattr(self, 'gem_dec3'):
            x1_up = self.gem_dec3(x1_up)

        # 复用FSG-Net的输出层
        _, _, h, w = x.size()
        out_3 = F.interpolate(x3_up, size=(h, w), mode='bilinear', align_corners=True)
        out_2 = F.interpolate(x2_up, size=(h, w), mode='bilinear', align_corners=True)
        out_3 = self.output_layer3(out_3)
        out_2 = self.output_layer2(out_2)
        out_1 = self.output_layer1(x1_up)

        # 解码器第4层GEM（最终输出前）
        if self.use_gem and self.use_gem_decoder[3] and hasattr(self, 'gem_dec4'):
            print(f"🔍 Applying gem_dec4...")
            out_1_before_gem = out_1.clone()
            out_1 = self.gem_dec4(out_1)
            print(f"  out_1 after gem_dec4: [{out_1.min().item():.6f}, {out_1.max().item():.6f}], mean: {out_1.mean().item():.6f}")
            print(f"  change: {(out_1 - out_1_before_gem).abs().mean().item():.6f}")

        # # 对于二分类分割任务，应用sigmoid激活函数
        # out_1 = torch.sigmoid(out_1)
        # out_2 = torch.sigmoid(out_2) 
        # out_3 = torch.sigmoid(out_3)

        return out_1, out_2, out_3