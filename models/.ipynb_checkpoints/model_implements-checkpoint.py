import torch.nn as nn
import torch

from models.backbones import Unet_part
from models.backbones import UNeTPluss
from models.backbones import ResUNet as ResUNets
from models.backbones import SAUNet as SAUNets
from models.backbones import DCSAUUNet
from models.backbones import AGNet as AGNet_parts
from models.backbones import ConvUNeXt as ConvUNeXt_parts
from models.backbones import R2UNet as R2UNet_parts
from models.backbones import FRUNet as FRUNet_parts
from models.backbones import FSGNet as FSGNet_parts
from models.backbones import RG_FSG 
from models.backbones import GEM_FSG
from models.backbones import RGEM_FSG  # 🔥 新增导入
from models.backbones import RG_UNet
from models.backbones import DualFSGNet as DualFSGNet_parts
from models.backbones import FSGNet_InputGuided as FSGNet_InputGuided_parts  # 新增
from models.backbones import FSGNet_EncoderGuided as FSGNet_EncoderGuided_parts  # 新增

# 在导入部分添加：
from models.backbones import FSGNet_InputGuided_v2 as FSGNet_InputGuided_v2_parts  # 新增

# 在类定义部分添加：
class FSGNet_InputGuided_v2(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, depths=[3, 3, 9, 3], base_c=64, kernel_size=3, **kwargs):
        super().__init__()
        self.FSGNet_InputGuided_v2 = FSGNet_InputGuided_v2_parts.FSGNet_InputGuided_v2(
            in_channels, n_classes, base_c, depths=depths, kernel_size=kernel_size)

    def forward(self, x):
        return self.FSGNet_InputGuided_v2(x)
class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, bilinear=True, **kwargs):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = Unet_part.DoubleConv(in_channels, 64)
        self.down1 = Unet_part.Down(64, 128)
        self.down2 = Unet_part.Down(128, 256)
        self.down3 = Unet_part.Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Unet_part.Down(512, 1024 // factor)
        self.up1 = Unet_part.Up(1024, 512 // factor, bilinear)
        self.up2 = Unet_part.Up(512, 256 // factor, bilinear)
        self.up3 = Unet_part.Up(256, 128 // factor, bilinear)
        self.up4 = Unet_part.Up(128, 64, bilinear)
        self.outc = Unet_part.OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return torch.sigmoid(logits)


class UNet2P(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, **kwargs):
        super().__init__()
        self.unet2p = UNeTPluss.UNet_2Plus(in_channels=in_channels, n_classes=n_classes)

    def forward(self, x):
        return self.unet2p(x)


class UNet3P_Deep(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, **kwargs):
        super().__init__()
        self.unet3p = UNeTPluss.UNet_3Plus_DeepSup(in_channels=in_channels, n_classes=n_classes)

    def forward(self, x):
        return self.unet3p(x)


class ResUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, **kwargs):
        super().__init__()
        self.resunet = ResUNets.ResUnet(channel=in_channels, n_classes=n_classes)

    def forward(self, x):
        return self.resunet(x)


class ResUNet2P(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, **kwargs):
        super().__init__()
        self.resunet2p = ResUNets.ResUnetPlusPlus(channel=in_channels, n_classes=n_classes)

    def forward(self, x):
        return self.resunet2p(x)


class SAUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, base_c=16, **kwargs):
        super().__init__()
        self.sa_unet = SAUNets.SA_UNet(in_channels=in_channels, num_classes=n_classes, base_c=base_c)

    def forward(self, x):
        return self.sa_unet(x)


class DCSAU_UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, **kwargs):
        super().__init__()
        self.dcsau_unet = DCSAUUNet.DCSAU_UNet(img_channels=in_channels, n_classes=n_classes)

    def forward(self, x):
        return torch.sigmoid(self.dcsau_unet(x))


class AGNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, **kwargs):
        super().__init__()
        self.ag_net = AGNet_parts.AG_Net(in_channels=in_channels, n_classes=n_classes)

    def forward(self, x):
        out = [torch.sigmoid(item) for item in self.ag_net(x)]
        return out


class ATTUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, **kwargs):
        super().__init__()
        self.attu_net = R2UNet_parts.AttU_Net(img_ch=in_channels, output_ch=n_classes)

    def forward(self, x):
        return torch.sigmoid(self.attu_net(x))


class R2UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, **kwargs):
        super().__init__()
        self.r2unet = R2UNet_parts.R2U_Net(img_ch=in_channels, output_ch=n_classes)

    def forward(self, x):
        return torch.sigmoid(self.r2unet(x))


class ConvUNeXt(nn.Module):
    def __init__(self, in_channels, n_classes, base_c=32, **kwargs):
        super().__init__()
        self.convunext = ConvUNeXt_parts.ConvUNeXt(in_channels=in_channels, num_classes=n_classes, base_c=base_c)

    def forward(self, x):
        out = self.convunext(x)
        out = out['out']

        return torch.sigmoid(out)


class FRUNet(nn.Module):
    def __init__(self, in_channels, n_classes, **kwargs):
        super().__init__()
        self.frunet = FRUNet_parts.FR_UNet(num_channels=in_channels, num_classes=n_classes)

    def forward(self, x):
        out = self.frunet(x)

        return torch.sigmoid(out)


class FSGNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 n_classes=1,
                 depths=[3, 3, 9, 3],
                 base_c=64,
                 kernel_size=3,
                 **kwargs):
        super().__init__()
        self.FSGNet = FSGNet_parts.FSGNet(in_channels, n_classes, base_c,
                                          depths=depths, kernel_size=kernel_size)

    def forward(self, x):
        return self.FSGNet(x)


class RG_FSGNet(nn.Module):
    """带集成区域生长的FSGNet - 在上采样过程中应用RG"""
    def __init__(self, **kwargs):
        super().__init__()
        
        # 从kwargs中提取参数，设置默认值
        channel = kwargs.get('in_channels', 3)
        n_classes = kwargs.get('n_classes', 1)
        base_c = kwargs.get('base_c', 64)
        depths = kwargs.get('depths', [3, 3, 9, 3])
        kernel_size = kwargs.get('kernel_size', 3)
        enable_rg = kwargs.get('enable_rg', True)
        mu_f = kwargs.get('mu_f', 0.0789)
        sigma_f = kwargs.get('sigma_f', 0.0774)
        alpha = kwargs.get('alpha', 1.0)
        
        # 创建 RG_FSGNet 实例
        self.rg_fsgnet = RG_FSG.RG_FSGNet(
            channel=channel,
            n_classes=n_classes, 
            base_c=base_c,
            depths=depths, 
            kernel_size=kernel_size,
            enable_rg=enable_rg,
            mu_f=mu_f,
            sigma_f=sigma_f,
            alpha=alpha
        )

    def forward(self, x):
        return self.rg_fsgnet(x)


class GEM_FSGNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, depths=[3, 3, 9, 3], base_c=64, kernel_size=3, **kwargs):
        super().__init__()
        self.gem_fsgnet = GEM_FSG.GEM_FSGNet(
            channel=in_channels, 
            n_classes=n_classes, 
            base_c=base_c,
            depths=depths, 
            kernel_size=kernel_size,
            **kwargs
        )

    def forward(self, x):
        return self.gem_fsgnet(x)


# 🔥 新增 RGEM_FSGNet 类
class RGEM_FSGNet(nn.Module):
    """反向GEM-FSGNet - 实验1：仅在最后分类层应用反向GEM"""
    def __init__(self, in_channels=3, n_classes=1, depths=[3, 3, 9, 3], base_c=64, kernel_size=3, **kwargs):
        super().__init__()
        
        # 提取反向GEM参数，设置默认值
        use_reverse_gem = kwargs.get('use_reverse_gem', True)
        bg_enhancement_factor = kwargs.get('bg_enhancement_factor', 0.5)
        vessel_protection_factor = kwargs.get('vessel_protection_factor', 0.1)
        
        self.rgem_fsgnet = RGEM_FSG.RGEM_FSGNet(
            in_channels=in_channels,
            n_classes=n_classes,
            base_c=base_c,
            depths=depths,
            kernel_size=kernel_size,
            use_reverse_gem=use_reverse_gem,
            bg_enhancement_factor=bg_enhancement_factor,
            vessel_protection_factor=vessel_protection_factor,
            **kwargs
        )

    def forward(self, x):
        return self.rgem_fsgnet(x)


class RG_UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, bilinear=True, **kwargs):
        super().__init__()
        self.rg_unet = RG_UNet.RG_UNet(
            in_channels=in_channels,
            n_classes=n_classes,
            bilinear=bilinear,
            **kwargs
        )

    def forward(self, x):
        return self.rg_unet(x)

class DualFSGNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, **kwargs):
        super().__init__()
        self.dual_fsgnet = DualFSGNet_parts.DualFSGNet(in_channels=in_channels, n_classes=n_classes, **kwargs)

    def forward(self, x):
        return self.dual_fsgnet(x)
    
class DualUNet2P(nn.Module):
    """UNet2P + UNet2P 双网络架构"""
    def __init__(self, in_channels=3, n_classes=1, **kwargs):
        super(DualUNet2P, self).__init__()
        
        # 两个独立的UNet2P
        self.unet2p_1 = UNeTPluss.UNet_2Plus(in_channels=in_channels, n_classes=n_classes)
        self.unet2p_2 = UNeTPluss.UNet_2Plus(in_channels=in_channels, n_classes=n_classes)
    
    def forward(self, x):
        # 两个UNet2P的输出
        unet2p1_out = self.unet2p_1(x)
        unet2p2_out = self.unet2p_2(x)
        
        return unet2p1_out, unet2p2_out

class FSGNet_InputGuided(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, depths=[3, 3, 9, 3], base_c=64, kernel_size=3, **kwargs):
        super().__init__()
        self.FSGNet_InputGuided = FSGNet_InputGuided_parts.FSGNet_InputGuided(
            in_channels, n_classes, base_c, depths=depths, kernel_size=kernel_size)

    def forward(self, x):
        return self.FSGNet_InputGuided(x)


class FSGNet_EncoderGuided(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, depths=[3, 3, 9, 3], base_c=64, kernel_size=3, **kwargs):
        super().__init__()
        self.FSGNet_EncoderGuided = FSGNet_EncoderGuided_parts.FSGNet_EncoderGuided(
            in_channels, n_classes, base_c, depths=depths, kernel_size=kernel_size)

    def forward(self, x):
        return self.FSGNet_EncoderGuided(x)