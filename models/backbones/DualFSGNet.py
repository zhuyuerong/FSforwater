import torch
import torch.nn as nn
from .FSGNet import FSGNet  

class DualFSGNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, base_c=64, depths=[3, 3, 9, 3], kernel_size=3, **kwargs):
        super(DualFSGNet, self).__init__()
        
        # 两个独立的FSGNet - PyTorch会自动给它们不同的随机初始化
        self.fsgnet1 = FSGNet(in_channels, n_classes, base_c, depths=depths, kernel_size=kernel_size)
        self.fsgnet2 = FSGNet(in_channels, n_classes, base_c, depths=depths, kernel_size=kernel_size)
        
    def forward(self, x):
        out1_1, out1_2, out1_3 = self.fsgnet1(x)  
        out2_1, out2_2, out2_3 = self.fsgnet2(x)  
        return ((out1_1, out1_2, out1_3), (out2_1, out2_2, out2_3))
