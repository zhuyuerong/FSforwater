import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from skimage.filters import gabor_kernel
import cv2
from models.backbones import Unet_part


class RegionGrowingModule(nn.Module):
    """区域生长模块"""
    def __init__(self, mu_f=0.0789, sigma_f=0.0774, alpha=1.0):
        super(RegionGrowingModule, self).__init__()
        self.mu_f = mu_f
        self.sigma_f = sigma_f
        self.alpha = alpha
    
    def get_seed_points(self, image):
        """获取种子点 - 使用四方向线检测"""
        if isinstance(image, torch.Tensor):
            image_np = image.squeeze().cpu().numpy()
        else:
            image_np = image
        
        angles = [45, 90, 135, 180]
        seed_points = []
        
        for angle in angles:
            kernel = gabor_kernel(frequency=0.1, theta=np.radians(angle))
            filtered = ndimage.convolve(image_np, kernel.real)
            local_maxima = (filtered == ndimage.maximum_filter(filtered, size=3))
            points = np.where(local_maxima & (filtered > np.mean(filtered)))
            seed_points.extend(list(zip(points[0], points[1])))
        
        return list(set(seed_points))
    
    def calculate_gradient(self, image):
        """计算梯度"""
        if isinstance(image, torch.Tensor):
            image_np = image.squeeze().cpu().numpy()
        else:
            image_np = image
        
        grad_x = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        return gradient
    
    def region_growing(self, image, seed_points):
        """区域生长算法"""
        if isinstance(image, torch.Tensor):
            image_np = image.squeeze().cpu().numpy()
        else:
            image_np = image
        
        h, w = image_np.shape
        segmented = np.zeros((h, w), dtype=np.uint8)
        gradient = self.calculate_gradient(image_np)
        
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                    (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for seed_y, seed_x in seed_points:
            if segmented[seed_y, seed_x] == 1:
                continue
                
            queue = [(seed_y, seed_x)]
            visited = set()
            
            while queue:
                y, x = queue.pop(0)
                
                if (y, x) in visited or y < 0 or y >= h or x < 0 or x >= w:
                    continue
                    
                visited.add((y, x))
                rho = gradient[y, x]
                
                # 根据论文公式判断: μf + ασf ≤ ρ (血管) vs μf ≥ ρ (背景)
                if self.mu_f + self.alpha * self.sigma_f <= rho:
                    segmented[y, x] = 1
                    
                    for dy, dx in neighbors:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < h and 0 <= nx < w and 
                            (ny, nx) not in visited and segmented[ny, nx] == 0):
                            queue.append((ny, nx))
        
        return segmented
    
    def forward(self, unet_output, original_image):
        """前向传播"""
        device = unet_output.device
        batch_size = unet_output.size(0)
        refined_outputs = []
        
        for i in range(batch_size):
            unet_pred = unet_output[i].squeeze().cpu().numpy()
            orig_img = original_image[i].squeeze().cpu().numpy()
            
            # 预处理：绿色通道提取和归一化（如果是彩色图像）
            if len(orig_img.shape) == 3:
                orig_img = orig_img[1]  # 绿色通道
            
            # 归一化
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
            
            # 背景减法增强
            background = cv2.GaussianBlur(orig_img, (101, 101), 0)
            enhanced_img = orig_img - background
            enhanced_img = np.clip(enhanced_img, 0, 1)
            
            # 获取种子点
            seed_points = self.get_seed_points(enhanced_img)
            
            # 区域生长
            region_grown = self.region_growing(enhanced_img, seed_points)
            
            # 与UNet结果结合
            # 判断是否需要sigmoid（检查数值范围）
            if unet_pred.max() > 1.0 or unet_pred.min() < 0.0:
                # 需要sigmoid
                unet_pred = 1 / (1 + np.exp(-unet_pred))
            
            combined = np.logical_or(unet_pred > 0.5, region_grown).astype(np.float32)
            
            refined_outputs.append(torch.tensor(combined).unsqueeze(0))
        
        refined_output = torch.stack(refined_outputs).to(device)
        return refined_output


class RG_UNet(nn.Module):
    """UNet + 区域生长模块"""
    def __init__(self, in_channels=3, n_classes=2, bilinear=True, 
                 mu_f=0.0789, sigma_f=0.0774, alpha=1.0, **kwargs):
        super(RG_UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 标准UNet结构
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
        
        # 区域生长模块
        self.region_growing = RegionGrowingModule(mu_f, sigma_f, alpha)

    def forward(self, x):
        # 保存原始输入用于区域生长
        original_input = x.clone()
        
        # UNet前向传播
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_up = self.up1(x5, x4)
        x_up = self.up2(x_up, x3)
        x_up = self.up3(x_up, x2)
        x_up = self.up4(x_up, x1)
        logits = self.outc(x_up)

        # 应用区域生长模块细化输出
        refined_output = self.region_growing(logits, original_input)
        
        # 对细化后的结果应用sigmoid
        refined_output = torch.sigmoid(refined_output)
        
        return refined_output