import torch
import torchvision.transforms as transforms
import random
from PIL import ImageFilter

class DualAugmentation:
    def __init__(self, base_transform=None):
        self.base_transform = base_transform
        
        # 强增强策略
        self.strong_aug = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
        ])
        
        # 弱增强策略
        self.weak_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.3),
        ])
    
    def __call__(self, sample):
        image, label = sample
        
        # 应用基础变换
        if self.base_transform:
            image, label = self.base_transform((image, label))
        
        # 生成两个不同增强的版本
        image1 = self.weak_aug(image)
        image2 = self.strong_aug(image)
        
        return (image1, image2), label