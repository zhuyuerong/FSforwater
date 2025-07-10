#!/usr/bin/env python3
"""
快速创建小样本测试
"""

import os
import shutil
import yaml
from pathlib import Path
import random

def create_small_test():
    """创建30张图片的快速测试"""
    
    # 源路径
    source_images = Path("data/RIVER/train/images")
    source_masks = Path("data/RIVER/train/masks")
    
    # 检查源路径
    if not source_images.exists():
        print(f"❌ 源图像路径不存在: {source_images}")
        return False
    
    # 创建小测试目录
    test_dir = Path("quick_test_data")
    for subdir in ['train/images', 'train/masks', 'val/images', 'val/masks']:
        (test_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # 随机选择30张图片
    image_files = list(source_images.glob("*.png"))
    if len(image_files) < 30:
        print(f"⚠️  只有{len(image_files)}张图片，全部使用")
        selected = image_files
    else:
        random.seed(42)
        selected = random.sample(image_files, 30)
    
    # 分配训练集和验证集
    train_files = selected[:24]  # 24张训练
    val_files = selected[24:]    # 6张验证
    
    # 复制文件
    print("📁 复制训练集...")
    for img_file in train_files:
        shutil.copy2(img_file, test_dir / 'train/images')
        mask_file = source_masks / img_file.name
        if mask_file.exists():
            shutil.copy2(mask_file, test_dir / 'train/masks')
    
    print("📁 复制验证集...")
    for img_file in val_files:
        shutil.copy2(img_file, test_dir / 'val/images')
        mask_file = source_masks / img_file.name
        if mask_file.exists():
            shutil.copy2(mask_file, test_dir / 'val/masks')
    
    print(f"✅ 小样本数据集创建完成: {test_dir}")
    return test_dir

def create_test_config():
    """创建测试配置"""
    
    config = {
        'debug': False,
        'mode': 'train',
        'cuda': True,
        'pin_memory': True,
        'wandb': False,  # 关闭wandb加快测试
        'worker': 4,
        'log_interval': 9999,
        'save_interval': 1,
        'saved_model_directory': 'quick_test_models',
        'train_fold': 1,
        'project_name': 'QuickTest',
        'CUDA_VISIBLE_DEVICES': '0',
        
        # 关键参数
        'model_name': 'FSGNet',
        'n_classes': 1,
        'in_channels': 3,
        'base_c': 64,
        'depths': [3, 3, 9, 3],
        'kernel_size': 11,  # 增大卷积核
        
        'dataloader': 'Image2Image_resize',
        'criterion': 'FocalDiceLoss',  # 使用Focal Loss
        'task': 'segmentation',
        'input_space': 'RGB',
        'input_channel': 3,
        'input_size': [512, 512],
        
        'optimizer': 'AdamW',
        'lr': 0.0005,  # 降低学习率
        'scheduler': 'WarmupCosine',
        'cycles': 100,
        'warmup_epoch': 5,
        'weight_decay': 0.01,
        
        'batch_size': 2,  # 小批次
        'epoch': 30,      # 快速测试
        'ema_decay': 0,
        'class_weight': [0.3, 0.7],  # 假设水体较少
        'model_path': '',
        'freeze_layer': False,
        
        # 数据增强
        'transform_blur': True,
        'transform_jitter': True,
        'transform_hflip': True,
        'transform_perspective': False,
        'transform_cutmix': True,
        'transform_rand_resize': True,
        'transform_rand_crop': 256,
        
        # 数据路径
        'train_x_path': 'quick_test_data/train/images',
        'train_y_path': 'quick_test_data/train/masks',
        'val_x_path': 'quick_test_data/val/images',
        'val_y_path': 'quick_test_data/val/masks',
    }
    
    # 保存配置
    Path("configs").mkdir(exist_ok=True)
    config_file = Path("configs/quick_test.yml")
    
    with open(config_file, 'w') as f:
        f.write("{\n")
        for key, value in config.items():
            if isinstance(value, str):
                f.write(f"  {key}: '{value}',\n")
            elif isinstance(value, list):
                f.write(f"  {key}: {value},\n")
            else:
                f.write(f"  {key}: {value},\n")
        f.write("}")
    
    print(f"✅ 测试配置已创建: {config_file}")
    return config_file

if __name__ == "__main__":
    print("🧪 创建快速测试环境...")
    
    # 创建小样本数据集
    test_dir = create_small_test()
    if not test_dir:
        exit(1)
    
    # 创建测试配置
    config_file = create_test_config()
    
    print("\n🚀 快速测试准备完成!")
    print("📋 下一步:")
    print("   1. 先分析数据: python dataset_analysis.py")
    print("   2. 开始测试: python main.py --config_path configs/quick_test.yml")

