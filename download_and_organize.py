#!/usr/bin/env python3
"""
直接下载并整理数据到FSG-Net项目结构
"""

import kagglehub
import shutil
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

def download_and_organize_data():
    """
    下载数据集并整理到FSG-Net项目结构
    """
    print("🌊 下载并整理水体数据集到FSG-Net结构...")
    
    # 1. 下载数据集
    print("📥 正在下载数据集...")
    download_path = kagglehub.dataset_download("franciscoescobar/satellite-images-of-water-bodies")
    print(f"✅ 数据集下载完成: {download_path}")
    
    # 2. 创建目标目录结构
    base_dir = Path("FSG-Net-pytorch/data/RIVER")
    train_img_dir = base_dir / "train" / "images"
    train_mask_dir = base_dir / "train" / "masks"
    val_img_dir = base_dir / "val" / "images" 
    val_mask_dir = base_dir / "val" / "masks"
    
    # 创建所有目录
    for dir_path in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建目录: {dir_path}")
    
    # 3. 查找原始数据
    source_path = Path(download_path)
    
    # 查找Images和Masks文件夹
    images_folder = None
    masks_folder = None
    
    for item in source_path.rglob("*"):
        if item.is_dir() and item.name.lower() in ["images", "image"]:
            images_folder = item
        elif item.is_dir() and item.name.lower() in ["masks", "mask", "labels"]:
            masks_folder = item
    
    if not images_folder or not masks_folder:
        print("❌ 未找到Images或Masks文件夹")
        print("📂 当前数据结构:")
        for item in source_path.rglob("*"):
            if item.is_dir():
                print(f"   📁 {item.relative_to(source_path)}")
        return False
    
    print(f"📂 找到图像文件夹: {images_folder}")
    print(f"📂 找到掩码文件夹: {masks_folder}")
    
    # 4. 获取所有图像文件
    image_files = []
    for ext in ["*.jpg", "*.png", "*.jpeg", "*.tif", "*.tiff"]:
        image_files.extend(list(images_folder.glob(ext)))
    
    print(f"📊 总共找到 {len(image_files)} 张图像")
    
    # 5. 划分训练集和验证集 (80% train, 20% val)
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    print(f"📊 训练集: {len(train_files)} 张")
    print(f"📊 验证集: {len(val_files)} 张")
    
    # 6. 复制文件到目标位置
    def copy_files(file_list, img_target_dir, mask_target_dir, split_name):
        print(f"📤 正在复制{split_name}集...")
        
        for i, img_file in enumerate(file_list):
            # 复制图像
            img_target = img_target_dir / f"{i:04d}.png"
            shutil.copy2(img_file, img_target)
            
            # 查找对应的掩码文件
            mask_name = img_file.stem  # 不含扩展名的文件名
            mask_file = None
            
            # 尝试不同的掩码文件名和扩展名
            for ext in [".jpg", ".png", ".jpeg", ".tif", ".tiff"]:
                potential_mask = masks_folder / (mask_name + ext)
                if potential_mask.exists():
                    mask_file = potential_mask
                    break
            
            if mask_file and mask_file.exists():
                mask_target = mask_target_dir / f"{i:04d}.png"
                shutil.copy2(mask_file, mask_target)
            else:
                print(f"⚠️  未找到对应掩码: {mask_name}")
            
            if (i + 1) % 100 == 0 or (i + 1) == len(file_list):
                print(f"   进度: {i + 1}/{len(file_list)}")
    
    # 复制训练集
    copy_files(train_files, train_img_dir, train_mask_dir, "训练")
    
    # 复制验证集  
    copy_files(val_files, val_img_dir, val_mask_dir, "验证")
    
    # 7. 验证结果
    print("\n📋 数据整理完成!")
    print("📂 最终目录结构:")
    print(f"   📁 {train_img_dir}: {len(list(train_img_dir.glob('*.png')))} 张图像")
    print(f"   📁 {train_mask_dir}: {len(list(train_mask_dir.glob('*.png')))} 张掩码")  
    print(f"   📁 {val_img_dir}: {len(list(val_img_dir.glob('*.png')))} 张图像")
    print(f"   📁 {val_mask_dir}: {len(list(val_mask_dir.glob('*.png')))} 张掩码")
    
    # 8. 创建配置文件
    config_content = f"""# configs/train_river.yml
{{
  ### Environment Parameters
  debug: false,
  mode: train,
  cuda: true,
  pin_memory: true,
  wandb: true,
  worker: 8,
  log_interval: 9999,
  save_interval: 1,
  saved_model_directory: 'river_models',
  train_fold: 1,
  project_name: 'River-Water-Segmentation',
  CUDA_VISIBLE_DEVICES: '0',

  ### Train Parameters
  model_name: 'FSGNet',
    n_classes: 1,
    in_channels: 3,
    base_c: 64,
    depths: [3, 3, 9, 3],
    kernel_size: 7,
  dataloader: 'Image2Image_resize',
  criterion: 'DiceBCELoss',
  task: 'segmentation',
  input_space: 'RGB',
  input_channel: 3,
  input_size: [512, 512],
  optimizer: 'AdamW',
    lr: 0.001,
    scheduler: 'WarmupCosine',
    cycles: 100,
    warmup_epoch: 10,
    weight_decay: 0.01,
  batch_size: 8,
  epoch: 500,
  ema_decay: 0,
  class_weight: [1.0, 1.0],
  model_path: '',
    freeze_layer: false,

  ### Augmentation Parameters
  transform_blur: true,
  transform_jitter: true,
  transform_hflip: true,
  transform_perspective: false,
  transform_cutmix: true,
  transform_rand_resize: true,
  transform_rand_crop: 256,

  ### 数据路径
  train_x_path: '{train_img_dir}',
  train_y_path: '{train_mask_dir}',
  val_x_path: '{val_img_dir}',
  val_y_path: '{val_mask_dir}',
}}"""
    
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "train_river.yml"
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"\n✅ 配置文件已创建: {config_file}")
    print("\n🚀 现在可以开始训练:")
    print("   python main.py --config_path configs/train_river.yml")
    
    return True

if __name__ == "__main__":
    try:
        download_and_organize_data()
        print("\n🎉 数据准备完成!")
    except Exception as e:
        print(f"\n❌ 出现错误: {e}")
        print("请检查网络连接或重试")