#!/usr/bin/env python3
"""
深度分析水体数据集的特性、难点和挑战
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict

class WaterBodyDatasetAnalyzer:
    def __init__(self, images_path, masks_path):
        self.images_path = Path(images_path)
        self.masks_path = Path(masks_path)
        self.analysis_results = {}
        
    def analyze_class_imbalance(self, image_files):
        """分析类别不平衡 - 最关键的分析"""
        water_ratios = []
        
        print("🔍 分析类别平衡...")
        for i, img_file in enumerate(image_files):
            mask_file = self.masks_path / img_file.name
            if not mask_file.exists():
                continue
                
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            water_pixels = np.sum(mask > 127)
            total_pixels = mask.size
            water_ratio = water_pixels / total_pixels
            water_ratios.append(water_ratio)
            
            if i % 10 == 0:
                print(f"   处理进度: {i+1}/{len(image_files)}")
        
        avg_ratio = np.mean(water_ratios)
        print(f"✅ 平均水体占比: {avg_ratio:.1%}")
        print(f"✅ 背景占比: {1-avg_ratio:.1%}")
        print(f"✅ 不平衡比例: {min(avg_ratio, 1-avg_ratio)/max(avg_ratio, 1-avg_ratio):.3f}")
        
        return {
            'avg_water_ratio': avg_ratio,
            'background_ratio': 1 - avg_ratio,
            'water_ratios': water_ratios,
            'min_ratio': min(water_ratios),
            'max_ratio': max(water_ratios)
        }
    
    def analyze_water_fragmentation(self, image_files):
        """分析水体碎片化程度"""
        fragment_counts = []
        
        print("🔍 分析水体碎片化...")
        for i, img_file in enumerate(image_files[:20]):  # 限制样本数
            mask_file = self.masks_path / img_file.name
            if not mask_file.exists():
                continue
                
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            num_labels = cv2.connectedComponents(binary)[0] - 1  # 减去背景
            fragment_counts.append(max(0, num_labels))
        
        avg_fragments = np.mean(fragment_counts) if fragment_counts else 0
        print(f"✅ 平均水体碎片数: {avg_fragments:.1f}")
        
        return {
            'avg_fragments': avg_fragments,
            'fragment_counts': fragment_counts
        }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="快速数据集分析")
    parser.add_argument("--images", default="data/RIVER/train/images", help="图像路径")
    parser.add_argument("--masks", default="data/RIVER/train/masks", help="掩码路径")
    parser.add_argument("--samples", type=int, default=30, help="分析样本数")
    
    args = parser.parse_args()
    
    print("🔍 开始快速数据集分析...")
    analyzer = WaterBodyDatasetAnalyzer(args.images, args.masks)
    
    # 获取样本文件
    image_files = list(Path(args.images).glob("*.png"))[:args.samples]
    print(f"📊 分析 {len(image_files)} 个样本")
    
    if len(image_files) == 0:
        print(f"❌ 在 {args.images} 中未找到PNG图像")
        return
    
    # 类别不平衡分析
    imbalance_result = analyzer.analyze_class_imbalance(image_files)
    
    # 碎片化分析
    fragment_result = analyzer.analyze_water_fragmentation(image_files)
    
    # 生成建议
    print("\n" + "="*50)
    print("💡 关键发现和建议")
    print("="*50)
    
    water_ratio = imbalance_result['avg_water_ratio']
    
    if water_ratio < 0.1:
        print("🚨 严重类别不平衡！水体占比过小")
        print("   建议使用 FocalDiceLoss")
        print(f"   建议类别权重: [{0.1:.1f}, {0.9:.1f}]")
    elif water_ratio < 0.3:
        print("⚠️  中等类别不平衡")
        print("   建议使用 FocalDiceLoss 或调整类别权重")
        print(f"   建议类别权重: [{1-water_ratio:.1f}, {water_ratio:.1f}]")
    else:
        print("✅ 类别相对平衡")
    
    if fragment_result['avg_fragments'] > 5:
        print("🔧 水体碎片化严重，建议:")
        print("   - 增大kernel_size到11或更大")
        print("   - 使用更强的数据增强")
    
    print("="*50)

if __name__ == "__main__":
    main()
