#!/usr/bin/env python3
"""
GEMFSG vs FSG 血管分割对比脚本
- GEMFSG: 背景预测 → 反转得到血管
- FSG: 直接血管预测
- 对比两种方法的差异并可视化
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

# 导入模型
from models.backbones.FSGNet import FSGNet
from models.backbones.GEM_FSG import GEM_FSGNet  # GEMFSG模型
import torchvision.transforms.functional as tf


class ModelComparator:
    """GEMFSG vs FSG 模型对比器"""
    
    def __init__(self, device='cuda:0'):
        self.device = device
        self.gemfsg_model = None
        self.fsg_model = None
        
    def load_gemfsg_model(self, checkpoint_path):
        """加载GEMFSG背景预测模型"""
        print(f"Loading GEMFSG model from: {checkpoint_path}")
        
        # 尝试不同的GEMFSG配置
        gemfsg_configs = [
            # 配置1: 只使用background_learning参数
            {"use_background_learning": True},
            # 配置2: 完整参数
            {"channel": 3, "n_classes": 2, "base_c": 32, "depths": [3, 3, 9, 3], "kernel_size": 3, "use_background_learning": True},
            # 配置3: 无参数
            {},
        ]
        
        # 加载权重以备后用
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 处理DataParallel权重
        if any(key.startswith('module.') for key in state_dict.keys()):
            print("🔧 Detected DataParallel weights, removing 'module.' prefix...")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        # 尝试不同配置
        for i, config in enumerate(gemfsg_configs):
            try:
                print(f"🔄 Trying GEMFSG config {i+1}: {config}")
                self.gemfsg_model = GEM_FSGNet(**config)
                self.gemfsg_model = self.gemfsg_model.to(self.device)
                self.gemfsg_model.load_state_dict(state_dict)
                self.gemfsg_model.eval()
                print("✅ GEMFSG model loaded successfully")
                return
            except Exception as e:
                print(f"❌ Config {i+1} failed: {e}")
                continue
                
        raise RuntimeError("❌ All GEMFSG configurations failed. Please check the model definition and weights.")
        
    def load_fsg_model(self, checkpoint_path):
        """加载FSG血管预测模型"""
        print(f"Loading FSG model from: {checkpoint_path}")
        
        # 🎯 根据错误日志分析出的正确配置
        # 从权重文件的tensor尺寸可以推断出真实的模型配置
        fsg_configs = [
            # 配置1: 根据权重文件推断的真实配置
            # base_c=32, depths=[3, 3, 9, 3] 对应权重中的通道数：64->128->256->512
            {"channel": 3, "n_classes": 1, "base_c": 32, "depths": [3, 3, 9, 3], "kernel_size": 3},
            
            # 配置2: 如果n_classes不对，尝试2类
            {"channel": 3, "n_classes": 2, "base_c": 32, "depths": [3, 3, 9, 3], "kernel_size": 3},
            
            # 配置3: 尝试更大的base_c
            {"channel": 3, "n_classes": 1, "base_c": 64, "depths": [3, 3, 9, 3], "kernel_size": 3},
            
            # 配置4: 最大base_c + 2类
            {"channel": 3, "n_classes": 2, "base_c": 64, "depths": [3, 3, 9, 3], "kernel_size": 3},
        ]
        
        # 加载权重以检查模型结构
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 处理DataParallel权重 (module.前缀)
        if any(key.startswith('module.') for key in state_dict.keys()):
            print("🔧 Detected DataParallel weights, removing 'module.' prefix...")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        # 🔥 处理FSGNet.前缀
        if any(key.startswith('FSGNet.') for key in state_dict.keys()):
            print("🔧 Detected FSGNet wrapper weights, removing 'FSGNet.' prefix...")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('FSGNet.'):
                    new_key = key[7:]  # 移除"FSGNet."前缀 (7个字符)
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        # 🔍 分析权重文件的实际配置信息
        print("🔍 Analyzing weight file structure...")
        sample_keys = list(state_dict.keys())[:10]
        for key in sample_keys:
            if 'input_layer.0.conv.0.weight' in key:
                shape = state_dict[key].shape
                print(f"  input channels: {shape[0]} (base_c * 2)")
                base_c_detected = shape[0] // 2
                print(f"  detected base_c: {base_c_detected}")
                break
        
        # 检查输出层来确定n_classes
        for key in state_dict.keys():
            if 'output_layer1.0.weight' in key:
                n_classes_detected = state_dict[key].shape[0]
                print(f"  detected n_classes: {n_classes_detected}")
                break
        
        # 尝试不同配置
        for i, config in enumerate(fsg_configs):
            try:
                print(f"🔄 Trying FSGNet config {i+1}: {config}")
                self.fsg_model = FSGNet(**config)
                self.fsg_model = self.fsg_model.to(self.device)
                
                # 🎯 使用strict=False来允许部分加载
                missing_keys, unexpected_keys = self.fsg_model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"  ⚠️ Missing keys: {len(missing_keys)} (showing first 5)")
                    for key in missing_keys[:5]:
                        print(f"    - {key}")
                
                if unexpected_keys:
                    print(f"  ⚠️ Unexpected keys: {len(unexpected_keys)} (showing first 5)")
                    for key in unexpected_keys[:5]:
                        print(f"    - {key}")
                
                # 如果模型加载成功（即使有警告），继续使用
                self.fsg_model.eval()
                print("✅ FSG model loaded successfully (with possible warnings)")
                return
                
            except Exception as e:
                print(f"❌ Config {i+1} failed: {str(e)[:200]}...")
                continue
        
        # 🔧 如果所有配置都失败，尝试从权重文件动态推断配置
        print("🔧 All predefined configs failed. Attempting dynamic configuration...")
        try:
            # 动态推断配置
            dynamic_config = self._infer_config_from_weights(state_dict)
            print(f"🔄 Trying dynamic config: {dynamic_config}")
            
            self.fsg_model = FSGNet(**dynamic_config)
            self.fsg_model = self.fsg_model.to(self.device)
            missing_keys, unexpected_keys = self.fsg_model.load_state_dict(state_dict, strict=False)
            self.fsg_model.eval()
            print("✅ FSG model loaded with dynamic configuration")
            return
            
        except Exception as e:
            print(f"❌ Dynamic config also failed: {e}")
                
        raise RuntimeError("❌ All FSGNet configurations failed. Please check the model definition and weights.")

    def _infer_config_from_weights(self, state_dict):
        """从权重文件动态推断模型配置"""
        try:
            # 从input_layer推断base_c
            input_weight_shape = state_dict['input_layer.0.conv.0.weight'].shape
            base_c = input_weight_shape[0] // 2  # 通常input层输出是base_c*2
            
            # 从output_layer推断n_classes
            output_weight_shape = state_dict['output_layer1.0.weight'].shape
            n_classes = output_weight_shape[0]
            
            # 固定其他参数
            config = {
                "channel": 3,
                "n_classes": n_classes,
                "base_c": base_c,
                "depths": [3, 3, 9, 3],  # 这个通常是固定的
                "kernel_size": 3
            }
            
            return config
            
        except Exception as e:
            print(f"Failed to infer config: {e}")
            # 返回默认配置
            return {"channel": 3, "n_classes": 1, "base_c": 32, "depths": [3, 3, 9, 3], "kernel_size": 3}
        
    def preprocess_image(self, image_path):
        """预处理图像"""
        # 读取图像
        image = Image.open(image_path).convert('RGB')
        
        # Resize到608x608
        image = tf.resize(image, [608, 608])
        
        # 转为tensor并归一化
        image_tensor = tf.to_tensor(image)
        image_tensor = tf.normalize(image_tensor,
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        
        return image_tensor.unsqueeze(0).to(self.device), np.array(image)
    
    def predict_with_gemfsg(self, image_tensor):
        """使用GEMFSG预测背景，然后反转得到血管"""
        with torch.no_grad():
            # GEMFSG输出背景预测
            bg_pred = self.gemfsg_model(image_tensor)
            
            # 如果有多个输出头，取第一个或最后一个
            if isinstance(bg_pred, (list, tuple)):
                bg_pred = bg_pred[-1]  # 通常最后一个是主输出
                
            # 应用sigmoid
            bg_pred = torch.sigmoid(bg_pred)
            
            # 反转得到血管预测: vessel = 1 - background
            vessel_pred = 1.0 - bg_pred
            
        return bg_pred.squeeze().cpu().numpy(), vessel_pred.squeeze().cpu().numpy()
    
    def predict_with_fsg(self, image_tensor):
        """使用FSG直接预测血管"""
        with torch.no_grad():
            # FSG输出血管预测
            vessel_pred = self.fsg_model(image_tensor)
            
            # 如果有多个输出头，取第一个或最后一个
            if isinstance(vessel_pred, (list, tuple)):
                vessel_pred = vessel_pred[-1]  # 通常最后一个是主输出
                
            # 应用sigmoid
            vessel_pred = torch.sigmoid(vessel_pred)
            
        return vessel_pred.squeeze().cpu().numpy()
    
    def calculate_metrics(self, pred, gt):
        """计算评估指标"""
        # 二值化
        pred_binary = (pred > 0.5).astype(np.uint8)
        gt_binary = (gt > 0.5).astype(np.uint8)
        
        # 计算混淆矩阵
        tp = np.sum((pred_binary == 1) & (gt_binary == 1))
        fp = np.sum((pred_binary == 1) & (gt_binary == 0))
        fn = np.sum((pred_binary == 0) & (gt_binary == 1))
        tn = np.sum((pred_binary == 0) & (gt_binary == 0))
        
        # 计算指标
        epsilon = 1e-8
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
    
    def create_comparison_overlay(self, original_image, gemfsg_vessel, fsg_vessel, gt=None):
        """创建对比叠加图像"""
        h, w = gemfsg_vessel.shape
        
        # 确保原图尺寸匹配
        if original_image.shape[:2] != (h, w):
            original_image = cv2.resize(original_image, (w, h))
        
        # 二值化预测
        gemfsg_binary = gemfsg_vessel > 0.5
        fsg_binary = fsg_vessel > 0.5
        
        # 创建RGB叠加图
        overlay = original_image.astype(np.float32) / 255.0
        
        # 计算差异区域
        both_predict = gemfsg_binary & fsg_binary          # 两种方法都预测为血管 - 白色
        only_gemfsg = gemfsg_binary & (~fsg_binary)        # 仅GEMFSG预测为血管 - 绿色  
        only_fsg = fsg_binary & (~gemfsg_binary)           # 仅FSG预测为血管 - 红色
        
        # 应用颜色编码
        alpha = 0.6
        
        # 白色: 两种方法一致预测为血管
        overlay[both_predict] = (1-alpha) * overlay[both_predict] + alpha * np.array([1, 1, 1])
        
        # 绿色: 仅GEMFSG(背景反转)预测为血管
        overlay[only_gemfsg] = (1-alpha) * overlay[only_gemfsg] + alpha * np.array([0, 1, 0])
        
        # 红色: 仅FSG预测为血管
        overlay[only_fsg] = (1-alpha) * overlay[only_fsg] + alpha * np.array([1, 0, 0])
        
        return np.clip(overlay, 0, 1)
    
    def create_detailed_comparison(self, original_image, bg_pred, gemfsg_vessel, fsg_vessel, gt=None):
        """创建详细的对比图像网格"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 第一行
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(bg_pred, cmap='gray')
        axes[0, 1].set_title('GEMFSG Background Prediction')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(gemfsg_vessel, cmap='gray')
        axes[0, 2].set_title('GEMFSG Vessel (1-Background)')
        axes[0, 2].axis('off')
        
        # 第二行
        axes[1, 0].imshow(fsg_vessel, cmap='gray')
        axes[1, 0].set_title('FSG Vessel Prediction')
        axes[1, 0].axis('off')
        
        # 对比叠加图
        overlay = self.create_comparison_overlay(original_image, gemfsg_vessel, fsg_vessel, gt)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Comparison Overlay\n(White: Both, Green: GEMFSG only, Red: FSG only)')
        axes[1, 1].axis('off')
        
        # 差异图
        diff = np.abs(gemfsg_vessel - fsg_vessel)
        axes[1, 2].imshow(diff, cmap='hot')
        axes[1, 2].set_title('Absolute Difference')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def analyze_differences(self, gemfsg_vessel, fsg_vessel):
        """分析两种方法的差异"""
        # 二值化
        gemfsg_binary = gemfsg_vessel > 0.5
        fsg_binary = fsg_vessel > 0.5
        
        # 计算各种像素区域
        both_predict = gemfsg_binary & fsg_binary
        only_gemfsg = gemfsg_binary & (~fsg_binary)
        only_fsg = fsg_binary & (~gemfsg_binary)
        neither = (~gemfsg_binary) & (~fsg_binary)
        
        total_pixels = gemfsg_vessel.size
        
        stats = {
            'both_predict': np.sum(both_predict),
            'only_gemfsg': np.sum(only_gemfsg),
            'only_fsg': np.sum(only_fsg),
            'neither': np.sum(neither),
            'total_pixels': total_pixels,
            'agreement_rate': np.sum(both_predict | neither) / total_pixels,
            'gemfsg_total': np.sum(gemfsg_binary),
            'fsg_total': np.sum(fsg_binary)
        }
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='GEMFSG vs FSG Comparison')
    parser.add_argument('--gemfsg_weights', type=str, 
                       default='model_ckpts/2025-06-10 174126/GEM_FSGNet-Epoch_2861-f1_score_0.985349114271054.pt',
                       help='Path to GEMFSG model weights')
    parser.add_argument('--fsg_weights', type=str, 
                       default='model_ckpts/FSG-Net-DRIVE.pt',
                       help='Path to FSG model weights')
    parser.add_argument('--test_dir', type=str, 
                       default='data/DRIVE/val/input',
                       help='Directory containing test images')
    parser.add_argument('--gt_dir', type=str, 
                       default='data/DRIVE/val/label',
                       help='Directory containing ground truth masks')
    parser.add_argument('--output_dir', type=str, default='./gemfsg_vs_fsg_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # 处理URL编码的路径
    if '%20' in args.gemfsg_weights:
        args.gemfsg_weights = args.gemfsg_weights.replace('%20', ' ')
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Paths:")
    print(f"  GEMFSG weights: {args.gemfsg_weights}")
    print(f"  FSG weights: {args.fsg_weights}")
    print(f"  Test images: {args.test_dir}")
    print(f"  Ground truth: {args.gt_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Current working directory: {os.getcwd()}")
    
    # 验证路径存在
    if not os.path.exists(args.gemfsg_weights):
        print(f"❌ GEMFSG weights not found: {args.gemfsg_weights}")
        
        # 尝试列出model_ckpts目录的内容
        ckpts_dir = "model_ckpts"
        if os.path.exists(ckpts_dir):
            print(f"📂 Contents of {ckpts_dir}:")
            for item in os.listdir(ckpts_dir):
                print(f"  - {item}")
                if os.path.isdir(os.path.join(ckpts_dir, item)):
                    subdir_path = os.path.join(ckpts_dir, item)
                    print(f"    📁 Contents of {subdir_path}:")
                    for subitem in os.listdir(subdir_path):
                        print(f"      - {subitem}")
        else:
            print(f"📂 model_ckpts directory not found")
        return
        
    if not os.path.exists(args.fsg_weights):
        print(f"❌ FSG weights not found: {args.fsg_weights}")
        return
    if not os.path.exists(args.test_dir):
        print(f"❌ Test directory not found: {args.test_dir}")
        return
    
    # 初始化对比器
    comparator = ModelComparator(device=args.device)
    
    # 加载模型
    comparator.load_gemfsg_model(args.gemfsg_weights)
    comparator.load_fsg_model(args.fsg_weights)
    
    # 获取测试图像列表
    test_images = list(Path(args.test_dir).glob('*.tif')) + list(Path(args.test_dir).glob('*.png'))
    test_images = sorted(test_images)
    
    print(f"Found {len(test_images)} test images")
    
    # 存储统计信息
    all_stats = []
    gemfsg_metrics = []
    fsg_metrics = []
    
    # 处理每张图像
    for img_path in tqdm(test_images):
        print(f"\n🔄 Processing: {img_path.name}")
        
        # 预处理图像
        image_tensor, original_image = comparator.preprocess_image(img_path)
        
        # GEMFSG预测
        bg_pred, gemfsg_vessel = comparator.predict_with_gemfsg(image_tensor)
        
        # FSG预测
        fsg_vessel = comparator.predict_with_fsg(image_tensor)
        
        # 分析差异
        stats = comparator.analyze_differences(gemfsg_vessel, fsg_vessel)
        all_stats.append(stats)
        
        # 加载ground truth (如果有)
        gt = None
        if args.gt_dir:
            gt_path = Path(args.gt_dir) / f"{img_path.stem}_manual1.png"
            if not gt_path.exists():
                gt_path = Path(args.gt_dir) / f"{img_path.stem}.png"
            
            if gt_path.exists():
                gt_img = Image.open(gt_path).convert('L')
                gt_img = tf.resize(gt_img, [608, 608], interpolation=tf.InterpolationMode.NEAREST)
                gt = np.array(gt_img) / 255.0
                
                # 计算指标
                gemfsg_metric = comparator.calculate_metrics(gemfsg_vessel, gt)
                fsg_metric = comparator.calculate_metrics(fsg_vessel, gt)
                gemfsg_metrics.append(gemfsg_metric)
                fsg_metrics.append(fsg_metric)
                
                print(f"  GEMFSG - F1: {gemfsg_metric['f1']:.4f}, Precision: {gemfsg_metric['precision']:.4f}, Recall: {gemfsg_metric['recall']:.4f}")
                print(f"  FSG    - F1: {fsg_metric['f1']:.4f}, Precision: {fsg_metric['precision']:.4f}, Recall: {fsg_metric['recall']:.4f}")
        
        # 创建对比图像
        fig = comparator.create_detailed_comparison(original_image, bg_pred, gemfsg_vessel, fsg_vessel, gt)
        fig.savefig(output_dir / f"{img_path.stem}_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 保存叠加图
        overlay = comparator.create_comparison_overlay(original_image, gemfsg_vessel, fsg_vessel, gt)
        overlay_img = (overlay * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{img_path.stem}_overlay.png"), 
                   cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        
        # 打印差异统计
        print(f"  Pixel Analysis:")
        print(f"    Both methods agree: {stats['both_predict']} ({stats['both_predict']/stats['total_pixels']*100:.1f}%)")
        print(f"    Only GEMFSG: {stats['only_gemfsg']} ({stats['only_gemfsg']/stats['total_pixels']*100:.1f}%)")
        print(f"    Only FSG: {stats['only_fsg']} ({stats['only_fsg']/stats['total_pixels']*100:.1f}%)")
        print(f"    Agreement rate: {stats['agreement_rate']*100:.1f}%")
    
    # 汇总统计
    print(f"\n📊 Overall Statistics:")
    print(f"Total images processed: {len(test_images)}")
    
    if gemfsg_metrics and fsg_metrics:
        avg_gemfsg = {k: np.mean([m[k] for m in gemfsg_metrics]) for k in gemfsg_metrics[0].keys() if k not in ['tp', 'fp', 'fn', 'tn']}
        avg_fsg = {k: np.mean([m[k] for m in fsg_metrics]) for k in fsg_metrics[0].keys() if k not in ['tp', 'fp', 'fn', 'tn']}
        
        print(f"\nAverage Metrics:")
        print(f"GEMFSG - F1: {avg_gemfsg['f1']:.4f}, Precision: {avg_gemfsg['precision']:.4f}, Recall: {avg_gemfsg['recall']:.4f}")
        print(f"FSG    - F1: {avg_fsg['f1']:.4f}, Precision: {avg_fsg['precision']:.4f}, Recall: {avg_fsg['recall']:.4f}")
    
    # 保存汇总统计
    avg_agreement = np.mean([s['agreement_rate'] for s in all_stats])
    print(f"\nAverage agreement rate: {avg_agreement*100:.1f}%")
    
    print(f"\n✅ Results saved to: {output_dir}")
    print(f"📈 Color coding:")
    print(f"  - White: Both methods predict vessel")
    print(f"  - Green: Only GEMFSG (background-inverted) predicts vessel")
    print(f"  - Red: Only FSG predicts vessel")


if __name__ == "__main__":
    main()