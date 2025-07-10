import torch
import torch.nn as nn
import cv2
import numpy as np
from models.backbones.FSGNet import FSGNet
from models import dataloader as dataloader_hub
import argparse
from PIL import Image
import os


class SimpleRegionGrowing:
    """简单区域生长类"""
    def __init__(self, mu_f=0.0789, sigma_f=0.0774, alpha=1.0):
        self.mu_f = mu_f
        self.sigma_f = sigma_f  
        self.alpha = alpha
        self.vessel_threshold = mu_f + alpha * sigma_f
    
    def preprocess_image(self, rgb_image):
        """预处理单张图像"""
        # rgb_image: [3, H, W] tensor
        green_channel = rgb_image[1].cpu().numpy()  # [H, W]
        
        # 归一化
        min_val, max_val = green_channel.min(), green_channel.max()
        if max_val > min_val:
            normalized = (green_channel - min_val) / (max_val - min_val)
        else:
            normalized = green_channel
        
        # 背景减除增强
        background = cv2.GaussianBlur(normalized, (101, 101), 0)
        enhanced = np.clip(normalized - background, 0, 1)
        
        return enhanced
    
    def calculate_gradient(self, enhanced_image):
        """计算梯度"""
        # Sobel算子
        grad_x = cv2.Sobel(enhanced_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(enhanced_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return gradient_magnitude
    
    def region_growing(self, fsg_prediction, gradient_magnitude, fsg_threshold):
        """区域生长"""
        # fsg_prediction: [H, W] numpy array (0-1概率)
        
        # 1. 将FSG预测概率转换为二值种子点
        seed_binary = (fsg_prediction > fsg_threshold).astype(np.uint8)
        print(f"Initial seed points: {seed_binary.sum()} pixels")
        
        if seed_binary.sum() == 0:
            print("No seed points found!")
            return np.zeros_like(fsg_prediction, dtype=np.float32)
        
        # 2. 基于梯度阈值选择候选血管像素
        vessel_candidates = (gradient_magnitude >= self.vessel_threshold).astype(np.uint8)
        print(f"Vessel candidates (gradient >= {self.vessel_threshold:.4f}): {vessel_candidates.sum()} pixels")
        
        # 3. 区域生长：从种子点扩展到候选像素
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grown_mask = seed_binary.copy()
        
        # 迭代扩展：只在候选像素区域内扩展
        for i in range(5):
            # 膨胀当前掩码
            dilated = cv2.dilate(grown_mask, kernel, iterations=1)
            # 只保留在血管候选区域内的扩展
            new_grown = dilated * vessel_candidates
            
            # 检查是否还有新增长
            if np.array_equal(new_grown, grown_mask):
                print(f"Converged after {i+1} iterations")
                break
            grown_mask = new_grown
        
        print(f"Final grown region: {grown_mask.sum()} pixels")
        return grown_mask.astype(np.float32)
    
    def create_circular_mask(self, image_shape):
        """创建圆形掩码，去除边框区域"""
        h, w = image_shape
        center_y, center_x = h // 2, w // 2
        radius = min(center_y, center_x) * 0.9  # 稍微小一点避免边缘
        
        Y, X = np.ogrid[:h, :w]
        mask = ((X - center_x) ** 2 + (Y - center_y) ** 2) <= radius ** 2
        return mask.astype(np.float32)
    
    def get_dynamic_threshold(self, prediction, circular_mask):
        """计算动态阈值"""
        valid_pixels = prediction[circular_mask > 0]
        if len(valid_pixels) > 0:
            pred_mean = valid_pixels.mean()
            pred_std = valid_pixels.std()
            # 使用均值 + 0.5 * 标准差作为阈值
            threshold = pred_mean + 0.5 * pred_std
            print(f"Dynamic threshold: {threshold:.3f} (mean={pred_mean:.3f}, std={pred_std:.3f})")
        else:
            threshold = 0.5
            print(f"Using default threshold: {threshold:.3f}")
        return threshold
    
    def fuse_multi_outputs(self, out1, out2, out3, circular_mask):
        """使用阈值判断后的并集融合多个输出"""
        print("=== Multi-Output Fusion ===")
        
        # 对每个输出计算动态阈值
        out1_masked = out1 * circular_mask
        out2_masked = out2 * circular_mask
        out3_masked = out3 * circular_mask
        
        threshold1 = self.get_dynamic_threshold(out1_masked, circular_mask)
        threshold2 = self.get_dynamic_threshold(out2_masked, circular_mask)
        threshold3 = self.get_dynamic_threshold(out3_masked, circular_mask)
        
        # 二值化每个输出
        binary1 = (out1_masked > threshold1).astype(np.float32)
        binary2 = (out2_masked > threshold2).astype(np.float32)
        binary3 = (out3_masked > threshold3).astype(np.float32)
        
        print(f"Binary pixels - out1: {binary1.sum():.0f}, out2: {binary2.sum():.0f}, out3: {binary3.sum():.0f}")
        
        # 计算并集
        fused_binary = np.logical_or(np.logical_or(binary1, binary2), binary3).astype(np.float32)
        
        # 应用圆形掩码
        fused_result = fused_binary * circular_mask
        
        print(f"Fused binary pixels: {fused_result.sum():.0f}")
        print(f"Coverage increase vs out1: {fused_result.sum() - binary1.sum():.0f} pixels")
        
        return fused_result, (threshold1, threshold2, threshold3), (binary1, binary2, binary3)
    
    def refine_prediction(self, fsg_output, original_image):
        """细化FSG预测结果"""
        # fsg_output: [H, W] numpy array (FSG预测概率 0-1)
        # original_image: [3, H, W] tensor
        
        print(f"FSG prediction range: [{fsg_output.min():.3f}, {fsg_output.max():.3f}]")
        
        # 创建圆形掩码去除边框
        circular_mask = self.create_circular_mask(fsg_output.shape)
        fsg_output_masked = fsg_output * circular_mask
        print(f"After circular masking: [{fsg_output_masked.min():.3f}, {fsg_output_masked.max():.3f}]")
        
        # 1. 预处理图像
        enhanced = self.preprocess_image(original_image)
        enhanced_masked = enhanced * circular_mask  # 同样应用掩码
        
        # 2. 计算梯度
        gradient = self.calculate_gradient(enhanced_masked)
        print(f"Gradient range: [{gradient.min():.3f}, {gradient.max():.3f}]")
        
        # 3. 计算动态阈值
        fsg_threshold = self.get_dynamic_threshold(fsg_output_masked, circular_mask)
        
        # 4. 区域生长：使用动态阈值的FSG预测作为种子点
        grown_result = self.region_growing(fsg_output_masked, gradient, fsg_threshold)
        
        # 5. 融合策略：取并集（OR操作）
        fsg_binary = (fsg_output_masked > fsg_threshold).astype(np.float32)
        final_result = np.logical_or(fsg_binary, grown_result).astype(np.float32)
        
        # 应用圆形掩码到最终结果
        final_result = final_result * circular_mask
        
        print(f"FSG binary pixels: {fsg_binary.sum():.0f}")
        print(f"Region grown pixels: {grown_result.sum():.0f}")
        print(f"Final result pixels: {final_result.sum():.0f}")
        print(f"Net gain: {final_result.sum() - fsg_binary.sum():.0f} pixels")
        
        return final_result


def create_overlay_image(original_image, prediction, ground_truth=None, colors=None):
    """创建叠加可视化图像"""
    if colors is None:
        colors = {
            'prediction': [0, 1, 0],      # 绿色 - 预测
            'ground_truth': [1, 0, 0],    # 红色 - 真实标签
            'both': [1, 1, 0],            # 黄色 - 重叠区域
            'alpha': 0.6                  # 透明度
        }
    
    # 确保输入是numpy数组
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if ground_truth is not None and isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    
    # 转换为RGB格式 [H, W, 3]
    if original_image.ndim == 3 and original_image.shape[0] == 3:
        original_image = original_image.transpose(1, 2, 0)
    
    # 创建叠加图像
    overlay = original_image.copy()
    
    # 二值化预测结果
    pred_binary = prediction > 0.5
    
    if ground_truth is not None:
        # 二值化真实标签
        gt_binary = ground_truth > 0.5
        
        # 创建不同区域的掩码
        both_mask = pred_binary & gt_binary      # 两者都有 - 黄色
        pred_only = pred_binary & (~gt_binary)   # 只有预测 - 绿色
        gt_only = gt_binary & (~pred_binary)     # 只有真实标签 - 红色
        
        # 应用颜色
        alpha = colors['alpha']
        
        # 黄色区域 (重叠)
        overlay[both_mask] = (1 - alpha) * overlay[both_mask] + alpha * np.array(colors['both'])
        
        # 绿色区域 (仅预测)
        overlay[pred_only] = (1 - alpha) * overlay[pred_only] + alpha * np.array(colors['prediction'])
        
        # 红色区域 (仅真实标签)
        overlay[gt_only] = (1 - alpha) * overlay[gt_only] + alpha * np.array(colors['ground_truth'])
        
    else:
        # 只有预测结果
        alpha = colors['alpha']
        overlay[pred_binary] = (1 - alpha) * overlay[pred_binary] + alpha * np.array(colors['prediction'])
    
    return np.clip(overlay, 0, 1)


def create_refinement_improvement_overlay(original_image, fsg_prediction, refined_prediction, ground_truth):
    """创建显示细化改进效果的叠加图"""
    # 确保输入是numpy数组
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy()
    if isinstance(fsg_prediction, torch.Tensor):
        fsg_prediction = fsg_prediction.cpu().numpy()
    if isinstance(refined_prediction, torch.Tensor):
        refined_prediction = refined_prediction.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    
    # 转换为RGB格式
    if original_image.ndim == 3 and original_image.shape[0] == 3:
        original_image = original_image.transpose(1, 2, 0)
    
    # 二值化
    fsg_binary = fsg_prediction > 0.5
    refined_binary = refined_prediction > 0.5
    gt_binary = ground_truth > 0.5
    
    # 计算变化区域
    added_pixels = refined_binary & (~fsg_binary)     # 细化新增的像素
    removed_pixels = fsg_binary & (~refined_binary)   # 细化移除的像素
    
    # 分析新增像素的正确性
    added_correct = added_pixels & gt_binary          # 新增的正确像素（真正的改进）
    added_incorrect = added_pixels & (~gt_binary)     # 新增的错误像素（假阳性增加）
    
    # 分析移除像素的影响
    removed_correct = removed_pixels & gt_binary      # 移除的正确像素（真阳性丢失）
    removed_incorrect = removed_pixels & (~gt_binary) # 移除的错误像素（假阳性减少，好事）
    
    # 创建叠加图像
    overlay = original_image.copy()
    alpha = 0.7
    
    # 应用颜色编码
    overlay[added_correct] = (1 - alpha) * overlay[added_correct] + alpha * np.array([0, 1, 0])      # 绿色 - 正确改进
    overlay[added_incorrect] = (1 - alpha) * overlay[added_incorrect] + alpha * np.array([1, 0, 0])  # 红色 - 错误增加
    overlay[removed_correct] = (1 - alpha) * overlay[removed_correct] + alpha * np.array([0, 0, 1])  # 蓝色 - 正确丢失
    overlay[removed_incorrect] = (1 - alpha) * overlay[removed_incorrect] + alpha * np.array([1, 1, 0]) # 黄色 - 错误减少
    
    # 统计信息
    improvement_stats = {
        'added_correct': int(added_correct.sum()),
        'added_incorrect': int(added_incorrect.sum()),
        'removed_correct': int(removed_correct.sum()),
        'removed_incorrect': int(removed_incorrect.sum()),
        'net_improvement': int(added_correct.sum() - removed_correct.sum()),
        'net_false_positive_change': int(added_incorrect.sum() - removed_incorrect.sum())
    }
    
    return np.clip(overlay, 0, 1), improvement_stats


def create_fusion_comparison_overlay(original_image, fused_prediction, individual_outputs, ground_truth=None):
    """创建融合结果与单个输出的对比图"""
    # individual_outputs: (out1, out2, out3) tuple
    out1, out2, out3 = individual_outputs
    
    # 确保输入是numpy数组
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy()
    if isinstance(fused_prediction, torch.Tensor):
        fused_prediction = fused_prediction.cpu().numpy()
    
    # 转换为RGB格式
    if original_image.ndim == 3 and original_image.shape[0] == 3:
        original_image = original_image.transpose(1, 2, 0)
    
    # 创建叠加图像
    overlay = original_image.copy()
    
    # 二值化
    fused_binary = fused_prediction > 0.5
    out1_binary = out1 > 0.5
    out2_binary = out2 > 0.5
    out3_binary = out3 > 0.5
    
    # 创建不同区域的掩码
    fusion_only = fused_binary & (~(out1_binary | out2_binary | out3_binary))  # 仅融合有的 - 红色
    common_area = fused_binary & out1_binary  # 融合与out1共同区域 - 绿色
    out2_contrib = fused_binary & out2_binary & (~out1_binary)  # out2贡献 - 蓝色
    out3_contrib = fused_binary & out3_binary & (~out1_binary) & (~out2_binary)  # out3贡献 - 紫色
    
    # 应用颜色
    alpha = 0.6
    overlay[common_area] = (1 - alpha) * overlay[common_area] + alpha * np.array([0, 1, 0])      # 绿色
    overlay[out2_contrib] = (1 - alpha) * overlay[out2_contrib] + alpha * np.array([0, 0, 1])    # 蓝色
    overlay[out3_contrib] = (1 - alpha) * overlay[out3_contrib] + alpha * np.array([1, 0, 1])    # 紫色
    overlay[fusion_only] = (1 - alpha) * overlay[fusion_only] + alpha * np.array([1, 0, 0])      # 红色
    
    return np.clip(overlay, 0, 1)


def load_fsg_model(model_path, device):
    """加载FSG模型"""
    model = FSGNet(channel=3, n_classes=1, base_c=64, depths=[3, 3, 9, 3], kernel_size=3)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 处理DataParallel保存的权重 (去掉module.FSGNet.前缀)
    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('module.FSGNet.'):
            # 去掉 'module.FSGNet.' 前缀
            new_key = key[len('module.FSGNet.'):]
            state_dict[new_key] = value
        elif key.startswith('module.'):
            # 去掉 'module.' 前缀
            new_key = key[len('module.'):]
            state_dict[new_key] = value
        else:
            state_dict[key] = value
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Successfully loaded FSG model with {len(state_dict)} parameters")
    return model


def find_ground_truth(img_file, possible_gt_dirs):
    """查找对应的真实标签文件"""
    base_name = os.path.splitext(img_file)[0]
    
    # 常见的标签文件命名模式
    possible_names = [
        f"{base_name}_manual1.gif",
        f"{base_name}_manual1.png",
        f"{base_name}_manual1.tif",
        f"{base_name}_gt.png",
        f"{base_name}_gt.gif",
        f"{base_name}_gt.tif",
        f"{base_name}.png",
        f"{base_name}.gif",
        f"{base_name}.tif"
    ]
    
    for gt_dir in possible_gt_dirs:
        if os.path.exists(gt_dir):
            for possible_name in possible_names:
                gt_path = os.path.join(gt_dir, possible_name)
                if os.path.exists(gt_path):
                    return gt_path
    return None


def inference_with_fusion_region_growing():
    # 配置 - 先尝试val目录，如果为空则使用train目录
    fsg_model_path = '/root/FSG-Net-pytorch/model_ckpts/FSG-Net-DRIVE.pt'
    
    # 尝试多个可能的数据路径
    possible_paths = [
        '/root/FSG-Net-pytorch/data/DRIVE/val/input',
        '/root/FSG-Net-pytorch/data/DRIVE/train/input',
        '/root/FSG-Net-pytorch/data/DRIVE/val',
        '/root/FSG-Net-pytorch/data/DRIVE/train'
    ]
    
    # 可能的真实标签路径
    possible_gt_dirs = [
        '/root/FSG-Net-pytorch/data/DRIVE/val/gt',
        '/root/FSG-Net-pytorch/data/DRIVE/train/gt',
        '/root/FSG-Net-pytorch/data/DRIVE/val/label',
        '/root/FSG-Net-pytorch/data/DRIVE/train/label',
        '/root/FSG-Net-pytorch/data/DRIVE/val/mask',
        '/root/FSG-Net-pytorch/data/DRIVE/train/mask'
    ]
    
    test_data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
            if len(files) > 0:
                test_data_path = path
                print(f"Found {len(files)} images in: {path}")
                break
            else:
                print(f"Path exists but no images found: {path}")
        else:
            print(f"Path does not exist: {path}")
    
    if test_data_path is None:
        print("No valid image directory found!")
        print("Please check your data structure or put some test images in one of these directories:")
        for path in possible_paths:
            print(f"  {path}")
        return
    
    # 更换输出目录
    output_dir = './fusion_region_growing_results'
    
    # 区域生长参数
    mu_f = 0.0789
    sigma_f = 0.0774
    alpha = 1.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载FSG模型
    print("Loading FSG-Net model...")
    fsg_model = load_fsg_model(fsg_model_path, device)
    
    # 初始化区域生长
    region_grower = SimpleRegionGrowing(mu_f, sigma_f, alpha)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/fsg_individual", exist_ok=True)  # 单个输出
    os.makedirs(f"{output_dir}/fsg_fusion", exist_ok=True)      # 融合输出
    os.makedirs(f"{output_dir}/refined_individual", exist_ok=True)  # 单个输出细化
    os.makedirs(f"{output_dir}/refined_fusion", exist_ok=True)      # 融合输出细化
    os.makedirs(f"{output_dir}/overlay_fusion_vs_gt", exist_ok=True)        # 融合结果与GT对比
    os.makedirs(f"{output_dir}/overlay_fusion_comparison", exist_ok=True)   # 融合与单个输出对比
    os.makedirs(f"{output_dir}/improvement_individual", exist_ok=True)      # 单个输出改进
    os.makedirs(f"{output_dir}/improvement_fusion", exist_ok=True)          # 融合输出改进
    
    # 查找图像文件
    all_files = os.listdir(test_data_path)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
    
    print(f"Found {len(all_files)} total files in {test_data_path}")
    print(f"Found {len(image_files)} image files")
    
    if len(image_files) == 0:
        print("No image files found! Listing all files in directory:")
        for f in all_files[:10]:  # 显示前10个文件
            print(f"  {f}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    # 处理每张图像
    for i, img_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {img_file}")
        print("=" * 80)
        
        try:
            # 加载图像
            img_path = os.path.join(test_data_path, img_file)
            image = Image.open(img_path).convert('RGB')
            print(f"  Image size: {image.size}")
            
            # 转换为tensor并归一化
            img_array = np.array(image) / 255.0
            img_tensor = torch.tensor(img_array.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
            print(f"  Tensor shape: {img_tensor.shape}")
            
            # 查找对应的真实标签
            gt_path = find_ground_truth(img_file, possible_gt_dirs)
            ground_truth = None
            if gt_path:
                try:
                    gt_image = Image.open(gt_path).convert('L')
                    ground_truth = np.array(gt_image) / 255.0
                    print(f"  Found ground truth: {os.path.basename(gt_path)}")
                except Exception as e:
                    print(f"  Error loading ground truth: {e}")
                    ground_truth = None
            else:
                print(f"  No ground truth found for {img_file}")
            
            # FSG推理 - 获取三个输出
            with torch.no_grad():
                fsg_out1, fsg_out2, fsg_out3 = fsg_model(img_tensor)
                
                # 转换为numpy
                out1 = fsg_out1.squeeze().cpu().numpy()  # [H, W]
                out2 = fsg_out2.squeeze().cpu().numpy()  # [H, W]
                out3 = fsg_out3.squeeze().cpu().numpy()  # [H, W]
                
                print(f"  Output shapes - out1: {out1.shape}, out2: {out2.shape}, out3: {out3.shape}")
                print(f"  Output ranges - out1: [{out1.min():.3f}, {out1.max():.3f}]")
                print(f"  Output ranges - out2: [{out2.min():.3f}, {out2.max():.3f}]")
                print(f"  Output ranges - out3: [{out3.min():.3f}, {out3.max():.3f}]")
            
            # 创建圆形掩码
            circular_mask = region_grower.create_circular_mask(out1.shape)
            
            # 多输出融合（阈值判断后的并集）
            fused_result, thresholds, individual_binaries = region_grower.fuse_multi_outputs(
                out1, out2, out3, circular_mask
            )
            
            # 区域生长细化
            original_tensor = img_tensor.squeeze()  # [3, H, W]
            
            # 对单个输出进行细化
            print("\n=== Individual Output Refinement ===")
            refined_out1 = region_grower.refine_prediction(out1, original_tensor)
            
            # 对融合输出进行细化
            print("\n=== Fusion Output Refinement ===")
            refined_fusion = region_grower.refine_prediction(fused_result, original_tensor)
            
            # 保存结果
            base_name = os.path.splitext(img_file)[0]
            
            # 1. 保存单个FSG输出
            out1_img = (out1 * 255).astype(np.uint8)
            out2_img = (out2 * 255).astype(np.uint8)
            out3_img = (out3 * 255).astype(np.uint8)
            Image.fromarray(out1_img).save(f"{output_dir}/fsg_individual/{base_name}_out1.png")
            Image.fromarray(out2_img).save(f"{output_dir}/fsg_individual/{base_name}_out2.png")
            Image.fromarray(out3_img).save(f"{output_dir}/fsg_individual/{base_name}_out3.png")
            
            # 2. 保存融合输出
            fused_img = (fused_result * 255).astype(np.uint8)
            Image.fromarray(fused_img).save(f"{output_dir}/fsg_fusion/{base_name}_fused.png")
            
            # 3. 保存细化结果
            refined_out1_img = (refined_out1 * 255).astype(np.uint8)
            refined_fusion_img = (refined_fusion * 255).astype(np.uint8)
            Image.fromarray(refined_out1_img).save(f"{output_dir}/refined_individual/{base_name}_refined_out1.png")
            Image.fromarray(refined_fusion_img).save(f"{output_dir}/refined_fusion/{base_name}_refined_fusion.png")
            
            # 4. 创建和保存overlay图像
            if ground_truth is not None:
                # 融合结果与真实标签的overlay
                fusion_gt_overlay = create_overlay_image(img_array, refined_fusion, ground_truth)
                fusion_gt_overlay_img = (fusion_gt_overlay * 255).astype(np.uint8)
                Image.fromarray(fusion_gt_overlay_img).save(f"{output_dir}/overlay_fusion_vs_gt/{base_name}_fusion_vs_gt.png")
                
                # 单个输出改进效果图
                individual_improvement_overlay, individual_stats = create_refinement_improvement_overlay(
                    img_array, out1, refined_out1, ground_truth
                )
                individual_improvement_img = (individual_improvement_overlay * 255).astype(np.uint8)
                Image.fromarray(individual_improvement_img).save(f"{output_dir}/improvement_individual/{base_name}_individual_improvement.png")
                
                # 融合输出改进效果图
                fusion_improvement_overlay, fusion_stats = create_refinement_improvement_overlay(
                    img_array, fused_result, refined_fusion, ground_truth
                )
                fusion_improvement_img = (fusion_improvement_overlay * 255).astype(np.uint8)
                Image.fromarray(fusion_improvement_img).save(f"{output_dir}/improvement_fusion/{base_name}_fusion_improvement.png")
                
                # 打印改进统计信息
                print(f"\n=== Individual Output (out1) Improvement Stats ===")
                print(f"  ✅ 新增正确像素: {individual_stats['added_correct']}")
                print(f"  ❌ 新增错误像素: {individual_stats['added_incorrect']}")
                print(f"  ⚠️  丢失正确像素: {individual_stats['removed_correct']}")
                print(f"  ✅ 减少错误像素: {individual_stats['removed_incorrect']}")
                print(f"  📈 净改进(正确像素): {individual_stats['net_improvement']}")
                print(f"  📊 假阳性净变化: {individual_stats['net_false_positive_change']}")
                
                print(f"\n=== Fusion Output Improvement Stats ===")
                print(f"  ✅ 新增正确像素: {fusion_stats['added_correct']}")
                print(f"  ❌ 新增错误像素: {fusion_stats['added_incorrect']}")
                print(f"  ⚠️  丢失正确像素: {fusion_stats['removed_correct']}")
                print(f"  ✅ 减少错误像素: {fusion_stats['removed_incorrect']}")
                print(f"  📈 净改进(正确像素): {fusion_stats['net_improvement']}")
                print(f"  📊 假阳性净变化: {fusion_stats['net_false_positive_change']}")
                
                # 对比个人输出vs融合输出的改进效果
                print(f"\n=== Individual vs Fusion Comparison ===")
                print(f"  Individual净改进: {individual_stats['net_improvement']}")
                print(f"  Fusion净改进: {fusion_stats['net_improvement']}")
                improvement_diff = fusion_stats['net_improvement'] - individual_stats['net_improvement']
                print(f"  Fusion相对优势: {improvement_diff:+d} pixels")
            
            # 5. 融合与单个输出的对比
            fusion_comparison_overlay = create_fusion_comparison_overlay(
                img_array, refined_fusion, (refined_out1, out2, out3), ground_truth
            )
            fusion_comparison_img = (fusion_comparison_overlay * 255).astype(np.uint8)
            Image.fromarray(fusion_comparison_img).save(f"{output_dir}/overlay_fusion_comparison/{base_name}_fusion_comparison.png")
            
            print(f"\n  Results saved for {base_name}")
            print("=" * 80)
            
        except Exception as e:
            print(f"  Error processing {img_file}: {str(e)}")
            continue
    
    print(f"\nAll results saved to {output_dir}")
    print(f"Check the following directories:")
    print(f"  - {output_dir}/fsg_individual/ (单个FSG输出: out1, out2, out3)")
    print(f"  - {output_dir}/fsg_fusion/ (阈值并集融合结果)")
    print(f"  - {output_dir}/refined_individual/ (单个输出细化结果)")
    print(f"  - {output_dir}/refined_fusion/ (融合输出细化结果)")
    print(f"  - {output_dir}/overlay_fusion_vs_gt/ (融合细化结果与真实标签对比)")
    print(f"  - {output_dir}/overlay_fusion_comparison/ (融合与单个输出对比)")
    print(f"  - {output_dir}/improvement_individual/ (单个输出改进效果图 ⭐)")
    print(f"  - {output_dir}/improvement_fusion/ (融合输出改进效果图 ⭐)")
    
    print("\n融合策略说明:")
    print("  🔍 对每个输出(out1, out2, out3)分别计算动态阈值")
    print("  🎯 将每个输出二值化后取并集(Union)")
    print("  📈 融合结果包含了所有输出的血管信息")
    print("  🔬 区域生长进一步细化融合结果")
    
    print("\nOverlay颜色说明:")
    print("  改进效果图:")
    print("    🟢 绿色 = 细化新增的正确像素 (真正的改进!)")
    print("    🔴 红色 = 细化新增的错误像素 (假阳性增加)")
    print("    🔵 蓝色 = 细化丢失的正确像素 (真阳性丢失)")
    print("    🟡 黄色 = 细化减少的错误像素 (假阳性减少,好事)")
    print("  融合对比图:")
    print("    🟢 绿色 = 融合与out1共同区域")
    print("    🔵 蓝色 = out2独有贡献")
    print("    🟣 紫色 = out3独有贡献")
    print("    🔴 红色 = 仅融合有的区域")


if __name__ == "__main__":
    inference_with_fusion_region_growing()