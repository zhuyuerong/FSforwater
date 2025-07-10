import torch
import torch.nn as nn
import cv2
import numpy as np
from models.backbones.FSGNet import FSGNet
import torchvision.transforms.functional as tf
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
        print(f"    Initial seed points: {seed_binary.sum()} pixels")
        
        if seed_binary.sum() == 0:
            print("    No seed points found!")
            return np.zeros_like(fsg_prediction, dtype=np.float32)
        
        # 2. 基于梯度阈值选择候选血管像素
        vessel_candidates = (gradient_magnitude >= self.vessel_threshold).astype(np.uint8)
        print(f"    Vessel candidates (gradient >= {self.vessel_threshold:.4f}): {vessel_candidates.sum()} pixels")
        
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
                print(f"    Converged after {i+1} iterations")
                break
            grown_mask = new_grown
        
        print(f"    Final grown region: {grown_mask.sum()} pixels")
        return grown_mask.astype(np.float32)
    
    def get_dynamic_threshold(self, prediction):
        """计算动态阈值 - 不使用圆形掩码"""
        if len(prediction[prediction > 0]) > 0:
            pred_mean = prediction.mean()
            pred_std = prediction.std()
            # 使用均值 + 0.5 * 标准差作为阈值
            threshold = pred_mean + 0.5 * pred_std
            print(f"    Dynamic threshold: {threshold:.3f} (mean={pred_mean:.3f}, std={pred_std:.3f})")
        else:
            threshold = 0.5
            print(f"    Using default threshold: {threshold:.3f}")
        return threshold
    
    def refine_prediction(self, fsg_output, original_image, output_name="output"):
        """细化FSG预测结果"""
        # fsg_output: [H, W] numpy array (FSG预测概率 0-1)
        # original_image: [3, H, W] tensor (已经过ImageNet归一化)
        
        print(f"  {output_name} prediction range: [{fsg_output.min():.3f}, {fsg_output.max():.3f}]")
        
        # 1. 反归一化图像以进行梯度计算
        if not hasattr(self, '_enhanced_cached'):
            # 反归一化ImageNet标准化 - 确保在相同设备上
            mean = torch.tensor([0.485, 0.456, 0.406], device=original_image.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=original_image.device).view(3, 1, 1)
            unnormalized_image = original_image * std + mean
            unnormalized_image = torch.clamp(unnormalized_image, 0, 1)
            
            enhanced = self.preprocess_image(unnormalized_image)
            gradient = self.calculate_gradient(enhanced)
            print(f"  Gradient range: [{gradient.min():.3f}, {gradient.max():.3f}]")
            
            # 缓存结果，避免重复计算
            self._enhanced_cached = enhanced
            self._gradient_cached = gradient
        else:
            enhanced = self._enhanced_cached
            gradient = self._gradient_cached
        
        # 2. 计算动态阈值
        fsg_threshold = self.get_dynamic_threshold(fsg_output)
        
        # 3. 区域生长：使用动态阈值的FSG预测作为种子点
        grown_result = self.region_growing(fsg_output, gradient, fsg_threshold)
        
        # 4. 融合策略：取并集（OR操作）
        fsg_binary = (fsg_output > fsg_threshold).astype(np.float32)
        final_result = np.logical_or(fsg_binary, grown_result).astype(np.float32)
        
        print(f"  {output_name} FSG binary pixels: {fsg_binary.sum():.0f}")
        print(f"  {output_name} Region grown pixels: {grown_result.sum():.0f}")
        print(f"  {output_name} Final result pixels: {final_result.sum():.0f}")
        print(f"  {output_name} Net gain: {final_result.sum() - fsg_binary.sum():.0f} pixels")
        
        return final_result, fsg_binary, grown_result, fsg_threshold
    
    def clear_cache(self):
        """清除缓存，用于处理新图像"""
        if hasattr(self, '_enhanced_cached'):
            delattr(self, '_enhanced_cached')
        if hasattr(self, '_gradient_cached'):
            delattr(self, '_gradient_cached')


def preprocess_image_like_original(image):
    """按照原始dataloader方式预处理图像"""
    # 1. Resize到608x608
    image = tf.resize(image, [608, 608])
    
    # 2. 转为tensor
    image_tensor = tf.to_tensor(image)
    
    # 3. ImageNet归一化
    image_tensor = tf.normalize(image_tensor,
                               mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
    
    return image_tensor


def preprocess_target_like_original(target):
    """按照原始dataloader方式预处理标签"""
    # 1. Resize到608x608
    target = tf.resize(target, [608, 608], interpolation=tf.InterpolationMode.NEAREST)
    
    # 2. 转为tensor
    target_tensor = torch.tensor(np.array(target))
    
    # 3. 二值化处理
    target_tensor[target_tensor < 128] = 0
    target_tensor[target_tensor >= 128] = 1
    
    # 4. 添加channel维度
    target_tensor = target_tensor.unsqueeze(0)
    
    return target_tensor


def calculate_metrics_like_original(prediction, ground_truth):
    """按照原始方式计算指标"""
    # 确保都是numpy数组
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    
    # 展平
    pred_flat = prediction.flatten()
    gt_flat = ground_truth.flatten()
    
    # 二值化预测
    pred_binary = np.zeros_like(pred_flat)
    pred_binary[pred_flat > 0.5] = 1
    
    # 计算混淆矩阵
    try:
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true=gt_flat, y_pred=pred_binary).ravel()
    except:
        # 简单计算
        tp = np.sum((pred_binary == 1) & (gt_flat == 1))
        fp = np.sum((pred_binary == 1) & (gt_flat == 0))
        fn = np.sum((pred_binary == 0) & (gt_flat == 1))
        tn = np.sum((pred_binary == 0) & (gt_flat == 0))
    
    epsilon = 2.22045e-16
    
    # 计算指标
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    sensitivity = tp / (tp + fn + epsilon)  # Recall
    precision = tp / (tp + fp + epsilon)
    f1_score = (2 * sensitivity * precision) / (sensitivity + precision + epsilon)
    
    return {
        'f1': f1_score,
        'precision': precision,
        'recall': sensitivity,
        'accuracy': accuracy,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


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


def create_improvement_overlay(original_image, fsg_prediction, refined_prediction, ground_truth):
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


def create_multi_output_comparison_overlay(original_image, out1_result, out2_result, out3_result):
    """创建多输出对比的叠加图"""
    # 确保输入是numpy数组
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy()
    
    # 转换为RGB格式
    if original_image.ndim == 3 and original_image.shape[0] == 3:
        original_image = original_image.transpose(1, 2, 0)
    
    # 创建叠加图像
    overlay = original_image.copy()
    
    # 二值化
    out1_binary = out1_result > 0.5
    out2_binary = out2_result > 0.5
    out3_binary = out3_result > 0.5
    
    # 创建不同区域的掩码
    all_three = out1_binary & out2_binary & out3_binary     # 三个都有 - 白色
    out1_and_2 = out1_binary & out2_binary & (~out3_binary) # out1和out2 - 黄色
    out1_and_3 = out1_binary & out3_binary & (~out2_binary) # out1和out3 - 紫色
    out2_and_3 = out2_binary & out3_binary & (~out1_binary) # out2和out3 - 青色
    out1_only = out1_binary & (~out2_binary) & (~out3_binary) # 仅out1 - 绿色
    out2_only = out2_binary & (~out1_binary) & (~out3_binary) # 仅out2 - 红色
    out3_only = out3_binary & (~out1_binary) & (~out2_binary) # 仅out3 - 蓝色
    
    # 应用颜色
    alpha = 0.7
    overlay[all_three] = (1 - alpha) * overlay[all_three] + alpha * np.array([1, 1, 1])      # 白色
    overlay[out1_and_2] = (1 - alpha) * overlay[out1_and_2] + alpha * np.array([1, 1, 0])    # 黄色
    overlay[out1_and_3] = (1 - alpha) * overlay[out1_and_3] + alpha * np.array([1, 0, 1])    # 紫色
    overlay[out2_and_3] = (1 - alpha) * overlay[out2_and_3] + alpha * np.array([0, 1, 1])    # 青色
    overlay[out1_only] = (1 - alpha) * overlay[out1_only] + alpha * np.array([0, 1, 0])      # 绿色
    overlay[out2_only] = (1 - alpha) * overlay[out2_only] + alpha * np.array([1, 0, 0])      # 红色
    overlay[out3_only] = (1 - alpha) * overlay[out3_only] + alpha * np.array([0, 0, 1])      # 蓝色
    
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


def find_ground_truth(img_file, gt_dir):
    """查找对应的真实标签文件"""
    base_name = os.path.splitext(img_file)[0]
    
    # DRIVE数据集的标签命名模式
    gt_name = f"{base_name}_manual1.png"
    gt_path = os.path.join(gt_dir, gt_name)
    
    if os.path.exists(gt_path):
        return gt_path
    return None


def fixed_individual_outputs_inference():
    # 配置 - 与原始inference保持一致
    fsg_model_path = '/root/FSG-Net-pytorch/model_ckpts/FSG-Net-DRIVE.pt'
    val_x_path = "/root/FSG-Net-pytorch/data/DRIVE/val/input"
    val_y_path = "/root/FSG-Net-pytorch/data/DRIVE/val/label"  # 注意是label不是gt
    
    if not os.path.exists(val_x_path):
        print(f"Input path not found: {val_x_path}")
        return
    if not os.path.exists(val_y_path):
        print(f"Label path not found: {val_y_path}")
        return
    
    # 输出目录
    output_dir = './fixed_individual_outputs_comparison'
    
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
    
    # 为每个输出创建子目录
    for out_name in ['out1', 'out2', 'out3']:
        os.makedirs(f"{output_dir}/{out_name}_original", exist_ok=True)
        os.makedirs(f"{output_dir}/{out_name}_refined", exist_ok=True)
        os.makedirs(f"{output_dir}/{out_name}_improvement", exist_ok=True)
        os.makedirs(f"{output_dir}/{out_name}_vs_gt", exist_ok=True)
    
    # 对比图
    os.makedirs(f"{output_dir}/multi_output_comparison", exist_ok=True)
    os.makedirs(f"{output_dir}/refined_comparison", exist_ok=True)
    
    # 查找图像文件
    all_files = os.listdir(val_x_path)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
    
    if len(image_files) == 0:
        print("No image files found!")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    # 统计所有图像的指标
    all_metrics = {
        'out1_original': [], 'out1_refined': [],
        'out2_original': [], 'out2_refined': [],
        'out3_original': [], 'out3_refined': []
    }
    
    all_improvement_stats = {
        'out1': [], 'out2': [], 'out3': []
    }
    
    # 处理每张图像
    for i, img_file in enumerate(image_files):
        print(f"\n{'='*80}")
        print(f"Processing {i+1}/{len(image_files)}: {img_file}")
        print("="*80)
        
        try:
            # 加载图像
            img_path = os.path.join(val_x_path, img_file)
            image = Image.open(img_path).convert('RGB')
            print(f"  Original image size: {image.size}")
            
            # 按照原始dataloader方式预处理图像
            img_tensor = preprocess_image_like_original(image)
            img_tensor = img_tensor.unsqueeze(0).to(device)  # 添加batch维度
            print(f"  Processed tensor shape: {img_tensor.shape}")
            
            # 查找对应的真实标签
            gt_path = find_ground_truth(img_file, val_y_path)
            ground_truth = None
            if gt_path:
                try:
                    gt_image = Image.open(gt_path).convert('L')
                    print(f"  Original GT size: {gt_image.size}")
                    
                    # 按照原始dataloader方式预处理标签
                    gt_tensor = preprocess_target_like_original(gt_image)
                    ground_truth = gt_tensor.squeeze().numpy()  # [H, W]
                    print(f"  Processed GT shape: {ground_truth.shape}")
                    print(f"  GT unique values: {np.unique(ground_truth)}")
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
                
                outputs = {
                    'out1': out1,
                    'out2': out2, 
                    'out3': out3
                }
                
                print(f"  Output shapes - out1: {out1.shape}, out2: {out2.shape}, out3: {out3.shape}")
                for name, output in outputs.items():
                    print(f"  {name} range: [{output.min():.3f}, {output.max():.3f}]")
            
            # 清除缓存，为新图像准备
            region_grower.clear_cache()
            
            # 对每个输出进行区域生长细化
            original_tensor = img_tensor.squeeze()  # [3, H, W]
            refined_outputs = {}
            binary_outputs = {}
            grown_outputs = {}
            thresholds = {}
            
            print(f"\n=== Individual Output Refinement ===")
            for out_name, output in outputs.items():
                print(f"\n--- Refining {out_name.upper()} ---")
                refined, binary, grown, threshold = region_grower.refine_prediction(
                    output, original_tensor, out_name
                )
                refined_outputs[out_name] = refined
                binary_outputs[out_name] = binary
                grown_outputs[out_name] = grown
                thresholds[out_name] = threshold
            
            # 保存原始输出和细化结果
            base_name = os.path.splitext(img_file)[0]
            
            for out_name in ['out1', 'out2', 'out3']:
                # 保存原始输出
                original_img = (outputs[out_name] * 255).astype(np.uint8)
                Image.fromarray(original_img).save(f"{output_dir}/{out_name}_original/{base_name}_{out_name}.png")
                
                # 保存细化结果
                refined_img = (refined_outputs[out_name] * 255).astype(np.uint8)
                Image.fromarray(refined_img).save(f"{output_dir}/{out_name}_refined/{base_name}_{out_name}_refined.png")
            
            # 计算指标并创建可视化（如果有真实标签）
            if ground_truth is not None:
                print(f"\n=== Metrics Comparison (Using Original Calculation) ===")
                
                # 计算指标
                for out_name in ['out1', 'out2', 'out3']:
                    # 原始输出指标
                    original_metrics = calculate_metrics_like_original(outputs[out_name], ground_truth)
                    all_metrics[f'{out_name}_original'].append(original_metrics)
                    
                    # 细化输出指标
                    refined_metrics = calculate_metrics_like_original(refined_outputs[out_name], ground_truth)
                    all_metrics[f'{out_name}_refined'].append(refined_metrics)
                    
                    print(f"\n  {out_name.upper()} Metrics:")
                    print(f"    Original - F1: {original_metrics['f1']:.4f}, Precision: {original_metrics['precision']:.4f}, Recall: {original_metrics['recall']:.4f}")
                    print(f"    Refined  - F1: {refined_metrics['f1']:.4f}, Precision: {refined_metrics['precision']:.4f}, Recall: {refined_metrics['recall']:.4f}")
                    print(f"    F1 Change: {refined_metrics['f1'] - original_metrics['f1']:+.4f}")
                    
                    # 创建改进效果图
                    # 需要反归一化图像用于可视化
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    vis_image = original_tensor * std + mean
                    vis_image = torch.clamp(vis_image, 0, 1).numpy()
                    
                    improvement_overlay, improvement_stats = create_improvement_overlay(
                        vis_image, outputs[out_name], refined_outputs[out_name], ground_truth
                    )
                    improvement_img = (improvement_overlay * 255).astype(np.uint8)
                    Image.fromarray(improvement_img).save(f"{output_dir}/{out_name}_improvement/{base_name}_{out_name}_improvement.png")
                    
                    # 保存改进统计
                    all_improvement_stats[out_name].append(improvement_stats)
                    
                    print(f"    Improvement Stats:")
                    print(f"      ✅ 新增正确: {improvement_stats['added_correct']}")
                    print(f"      ❌ 新增错误: {improvement_stats['added_incorrect']}")
                    print(f"      📈 净改进: {improvement_stats['net_improvement']}")
                    print(f"      📊 假阳性净变化: {improvement_stats['net_false_positive_change']}")
                    
                    # 创建与GT对比图
                    gt_overlay = create_overlay_image(vis_image, refined_outputs[out_name], ground_truth)
                    gt_overlay_img = (gt_overlay * 255).astype(np.uint8)
                    Image.fromarray(gt_overlay_img).save(f"{output_dir}/{out_name}_vs_gt/{base_name}_{out_name}_vs_gt.png")
            
            # 创建多输出对比图
            # 使用反归一化的图像用于可视化
            if ground_truth is not None:
                mean = torch.tensor([0.485, 0.456, 0.406], device=original_tensor.device).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=original_tensor.device).view(3, 1, 1)
                vis_image = original_tensor * std + mean
                vis_image = torch.clamp(vis_image, 0, 1).cpu().numpy()
            else:
                # 如果没有GT，使用简单的归一化
                vis_image = (original_tensor.cpu().numpy() + 1) / 2  # 假设范围是[-1,1]
            
            # 原始输出对比
            multi_original_overlay = create_multi_output_comparison_overlay(
                vis_image, outputs['out1'], outputs['out2'], outputs['out3']
            )
            multi_original_img = (multi_original_overlay * 255).astype(np.uint8)
            Image.fromarray(multi_original_img).save(f"{output_dir}/multi_output_comparison/{base_name}_original_comparison.png")
            
            # 细化输出对比
            multi_refined_overlay = create_multi_output_comparison_overlay(
                vis_image, refined_outputs['out1'], refined_outputs['out2'], refined_outputs['out3']
            )
            multi_refined_img = (multi_refined_overlay * 255).astype(np.uint8)
            Image.fromarray(multi_refined_img).save(f"{output_dir}/refined_comparison/{base_name}_refined_comparison.png")
            
            print(f"\n  Results saved for {base_name}")
            
        except Exception as e:
            print(f"  Error processing {img_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 计算并显示总体统计
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS - FIXED TO MATCH ORIGINAL INFERENCE")
    print("="*80)
    
    if len(all_metrics['out1_original']) > 0:
        # 计算平均指标
        print(f"\n📊 Average Metrics Comparison:")
        print(f"{'Output':<8} {'Type':<10} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Accuracy':<10}")
        print("-" * 60)
        
        for out_name in ['out1', 'out2', 'out3']:
            # 原始指标
            orig_metrics = all_metrics[f'{out_name}_original']
            if orig_metrics:
                avg_f1_orig = np.mean([m['f1'] for m in orig_metrics])
                avg_prec_orig = np.mean([m['precision'] for m in orig_metrics])
                avg_recall_orig = np.mean([m['recall'] for m in orig_metrics])
                avg_acc_orig = np.mean([m['accuracy'] for m in orig_metrics])
                
                print(f"{out_name.upper():<8} {'Original':<10} {avg_f1_orig:<8.4f} {avg_prec_orig:<10.4f} {avg_recall_orig:<8.4f} {avg_acc_orig:<10.4f}")
            
            # 细化指标
            ref_metrics = all_metrics[f'{out_name}_refined']
            if ref_metrics:
                avg_f1_ref = np.mean([m['f1'] for m in ref_metrics])
                avg_prec_ref = np.mean([m['precision'] for m in ref_metrics])
                avg_recall_ref = np.mean([m['recall'] for m in ref_metrics])
                avg_acc_ref = np.mean([m['accuracy'] for m in ref_metrics])
                
                print(f"{'':<8} {'Refined':<10} {avg_f1_ref:<8.4f} {avg_prec_ref:<10.4f} {avg_recall_ref:<8.4f} {avg_acc_ref:<10.4f}")
                
                # 计算改进
                if orig_metrics:
                    f1_improvement = avg_f1_ref - avg_f1_orig
                    print(f"{'':<8} {'Change':<10} {f1_improvement:+<8.4f}")
            
            print()
        
        # 检查是否F1分数现在与原始inference一致
        print(f"\n🔍 F1 Score Verification:")
        for out_name in ['out1', 'out2', 'out3']:
            orig_metrics = all_metrics[f'{out_name}_original']
            if orig_metrics:
                avg_f1 = np.mean([m['f1'] for m in orig_metrics])
                print(f"  {out_name.upper()} Original F1: {avg_f1:.4f}")
                if avg_f1 > 0.8:
                    print(f"    ✅ 这个F1分数与你的原始inference接近!")
                else:
                    print(f"    ⚠️  F1分数仍然偏低，可能还有其他差异")
        
        # 改进统计汇总
        print(f"\n🔍 Region Growing Improvement Summary:")
        print(f"{'Output':<8} {'Total Added✅':<12} {'Total Incorrect❌':<15} {'Net Improvement📈':<18} {'False Pos Change📊':<18}")
        print("-" * 80)
        
        for out_name in ['out1', 'out2', 'out3']:
            stats_list = all_improvement_stats[out_name]
            if stats_list:
                total_added_correct = sum([s['added_correct'] for s in stats_list])
                total_added_incorrect = sum([s['added_incorrect'] for s in stats_list])
                total_net_improvement = sum([s['net_improvement'] for s in stats_list])
                total_fp_change = sum([s['net_false_positive_change'] for s in stats_list])
                
                print(f"{out_name.upper():<8} {total_added_correct:<12} {total_added_incorrect:<15} {total_net_improvement:<18} {total_fp_change:<18}")
        
        # 排名分析
        print(f"\n🏆 Performance Ranking:")
        
        # 按平均F1分数排名（原始）
        original_f1_scores = {}
        refined_f1_scores = {}
        improvement_scores = {}
        
        for out_name in ['out1', 'out2', 'out3']:
            orig_metrics = all_metrics[f'{out_name}_original']
            ref_metrics = all_metrics[f'{out_name}_refined']
            
            if orig_metrics and ref_metrics:
                orig_f1 = np.mean([m['f1'] for m in orig_metrics])
                ref_f1 = np.mean([m['f1'] for m in ref_metrics])
                improvement = ref_f1 - orig_f1
                
                original_f1_scores[out_name] = orig_f1
                refined_f1_scores[out_name] = ref_f1
                improvement_scores[out_name] = improvement
        
        # 排序并显示
        if original_f1_scores:
            print(f"\n  Original F1 Ranking:")
            sorted_orig = sorted(original_f1_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (out_name, score) in enumerate(sorted_orig, 1):
                print(f"    {rank}. {out_name.upper()}: {score:.4f}")
            
            print(f"\n  Refined F1 Ranking:")
            sorted_ref = sorted(refined_f1_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (out_name, score) in enumerate(sorted_ref, 1):
                print(f"    {rank}. {out_name.upper()}: {score:.4f}")
            
            print(f"\n  F1 Improvement Ranking:")
            sorted_imp = sorted(improvement_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (out_name, score) in enumerate(sorted_imp, 1):
                print(f"    {rank}. {out_name.upper()}: {score:+.4f}")
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"✅ All results saved to: {output_dir}")
    print(f"\n📁 Directory Structure:")
    print(f"  - out1_original/     (Out1原始输出)")
    print(f"  - out1_refined/      (Out1细化结果)")
    print(f"  - out1_improvement/  (Out1改进效果图)")
    print(f"  - out1_vs_gt/        (Out1与GT对比)")
    print(f"  - out2_*/ out3_*/    (Out2和Out3的相应结果)")
    print(f"  - multi_output_comparison/  (原始输出对比)")
    print(f"  - refined_comparison/       (细化输出对比)")
    
    print(f"\n🔧 Fixed Issues:")
    print(f"  ✅ 使用Image2Image_resize预处理 (resize到608x608)")
    print(f"  ✅ 使用ImageNet标准化 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])")
    print(f"  ✅ 使用正确的GT路径 (/val/label/)")
    print(f"  ✅ 使用原始的二值化处理 (< 128 = 0, >= 128 = 1)")
    print(f"  ✅ 使用原始的metrics计算方法")
    print(f"  ✅ 移除了圆形掩码 (与原始inference保持一致)")
    
    print(f"\n📈 Expected Results:")
    print(f"  - F1分数现在应该与你的原始inference(80+)接近")
    print(f"  - 可以准确对比三个输出的区域生长效果")
    print(f"  - 基于真实的F1改进来选择最佳输出")
    
    print(f"\n🎨 Visualization Color Codes:")
    print(f"  Multi-output comparison:")
    print(f"    🟢 绿色 = 仅Out1")
    print(f"    🔴 红色 = 仅Out2") 
    print(f"    🔵 蓝色 = 仅Out3")
    print(f"    🟡 黄色 = Out1+Out2")
    print(f"    🟣 紫色 = Out1+Out3")
    print(f"    🔵 青色 = Out2+Out3")
    print(f"    ⚪ 白色 = 三个都有")
    
    print(f"\n  Improvement overlay:")
    print(f"    🟢 绿色 = 细化新增的正确像素 ⭐")
    print(f"    🔴 红色 = 细化新增的错误像素")
    print(f"    🔵 蓝色 = 细化丢失的正确像素")
    print(f"    🟡 黄色 = 细化减少的错误像素 ⭐")
    
    print(f"\n📈 Key Insights:")
    print(f"  - F1分数现在应该接近你的原始inference结果")
    print(f"  - 查看F1改进排名，了解哪个输出最适合区域生长")
    print(f"  - 绿色区域多的输出表示区域生长效果好")
    print(f"  - 对比净改进分数，选择最佳输出进行后处理")


if __name__ == "__main__":
    fixed_individual_outputs_inference()