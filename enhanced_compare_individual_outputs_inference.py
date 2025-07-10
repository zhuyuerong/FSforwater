import torch
import torch.nn as nn
import cv2
import numpy as np
from models.backbones.FSGNet import FSGNet
import torchvision.transforms.functional as tf
import argparse
from PIL import Image
import os
import itertools
import json
from tqdm import tqdm
import time


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
        
        if seed_binary.sum() == 0:
            return np.zeros_like(fsg_prediction, dtype=np.float32)
        
        # 2. 基于梯度阈值选择候选血管像素
        vessel_candidates = (gradient_magnitude >= self.vessel_threshold).astype(np.uint8)
        
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
                break
            grown_mask = new_grown
        
        return grown_mask.astype(np.float32)
    
    def get_dynamic_threshold(self, prediction):
        """计算动态阈值"""
        if len(prediction[prediction > 0]) > 0:
            pred_mean = prediction.mean()
            pred_std = prediction.std()
            # 使用均值 + 0.5 * 标准差作为阈值
            threshold = pred_mean + 0.5 * pred_std
        else:
            threshold = 0.5
        return threshold
    
    def refine_prediction(self, fsg_output, original_image, output_name="output"):
        """细化FSG预测结果"""
        # fsg_output: [H, W] numpy array (FSG预测概率 0-1)
        # original_image: [3, H, W] tensor (已经过ImageNet归一化)
        
        # 1. 反归一化图像以进行梯度计算
        if not hasattr(self, '_enhanced_cached'):
            # 反归一化ImageNet标准化 - 确保在相同设备上
            mean = torch.tensor([0.485, 0.456, 0.406], device=original_image.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=original_image.device).view(3, 1, 1)
            unnormalized_image = original_image * std + mean
            unnormalized_image = torch.clamp(unnormalized_image, 0, 1)
            
            enhanced = self.preprocess_image(unnormalized_image)
            gradient = self.calculate_gradient(enhanced)
            
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
        
        return final_result, fsg_binary, grown_result, fsg_threshold
    
    def clear_cache(self):
        """清除缓存，用于处理新图像"""
        if hasattr(self, '_enhanced_cached'):
            delattr(self, '_enhanced_cached')
        if hasattr(self, '_gradient_cached'):
            delattr(self, '_gradient_cached')
    
    def update_params(self, mu_f=None, sigma_f=None, alpha=None):
        """更新参数"""
        if mu_f is not None:
            self.mu_f = mu_f
        if sigma_f is not None:
            self.sigma_f = sigma_f
        if alpha is not None:
            self.alpha = alpha
        self.vessel_threshold = self.mu_f + self.alpha * self.sigma_f


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


class GridSearchOptimizer:
    """区域生长参数网格搜索优化器"""
    
    def __init__(self, fsg_model_path, val_x_path, val_y_path, device='cuda'):
        self.device = device
        self.val_x_path = val_x_path
        self.val_y_path = val_y_path
        
        # 加载预训练的FSG模型
        print("Loading FSG model...")
        self.fsg_model = load_fsg_model(fsg_model_path, device)
        
        # 加载验证数据
        self.load_validation_data()
        
        # 参数搜索空间
        self.param_space = {
            'mu_f': [0.05, 0.0789, 0.1, 0.12, 0.15],
            'sigma_f': [0.05, 0.0774, 0.1, 0.12, 0.15],
            'alpha': [0.5, 1.0, 1.5, 2.0, 2.5],
            'apply_to': ['out1', 'out2', 'out3']
        }
        
        # 结果存储
        self.results = []
        
    def load_validation_data(self):
        """加载验证数据"""
        print("Loading validation data...")
        
        all_files = os.listdir(self.val_x_path)
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        
        self.validation_data = []
        
        for img_file in image_files:
            try:
                # 加载图像
                img_path = os.path.join(self.val_x_path, img_file)
                image = Image.open(img_path).convert('RGB')
                img_tensor = preprocess_image_like_original(image)
                
                # 查找对应的真实标签
                gt_path = find_ground_truth(img_file, self.val_y_path)
                if gt_path:
                    gt_image = Image.open(gt_path).convert('L')
                    gt_tensor = preprocess_target_like_original(gt_image)
                    ground_truth = gt_tensor.squeeze().numpy()
                    
                    self.validation_data.append({
                        'image': img_tensor,
                        'ground_truth': ground_truth,
                        'filename': img_file
                    })
                
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
                continue
        
        print(f"Loaded {len(self.validation_data)} validation samples")
    
    def evaluate_params(self, params, num_samples=None):
        """评估给定参数组合的性能"""
        if num_samples is None:
            num_samples = len(self.validation_data)
        
        # 创建区域生长器
        region_grower = SimpleRegionGrowing(
            mu_f=params['mu_f'],
            sigma_f=params['sigma_f'],
            alpha=params['alpha']
        )
        
        all_metrics = []
        
        with torch.no_grad():
            for i, data in enumerate(self.validation_data[:num_samples]):
                try:
                    img_tensor = data['image'].unsqueeze(0).to(self.device)
                    ground_truth = data['ground_truth']
                    
                    # 清除缓存
                    region_grower.clear_cache()
                    
                    # FSG推理
                    out1, out2, out3 = self.fsg_model(img_tensor)
                    
                    # 选择要应用区域生长的输出
                    if params['apply_to'] == 'out1':
                        target_output = out1
                    elif params['apply_to'] == 'out2':
                        target_output = out2
                    elif params['apply_to'] == 'out3':
                        target_output = out3
                    
                    # 应用区域生长
                    output_np = target_output.squeeze().cpu().numpy()
                    original_tensor = img_tensor.squeeze()
                    
                    refined_result, _, _, _ = region_grower.refine_prediction(
                        output_np, original_tensor
                    )
                    
                    # 计算指标
                    metrics = calculate_metrics_like_original(refined_result, ground_truth)
                    all_metrics.append(metrics)
                    
                except Exception as e:
                    print(f"Error evaluating sample {i}: {e}")
                    continue
        
        # 计算平均指标
        if all_metrics:
            avg_metrics = {
                'f1': np.mean([m['f1'] for m in all_metrics]),
                'precision': np.mean([m['precision'] for m in all_metrics]),
                'recall': np.mean([m['recall'] for m in all_metrics]),
                'accuracy': np.mean([m['accuracy'] for m in all_metrics])
            }
        else:
            avg_metrics = {'f1': 0, 'precision': 0, 'recall': 0, 'accuracy': 0}
        
        return avg_metrics
    
    def grid_search(self, max_combinations=100, quick_eval_samples=5):
        """执行网格搜索"""
        print("Starting grid search...")
        
        # 生成所有参数组合
        param_names = list(self.param_space.keys())
        param_values = [self.param_space[name] for name in param_names]
        all_combinations = list(itertools.product(*param_values))
        
        # 如果组合数太多，随机采样
        if len(all_combinations) > max_combinations:
            print(f"Too many combinations ({len(all_combinations)}), sampling {max_combinations}")
            np.random.shuffle(all_combinations)
            all_combinations = all_combinations[:max_combinations]
        
        print(f"Testing {len(all_combinations)} parameter combinations...")
        
        best_f1 = 0
        best_params = None
        
        for i, combination in enumerate(tqdm(all_combinations, desc="Grid search")):
            params = dict(zip(param_names, combination))
            
            try:
                start_time = time.time()
                metrics = self.evaluate_params(params, quick_eval_samples)
                eval_time = time.time() - start_time
                
                result = {
                    'params': params,
                    'metrics': metrics,
                    'eval_time': eval_time
                }
                self.results.append(result)
                
                # 更新最佳结果
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_params = params
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{len(all_combinations)}, best F1 so far: {best_f1:.4f}")
                    
            except Exception as e:
                print(f"Error in combination {i}: {e}")
                continue
        
        # 排序结果
        self.results.sort(key=lambda x: x['metrics']['f1'], reverse=True)
        
        return best_params, best_f1
    
    def save_results(self, save_path='grid_search_results.json'):
        """保存搜索结果"""
        if not self.results:
            print("No results to save!")
            return
        
        save_data = {
            'best_params': self.results[0]['params'],
            'best_metrics': self.results[0]['metrics'],
            'all_results': self.results,
            'param_space': self.param_space,
            'num_samples': len(self.validation_data)
        }
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Results saved to {save_path}")


def enhanced_individual_outputs_inference():
    # 配置 - 与原始inference保持一致
    fsg_model_path = '/root/FSG-Net-pytorch/model_ckpts/FSG-Net-DRIVE.pt'
    val_x_path = "/root/FSG-Net-pytorch/data/DRIVE/val/input"
    val_y_path = "/root/FSG-Net-pytorch/data/DRIVE/val/label"
    
    if not os.path.exists(val_x_path):
        print(f"Input path not found: {val_x_path}")
        return
    if not os.path.exists(val_y_path):
        print(f"Label path not found: {val_y_path}")
        return
    
    # 输出目录
    output_dir = './enhanced_individual_outputs_with_grid_search'
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: 网格搜索最佳参数
    print("="*80)
    print("STEP 1: GRID SEARCH FOR OPTIMAL PARAMETERS")
    print("="*80)
    
    optimizer = GridSearchOptimizer(fsg_model_path, val_x_path, val_y_path, device)
    
    best_params, best_f1 = optimizer.grid_search(
        max_combinations=60,  # 5*5*5*3 = 375种组合，采样60个
        quick_eval_samples=5  # 每个组合用5个样本快速评估
    )
    
    print(f"\n🎯 BEST PARAMETERS FOUND:")
    print(f"  mu_f: {best_params['mu_f']}")
    print(f"  sigma_f: {best_params['sigma_f']}")
    print(f"  alpha: {best_params['alpha']}")
    print(f"  apply_to: {best_params['apply_to']}")
    print(f"  Best F1: {best_f1:.4f}")
    
    # 保存网格搜索结果
    optimizer.save_results(os.path.join(output_dir, 'grid_search_results.json'))
    
    # Step 2: 使用最佳参数进行完整推理
    print("\n" + "="*80)
    print("STEP 2: FULL INFERENCE WITH OPTIMIZED PARAMETERS")
    print("="*80)
    
    # 加载FSG模型
    print("Loading FSG-Net model...")
    fsg_model = load_fsg_model(fsg_model_path, device)
    
    # 使用最佳参数初始化区域生长
    region_grower = SimpleRegionGrowing(
        mu_f=best_params['mu_f'],
        sigma_f=best_params['sigma_f'],
        alpha=best_params['alpha']
    )
    
    # 创建输出目录
    for out_name in ['out1', 'out2', 'out3']:
        os.makedirs(f"{output_dir}/{out_name}_original", exist_ok=True)
        os.makedirs(f"{output_dir}/{out_name}_refined", exist_ok=True)
    
    # 查找图像文件
    all_files = os.listdir(val_x_path)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
    
    if len(image_files) == 0:
        print("No image files found!")
        return
    
    print(f"Processing {len(image_files)} images with optimized parameters...")
    
    # 统计所有图像的指标
    all_metrics = {
        'out1_original': [], 'out1_refined': [],
        'out2_original': [], 'out2_refined': [],
        'out3_original': [], 'out3_refined': []
    }
    
    # 处理每张图像
    for i, img_file in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {img_file}")
        
        try:
            # 加载图像
            img_path = os.path.join(val_x_path, img_file)
            image = Image.open(img_path).convert('RGB')
            
            # 按照原始dataloader方式预处理图像
            img_tensor = preprocess_image_like_original(image)
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # 查找对应的真实标签
            gt_path = find_ground_truth(img_file, val_y_path)
            ground_truth = None
            if gt_path:
                try:
                    gt_image = Image.open(gt_path).convert('L')
                    gt_tensor = preprocess_target_like_original(gt_image)
                    ground_truth = gt_tensor.squeeze().numpy()
                except Exception as e:
                    print(f"  Error loading ground truth: {e}")
                    ground_truth = None
            
            # FSG推理 - 获取三个输出
            with torch.no_grad():
                fsg_out1, fsg_out2, fsg_out3 = fsg_model(img_tensor)
                
                # 转换为numpy
                out1 = fsg_out1.squeeze().cpu().numpy()
                out2 = fsg_out2.squeeze().cpu().numpy()
                out3 = fsg_out3.squeeze().cpu().numpy()
                
                outputs = {
                    'out1': out1,
                    'out2': out2, 
                    'out3': out3
                }
            
            # 清除缓存，为新图像准备
            region_grower.clear_cache()
            
            # 只对最佳输出应用区域生长
            original_tensor = img_tensor.squeeze()
            refined_outputs = {}
            
            best_output = best_params['apply_to']
            print(f"  Applying RG to {best_output} with optimized parameters")
            
            for out_name, output in outputs.items():
                if out_name == best_output:
                    # 应用区域生长
                    refined, _, _, _ = region_grower.refine_prediction(
                        output, original_tensor, out_name
                    )
                    refined_outputs[out_name] = refined
                else:
                    # 其他输出保持原样
                    refined_outputs[out_name] = output
            
            # 保存结果
            base_name = os.path.splitext(img_file)[0]
            
            for out_name in ['out1', 'out2', 'out3']:
                # 保存原始输出
                original_img = (outputs[out_name] * 255).astype(np.uint8)
                Image.fromarray(original_img).save(f"{output_dir}/{out_name}_original/{base_name}_{out_name}.png")
                
                # 保存细化结果
                refined_img = (refined_outputs[out_name] * 255).astype(np.uint8)
                Image.fromarray(refined_img).save(f"{output_dir}/{out_name}_refined/{base_name}_{out_name}_refined.png")
            
            # 计算指标（如果有真实标签）
            if ground_truth is not None:
                for out_name in ['out1', 'out2', 'out3']:
                    # 原始输出指标
                    original_metrics = calculate_metrics_like_original(outputs[out_name], ground_truth)
                    all_metrics[f'{out_name}_original'].append(original_metrics)
                    
                    # 细化输出指标
                    refined_metrics = calculate_metrics_like_original(refined_outputs[out_name], ground_truth)
                    all_metrics[f'{out_name}_refined'].append(refined_metrics)
                    
                    if out_name == best_output:
                        print(f"  {out_name.upper()} (Optimized):")
                        print(f"    Original F1: {original_metrics['f1']:.4f}")
                        print(f"    Refined F1:  {refined_metrics['f1']:.4f}")
                        print(f"    Improvement: {refined_metrics['f1'] - original_metrics['f1']:+.4f}")
            
        except Exception as e:
            print(f"  Error processing {img_file}: {str(e)}")
            continue
    
    # 计算并显示总体统计
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS WITH GRID-SEARCHED PARAMETERS")
    print("="*80)
    
    if len(all_metrics['out1_original']) > 0:
        print(f"\n📊 Results Summary:")
        print(f"Best parameters applied to: {best_params['apply_to'].upper()}")
        print()
        
        for out_name in ['out1', 'out2', 'out3']:
            orig_metrics = all_metrics[f'{out_name}_original']
            ref_metrics = all_metrics[f'{out_name}_refined']
            
            if orig_metrics and ref_metrics:
                avg_f1_orig = np.mean([m['f1'] for m in orig_metrics])
                avg_f1_ref = np.mean([m['f1'] for m in ref_metrics])
                improvement = avg_f1_ref - avg_f1_orig
                
                marker = "🎯" if out_name == best_params['apply_to'] else "  "
                print(f"{marker} {out_name.upper()}:")
                print(f"    Original F1: {avg_f1_orig:.4f}")
                print(f"    Refined F1:  {avg_f1_ref:.4f}")
                print(f"    Improvement: {improvement:+.4f}")
                print()
        
        # 显示网格搜索的top 5结果
        print(f"🏆 Top 5 Parameter Combinations:")
        for i, result in enumerate(optimizer.results[:5]):
            params = result['params']
            metrics = result['metrics']
            print(f"  {i+1}. F1={metrics['f1']:.4f}, mu_f={params['mu_f']}, sigma_f={params['sigma_f']}, alpha={params['alpha']}, apply_to={params['apply_to']}")
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"✅ Grid search completed with {len(optimizer.results)} combinations tested")
    print(f"✅ Best parameters found and applied to {len(image_files)} images")
    print(f"✅ All results saved to: {output_dir}")
    print(f"\n📁 Directory Structure:")
    print(f"  - grid_search_results.json  (完整的网格搜索结果)")
    print(f"  - out1_original/            (Out1原始输出)")
    print(f"  - out1_refined/             (Out1细化结果)")
    print(f"  - out2_original/            (Out2原始输出)")
    print(f"  - out2_refined/             (Out2细化结果)")
    print(f"  - out3_original/            (Out3原始输出)")
    print(f"  - out3_refined/             (Out3细化结果)")
    
    print(f"\n🎯 Key Findings:")
    print(f"  - Best parameter combination: mu_f={best_params['mu_f']}, sigma_f={best_params['sigma_f']}, alpha={best_params['alpha']}")
    print(f"  - Best output for RG: {best_params['apply_to']}")
    print(f"  - Expected F1 improvement: +{best_f1:.4f}")
    
    print(f"\n💡 Next Steps:")
    print(f"  1. Review the grid_search_results.json for detailed analysis")
    print(f"  2. Use these optimized parameters in your training configuration")
    print(f"  3. Consider fine-tuning around the best parameters for further improvement")


if __name__ == "__main__":
    enhanced_individual_outputs_inference()