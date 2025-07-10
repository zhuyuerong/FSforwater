
import os
import numpy as np
import cv2
from PIL import Image
import torch
from skimage import filters, morphology, measure
# 修复frangi导入问题
try:
    from skimage.filters import frangi
except ImportError:
    try:
        from skimage.feature import frangi
    except ImportError:
        # 如果都没有，我们自己实现一个简单版本
        def frangi(image, sigmas=[1, 2, 3, 4], alpha=0.5, beta=0.5, gamma=15):
            """简化的Frangi滤波器实现"""
            responses = []
            for sigma in sigmas:
                # 使用高斯模糊和拉普拉斯算子近似
                blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigma)
                laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
                responses.append(np.abs(laplacian))
            
            # 取最大响应
            return np.maximum.reduce(responses)

import itertools
import json
from tqdm import tqdm
class TraditionalVesselSegmentation:
    """纯传统方法的血管分割"""
    
    def __init__(self, params=None):
        # 默认参数
        self.default_params = {
            # Frangi滤波器参数
            'frangi_sigmas': [1, 2, 3, 4],
            'frangi_alpha': 0.5,
            'frangi_beta': 0.5,
            'frangi_gamma': 15,
            
            # 自适应阈值参数
            'adaptive_block_size': 15,
            'adaptive_c': 2,
            
            # 区域生长参数
            'seed_threshold': 0.3,  # Frangi响应阈值
            'growth_threshold': 0.1,  # 区域生长梯度阈值
            'kernel_size': 3,  # 邻域大小
            'max_iterations': 5,
            
            # 形态学参数
            'morph_open_size': 2,
            'morph_close_size': 3,
            
            # 连通域过滤
            'min_area': 50,  # 最小连通域面积
        }
        
        self.params = params if params else self.default_params.copy()
    
    def preprocess_image(self, image_tensor):
        """预处理图像"""
        # image_tensor: [3, H, W] -> [H, W] grayscale
        if len(image_tensor.shape) == 3:
            # 转换为灰度图
            gray = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
        else:
            gray = image_tensor
        
        gray_np = gray.cpu().numpy() if isinstance(gray, torch.Tensor) else gray
        
        # 创建圆形掩码
        h, w = gray_np.shape
        center_y, center_x = h // 2, w // 2
        radius = min(center_y, center_x) * 0.9
        Y, X = np.ogrid[:h, :w]
        mask = ((X - center_x) ** 2 + (Y - center_y) ** 2) <= radius ** 2
        
        # 应用掩码
        masked_image = gray_np * mask.astype(np.float32)
        
        # 对比度增强（CLAHE）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply((masked_image * 255).astype(np.uint8))
        
        return enhanced.astype(np.float32) / 255.0, mask.astype(np.float32)
    
    def frangi_vessel_enhancement(self, image):
        """Frangi血管增强滤波"""
        frangi_response = frangi(
            image,
            sigmas=self.params['frangi_sigmas'],
            alpha=self.params['frangi_alpha'],
            beta=self.params['frangi_beta'],
            gamma=self.params['frangi_gamma']
        )
        return frangi_response
    
    def extract_seed_points(self, frangi_response, mask):
        """提取种子点"""
        # 使用Frangi响应的高值区域作为种子
        seeds = (frangi_response > self.params['seed_threshold']) * mask
        return seeds.astype(np.uint8)
    
    def traditional_region_growing(self, image, seeds, mask):
        """传统区域生长算法"""
        # 计算图像梯度
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 归一化梯度
        if gradient_magnitude.max() > 0:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
        
        # 创建膨胀核
        kernel_size = self.params['kernel_size']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 初始化
        current_vessels = seeds.copy()
        growth_threshold = self.params['growth_threshold']
        
        print(f"Starting region growing with kernel size {kernel_size}x{kernel_size}")
        print(f"Initial seeds: {seeds.sum()} pixels")
        print(f"Growth threshold: {growth_threshold}")
        
        # 迭代生长
        for iteration in range(self.params['max_iterations']):
            prev_count = current_vessels.sum()
            
            # 膨胀当前区域
            dilated = cv2.dilate(current_vessels, kernel, iterations=1)
            
            # 候选扩展区域
            candidates = dilated - current_vessels
            
            # 基于梯度的生长条件
            # 低梯度区域更可能是血管内部
            growth_condition = (gradient_magnitude <= growth_threshold) * mask.astype(np.uint8)
            
            # 应用生长条件
            new_growth = candidates * growth_condition
            
            # 更新血管区域
            current_vessels = current_vessels + new_growth
            
            current_count = current_vessels.sum()
            added = current_count - prev_count
            
            print(f"Iteration {iteration + 1}: added {added} pixels, total: {current_count}")
            
            # 如果没有新增长，提前停止
            if added == 0:
                print(f"Convergence reached at iteration {iteration + 1}")
                break
        
        return current_vessels.astype(np.float32)
    
    def post_process(self, vessels, mask):
        """后处理"""
        # 形态学操作
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (self.params['morph_open_size'], self.params['morph_open_size']))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                (self.params['morph_close_size'], self.params['morph_close_size']))
        
        # 开运算去噪
        cleaned = cv2.morphologyEx(vessels.astype(np.uint8), cv2.MORPH_OPEN, open_kernel)
        
        # 闭运算连接
        connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
        
        # 连通域过滤
        labeled = measure.label(connected)
        props = measure.regionprops(labeled)
        
        # 保留大于最小面积的连通域
        filtered = np.zeros_like(connected)
        for prop in props:
            if prop.area >= self.params['min_area']:
                filtered[labeled == prop.label] = 1
        
        # 应用掩码
        final_result = filtered * mask
        
        return final_result.astype(np.float32)
    
    def segment_vessels(self, image_tensor):
        """完整的血管分割流程"""
        # 1. 预处理
        processed_image, mask = self.preprocess_image(image_tensor)
        
        # 2. Frangi血管增强
        frangi_response = self.frangi_vessel_enhancement(processed_image)
        
        # 3. 提取种子点
        seeds = self.extract_seed_points(frangi_response, mask)
        
        if seeds.sum() == 0:
            print("Warning: No seed points found!")
            return np.zeros_like(processed_image)
        
        # 4. 区域生长
        vessels = self.traditional_region_growing(processed_image, seeds, mask)
        
        # 5. 后处理
        final_vessels = self.post_process(vessels, mask)
        
        return final_vessels, frangi_response, seeds
class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def define_parameter_grid(self):
        """定义参数网格"""
        param_grid = {
            'seed_threshold': [0.2, 0.3, 0.4],
            'growth_threshold': [0.05, 0.1, 0.15, 0.2],
            'kernel_size': [3, 4, 5],
            'max_iterations': [3, 5, 7],
            'frangi_gamma': [10, 15, 20],
            'morph_close_size': [2, 3, 4]
        }
        return param_grid
    
    def grid_search(self, max_combinations=50):
        """网格搜索最佳参数"""
        param_grid = self.define_parameter_grid()
        
        # 生成所有参数组合
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        # 限制组合数量
        if len(combinations) > max_combinations:
            combinations = combinations[:max_combinations]
            print(f"Limiting to first {max_combinations} combinations")
        
        print(f"Testing {len(combinations)} parameter combinations")
        
        # 获取测试图像
        image_files = [f for f in os.listdir(self.data_path) 
                      if f.lower().endswith(('.tif', '.png', '.jpg'))]
        test_files = image_files[:3]  # 使用前3张图进行参数搜索
        
        results = []
        
        for i, combination in enumerate(tqdm(combinations)):
            params = dict(zip(keys, combination))
            
            # 更新默认参数
            segmenter = TraditionalVesselSegmentation()
            segmenter.params.update(params)
            
            # 在测试图像上评估
            total_vessels = 0
            total_seeds = 0
            
            for img_file in test_files:
                try:
                    img_path = os.path.join(self.data_path, img_file)
                    image = Image.open(img_path).convert('RGB')
                    img_array = np.array(image) / 255.0
                    img_tensor = torch.tensor(img_array.transpose(2, 0, 1), dtype=torch.float32)
                    
                    result = segmenter.segment_vessels(img_tensor)
                    
                    # 检查返回值数量
                    if len(result) == 3:
                        vessels, frangi, seeds = result
                        total_vessels += vessels.sum()
                        total_seeds += seeds.sum()
                    else:
                        print(f"Unexpected return format: {len(result)} values")
                        total_vessels = 0
                        break
                    
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    total_vessels = 0
                    break
            
            # 评估指标（可以根据需要调整）
            score = total_vessels / (total_seeds + 1)  # 扩展比例
            
            results.append({
                'params': params,
                'score': score,
                'total_vessels': total_vessels,
                'total_seeds': total_seeds
            })
        
        # 排序结果
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # 保存结果
        with open(os.path.join(self.output_dir, 'grid_search_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTop 5 parameter combinations:")
        for i, result in enumerate(results[:5]):
            print(f"{i+1}. Score: {result['score']:.2f}, Params: {result['params']}")
        
        return results[0]['params']  # 返回最佳参数

def process_validation_set(data_path, output_dir, best_params=None):
    """处理验证集"""
    os.makedirs(f"{output_dir}/traditional_results", exist_ok=True)
    os.makedirs(f"{output_dir}/frangi_responses", exist_ok=True)
    os.makedirs(f"{output_dir}/seed_points", exist_ok=True)
    
    # 创建分割器
    segmenter = TraditionalVesselSegmentation(best_params)
    
    # 获取所有图像
    image_files = [f for f in os.listdir(data_path) 
                  if f.lower().endswith(('.tif', '.png', '.jpg'))]
    
    print(f"Processing {len(image_files)} images with traditional methods...")
    
    for i, img_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {img_file}")
        
        try:
            # 加载图像
            img_path = os.path.join(data_path, img_file)
            image = Image.open(img_path).convert('RGB')
            img_array = np.array(image) / 255.0
            img_tensor = torch.tensor(img_array.transpose(2, 0, 1), dtype=torch.float32)
            
            # 分割
            vessels, frangi_response, seeds = segmenter.segment_vessels(img_tensor)
            
            # 保存结果
            base_name = os.path.splitext(img_file)[0]
            
            # 保存血管分割结果
            vessel_img = (vessels * 255).astype(np.uint8)
            Image.fromarray(vessel_img).save(f"{output_dir}/traditional_results/{base_name}_vessels.png")
            
            # 保存Frangi响应
            frangi_normalized = (frangi_response / frangi_response.max() * 255).astype(np.uint8)
            Image.fromarray(frangi_normalized).save(f"{output_dir}/frangi_responses/{base_name}_frangi.png")
            
            # 保存种子点
            seed_img = (seeds * 255).astype(np.uint8)
            Image.fromarray(seed_img).save(f"{output_dir}/seed_points/{base_name}_seeds.png")
            
            print(f"  Vessels: {vessels.sum():.0f} pixels")
            print(f"  Seeds: {seeds.sum():.0f} pixels")
            print(f"  Expansion ratio: {vessels.sum()/(seeds.sum()+1):.2f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print(f"\nResults saved to {output_dir}/")
    print("- traditional_results/: Final vessel segmentation")
    print("- frangi_responses/: Frangi filter responses")
    print("- seed_points/: Initial seed points")

def main():
    # 配置路径
    data_path = '/root/FSG-Net-pytorch/data/DRIVE/val/input'
    output_dir = './traditional_vessel_results'
    
    print("=== Traditional Vessel Segmentation Pipeline ===")
    
    # 1. 参数优化
    print("\n1. Parameter Optimization")
    optimizer = ParameterOptimizer(data_path, output_dir)
    best_params = optimizer.grid_search(max_combinations=30)
    
    print(f"\nBest parameters found: {best_params}")
    
    # 2. 处理验证集
    print("\n2. Processing Validation Set")
    process_validation_set(data_path, output_dir, best_params)
    
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()