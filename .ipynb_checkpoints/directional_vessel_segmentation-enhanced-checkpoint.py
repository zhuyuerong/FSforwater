import os
import numpy as np
import cv2
from PIL import Image
import torch
from skimage import morphology, measure, filters
import itertools
import json
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

class DirectionalLineVesselSegmentation:
    """基于方向核线检测的血管分割 - 修复版"""
    
    def __init__(self, params=None):
        # 默认参数
        self.default_params = {
            # 图像增强参数
            'background_kernel_size': 101,  # 确保为奇数
            'clahe_clip_limit': 2.0,
            'clahe_tile_size': 8,
            
            # 多尺度检测参数
            'scales': [1.0, 1.5, 2.0],
            'gaussian_sigma': 1.0,
            
            # 线检测参数
            'line_response_threshold': 0.1,
            'local_max_radius': 3,
            'non_max_suppression': True,
            
            # Hessian增强
            'use_hessian': True,
            'hessian_sigma': [1.0, 2.0, 3.0],
            'hessian_beta': 0.5,
            
            # 区域生长参数
            'mu_f': 0.0789,
            'sigma_f': 0.0774,
            'alpha': 1.0,
            'adaptive_threshold': True,
            
            # 形态学参数
            'morph_open_size': 2,
            'morph_close_size': 3,
            'skeleton_cleaning': False,
            
            # 连通域过滤
            'min_area': 30,
            'max_area': 10000,
            'aspect_ratio_threshold': 10.0,
        }
        
        self.params = params if params else self.default_params.copy()
        self.create_multi_scale_kernels()
    
    def ensure_odd_kernel_size(self, size):
        """确保卷积核尺寸为奇数"""
        return size if size % 2 == 1 else size + 1
    
    def create_multi_scale_kernels(self):
        """创建多尺度方向检测核"""
        self.multi_kernels = {}
        
        # 基础核模板 - 增强对比度
        base_kernels = {
            '0°': np.array([[-1, -1, -1],
                           [ 3,  3,  3],
                           [-1, -1, -1]], dtype=np.float32),
            
            '45°': np.array([[-1, -1,  3],
                            [-1,  3, -1],
                            [ 3, -1, -1]], dtype=np.float32),
            
            '90°': np.array([[-1,  3, -1],
                            [-1,  3, -1],
                            [-1,  3, -1]], dtype=np.float32),
            
            '135°': np.array([[ 3, -1, -1],
                             [-1,  3, -1],
                             [-1, -1,  3]], dtype=np.float32)
        }
        
        # 为每个尺度创建核
        for scale in self.params['scales']:
            scale_kernels = {}
            for direction, kernel in base_kernels.items():
                if scale == 1.0:
                    scale_kernels[direction] = kernel
                else:
                    # 扩展核尺寸
                    size = int(3 * scale)
                    size = self.ensure_odd_kernel_size(size)
                    
                    # 使用双线性插值放大核
                    resized = cv2.resize(kernel, (size, size), interpolation=cv2.INTER_LINEAR)
                    scale_kernels[direction] = resized
            
            self.multi_kernels[scale] = scale_kernels
    
    def enhanced_preprocessing(self, image_tensor):
        """增强的图像预处理 - 简化版"""
        print("=== 图像预处理 ===")
        
        # 提取绿色通道
        if len(image_tensor.shape) == 3:
            green_channel = image_tensor[1]
        else:
            green_channel = image_tensor
        
        green_np = green_channel.cpu().numpy() if isinstance(green_channel, torch.Tensor) else green_channel
        
        # 确保数据类型
        green_np = green_np.astype(np.float32)
        
        # 创建圆形掩码
        h, w = green_np.shape
        center_y, center_x = h // 2, w // 2
        radius = min(center_y, center_x) * 0.9
        Y, X = np.ogrid[:h, :w]
        mask = ((X - center_x) ** 2 + (Y - center_y) ** 2) <= radius ** 2
        
        # 1. 基础归一化
        normalized = green_np * mask.astype(np.float32)
        if normalized.max() > 0:
            normalized = normalized / normalized.max()
        
        # 2. CLAHE增强
        clahe = cv2.createCLAHE(
            clipLimit=self.params['clahe_clip_limit'],
            tileGridSize=(self.params['clahe_tile_size'], self.params['clahe_tile_size'])
        )
        clahe_enhanced = clahe.apply((normalized * 255).astype(np.uint8)).astype(np.float32) / 255.0
        
        # 3. 背景估计和减除
        kernel_size = self.ensure_odd_kernel_size(self.params['background_kernel_size'])
        background = cv2.GaussianBlur(clahe_enhanced, (kernel_size, kernel_size), 0)
        
        # 4. 背景减除
        enhanced = clahe_enhanced - background
        enhanced = np.maximum(enhanced, 0)
        
        # 5. 对比度拉伸
        if enhanced.max() > enhanced.min():
            enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())
        
        # 确保输出为float32
        enhanced = enhanced.astype(np.float32)
        
        print(f"预处理完成，范围: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
        return enhanced, mask
    
    def frangi_vessel_enhancement(self, image):
        """Frangi血管增强滤波器 - 使用scipy替代"""
        print("应用Frangi血管增强...")
        
        from scipy.ndimage import gaussian_filter
        
        responses = []
        for sigma in self.params['hessian_sigma']:
            # 使用scipy计算Hessian矩阵的各个分量
            # 二阶偏导数
            Ixx = gaussian_filter(image, sigma, order=[0, 2])
            Iyy = gaussian_filter(image, sigma, order=[2, 0])
            Ixy = gaussian_filter(image, sigma, order=[1, 1])
            
            # 计算特征值
            det = Ixx * Iyy - Ixy ** 2
            trace = Ixx + Iyy
            
            # 计算特征值
            discriminant = np.sqrt(np.maximum(trace**2 - 4*det, 0))
            lambda1 = (trace + discriminant) / 2  # 较大特征值
            lambda2 = (trace - discriminant) / 2  # 较小特征值
            
            # Frangi滤波器参数
            alpha = 0.5  # 控制对blob的敏感度
            beta = 0.5   # 控制对线状结构的敏感度
            c = 15       # 控制对噪声的敏感度
            
            # 计算各项
            Ra = np.abs(lambda2) / (np.abs(lambda1) + 1e-6)  # 避免除零
            Rb = np.abs(lambda1 * lambda2)
            S = np.sqrt(lambda1**2 + lambda2**2)
            
            # Frangi响应
            response = np.zeros_like(image)
            
            # 只在lambda2 < 0的地方有血管响应
            mask = lambda2 < 0
            response[mask] = (1 - np.exp(-Ra[mask]**2 / (2*alpha**2))) * \
                           np.exp(-Rb[mask] / (2*beta**2)) * \
                           (1 - np.exp(-S[mask]**2 / (2*c**2)))
            
            responses.append(response)
        
        # 取最大响应
        frangi_response = np.maximum.reduce(responses)
        
        # 归一化
        if frangi_response.max() > 0:
            frangi_response = frangi_response / frangi_response.max()
        
        print(f"Frangi增强完成，响应范围: [{frangi_response.min():.3f}, {frangi_response.max():.3f}]")
        return frangi_response
    
    def multi_scale_line_detection(self, enhanced_image):
        """多尺度方向线检测 - 改进版"""
        print("=== 多尺度方向线检测 ===")
        
        # 先应用Frangi增强
        frangi_enhanced = self.frangi_vessel_enhancement(enhanced_image)
        
        # 确保数据类型正确
        frangi_enhanced = frangi_enhanced.astype(np.float32)
        
        all_responses = []
        
        for scale in self.params['scales']:
            scale_responses = []
            kernels = self.multi_kernels[scale]
            
            for direction, kernel in kernels.items():
                # 确保kernel也是float32类型
                kernel = kernel.astype(np.float32)
                
                # 对Frangi增强后的图像进行线检测
                response = cv2.filter2D(frangi_enhanced, cv2.CV_32F, kernel)
                response = np.maximum(0, response)  # ReLU
                scale_responses.append(response)
            
            # 每个尺度的最大响应
            scale_max_response = np.maximum.reduce(scale_responses)
            all_responses.append(scale_max_response)
        
        # 跨尺度最大响应
        combined_response = np.maximum.reduce(all_responses)
        
        # 非极大值抑制
        if self.params['non_max_suppression']:
            combined_response = self.non_maximum_suppression(combined_response)
        
        print(f"多尺度线检测完成，范围: [{combined_response.min():.3f}, {combined_response.max():.3f}]")
        return combined_response
    
    def non_maximum_suppression(self, response, window_size=3):
        """非极大值抑制"""
        local_maxima = maximum_filter(response, size=window_size)
        suppressed = np.where(response == local_maxima, response, 0)
        return suppressed
    
    def adaptive_seed_extraction(self, combined_response, mask):
        """改进的种子点提取"""
        print("自适应种子点提取...")
        
        # 更激进的自适应阈值
        valid_response = combined_response[mask > 0]
        if len(valid_response) > 0:
            # 使用更高的百分位数来选择更强的响应
            threshold = np.percentile(valid_response, 95)  # 前5%
            print(f"自适应阈值: {threshold:.3f}")
        else:
            threshold = self.params['line_response_threshold']
        
        # 种子点检测
        radius = self.params['local_max_radius']
        seed_points = []
        h, w = combined_response.shape
        
        # 创建候选点
        candidates = np.where((combined_response > threshold) & mask)
        
        for i, j in zip(candidates[0], candidates[1]):
            if (radius <= i < h - radius and radius <= j < w - radius):
                # 局部最大值检查
                local_region = combined_response[i-radius:i+radius+1, j-radius:j+radius+1]
                if combined_response[i, j] == local_region.max():
                    seed_points.append((i, j))
        
        print(f"找到种子点数量: {len(seed_points)}")
        
        # 转换为掩码
        seeds_mask = np.zeros_like(combined_response, dtype=np.uint8)
        for i, j in seed_points:
            seeds_mask[i, j] = 1
        
        return seeds_mask, seed_points, threshold
    
    def improved_region_growing(self, enhanced_image, seed_points, mask, threshold):
        """简化但更有效的区域生长"""
        print("=== 区域生长 ===")
        
        h, w = enhanced_image.shape
        segmented = np.zeros((h, w), dtype=np.uint8)
        visited = np.zeros((h, w), dtype=bool)
        
        # 计算梯度幅值
        grad_x = cv2.Sobel(enhanced_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(enhanced_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # 自适应阈值 - 更宽松的生长条件
        valid_gradient = gradient[mask > 0]
        if len(valid_gradient) > 0:
            grad_mean = np.mean(valid_gradient)
            grad_std = np.std(valid_gradient)
            vessel_threshold = grad_mean + 0.5 * grad_std  # 更宽松
            intensity_threshold = threshold * 0.3  # 更宽松
        else:
            vessel_threshold = 0.1
            intensity_threshold = threshold * 0.5
        
        print(f"血管阈值: {vessel_threshold:.3f}, 强度阈值: {intensity_threshold:.3f}")
        
        # 8邻域
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        total_vessel_pixels = 0
        
        for seed_idx, (seed_x, seed_y) in enumerate(seed_points):
            if visited[seed_x, seed_y]:
                continue
            
            queue = deque([(seed_x, seed_y)])
            region_pixels = 0
            
            while queue:
                x, y = queue.popleft()
                
                if visited[x, y]:
                    continue
                
                visited[x, y] = True
                
                # 更宽松的判断条件
                grad_val = gradient[x, y]
                intensity_val = enhanced_image[x, y]
                
                # 血管判断 - 要么梯度高，要么强度高
                is_vessel = (grad_val >= vessel_threshold or 
                           intensity_val >= intensity_threshold)
                
                if is_vessel:
                    segmented[x, y] = 255
                    region_pixels += 1
                    
                    # 添加邻域
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < h and 0 <= ny < w and 
                            not visited[nx, ny] and mask[nx, ny]):
                            queue.append((nx, ny))
            
            total_vessel_pixels += region_pixels
        
        print(f"区域生长完成，总血管像素: {total_vessel_pixels}")
        return segmented.astype(np.float32) / 255.0
    
    def enhanced_post_processing(self, vessels, mask):
        """简化的后处理"""
        print("=== 后处理 ===")
        
        vessels = vessels * mask
        vessels_uint8 = (vessels * 255).astype(np.uint8)
        
        # 1. 形态学清理
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(vessels_uint8, cv2.MORPH_OPEN, open_kernel)
        
        # 2. 连接断裂血管
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
        
        # 3. 简单连通域过滤
        labeled = measure.label(connected)
        props = measure.regionprops(labeled)
        
        filtered = np.zeros_like(connected)
        kept_regions = 0
        
        for prop in props:
            # 只过滤太小的区域
            if prop.area >= self.params['min_area']:
                filtered[labeled == prop.label] = 255
                kept_regions += 1
        
        print(f"连通域过滤: 保留 {kept_regions}/{len(props)} 个区域")
        
        return filtered.astype(np.float32) / 255.0
    
    def segment_vessels(self, image_tensor):
        """完整的血管分割流程 - 简化版"""
        print("开始血管分割流程...")
        
        # 1. 预处理
        enhanced_image, mask = self.enhanced_preprocessing(image_tensor)
        
        # 2. 多尺度线检测（内含Frangi增强）
        combined_response = self.multi_scale_line_detection(enhanced_image)
        
        # 3. 种子点提取
        seeds_mask, seed_points, threshold = self.adaptive_seed_extraction(
            combined_response, mask)
        
        if len(seed_points) == 0:
            print("警告: 未找到种子点!")
            return np.zeros_like(enhanced_image), combined_response, seeds_mask
        
        # 4. 区域生长
        vessels = self.improved_region_growing(
            enhanced_image, seed_points, mask, threshold)
        
        # 5. 后处理
        final_vessels = self.enhanced_post_processing(vessels, mask)
        
        return final_vessels, combined_response, seeds_mask

def compute_vessel_metrics(vessels, seeds):
    """计算血管质量指标"""
    if seeds.sum() == 0:
        return 0.0
    
    expansion_ratio = vessels.sum() / seeds.sum()
    labeled = measure.label(vessels > 0.5)
    num_components = labeled.max()
    connectivity_score = 1.0 / (num_components + 1)
    vessel_density = vessels.sum() / (vessels.shape[0] * vessels.shape[1])
    
    score = expansion_ratio * 0.5 + connectivity_score * 0.3 + vessel_density * 1000 * 0.2
    return score

def simple_test(data_path, output_dir, num_images=3):
    """简单测试函数"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用默认参数
    segmenter = DirectionalLineVesselSegmentation()
    
    # 获取测试图像
    image_files = [f for f in os.listdir(data_path) 
                  if f.lower().endswith(('.tif', '.png', '.jpg'))][:num_images]
    
    print(f"测试 {len(image_files)} 张图像...")
    
    for i, img_file in enumerate(image_files):
        print(f"\n{'='*50}")
        print(f"处理 {i+1}/{len(image_files)}: {img_file}")
        print(f"{'='*50}")
        
        try:
            # 加载图像
            img_path = os.path.join(data_path, img_file)
            image = Image.open(img_path).convert('RGB')
            img_array = np.array(image) / 255.0
            img_tensor = torch.tensor(img_array.transpose(2, 0, 1), dtype=torch.float32)
            
            # 分割
            vessels, line_response, seeds = segmenter.segment_vessels(img_tensor)
            
            # 计算指标
            vessel_pixels = vessels.sum()
            seed_count = seeds.sum()
            vessel_percentage = vessel_pixels / (vessels.shape[0] * vessels.shape[1]) * 100
            score = compute_vessel_metrics(vessels, seeds)
            
            # 保存结果
            base_name = os.path.splitext(img_file)[0]
            
            # 保存血管分割结果
            vessel_img = (vessels * 255).astype(np.uint8)
            Image.fromarray(vessel_img).save(f"{output_dir}/{base_name}_vessels.png")
            
            # 保存线响应
            if line_response.max() > 0:
                response_normalized = (line_response / line_response.max() * 255).astype(np.uint8)
            else:
                response_normalized = (line_response * 255).astype(np.uint8)
            Image.fromarray(response_normalized).save(f"{output_dir}/{base_name}_response.png")
            
            # 保存种子点
            seed_img = (seeds * 255).astype(np.uint8)
            Image.fromarray(seed_img).save(f"{output_dir}/{base_name}_seeds.png")
            
            print(f"结果:")
            print(f"  血管像素: {vessel_pixels:.0f} ({vessel_percentage:.2f}%)")
            print(f"  种子点: {seed_count:.0f}")
            print(f"  质量得分: {score:.3f}")
            
        except Exception as e:
            print(f"处理错误: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n结果保存到: {output_dir}/")

if __name__ == "__main__":
    # 简单测试
    data_path = '/root/FSG-Net-pytorch/data/DRIVE/val/input'
    output_dir = './simple_vessel_results'
    
    print("=== 简化版血管分割测试 ===")
    simple_test(data_path, output_dir, num_images=3)