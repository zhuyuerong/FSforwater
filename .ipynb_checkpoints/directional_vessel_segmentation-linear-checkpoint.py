import os
import numpy as np
import cv2
from PIL import Image
import torch
from skimage import morphology, measure
import itertools
import json
from tqdm import tqdm
from collections import deque

class DirectionalLineVesselSegmentation:
    """基于垂直梯度扩散的血管分割"""
    
    def __init__(self, params=None):
        # 默认参数
        self.default_params = {
            # 图像增强参数
            'background_kernel_size': 100,  # 均值滤波核大小
            
            # 线检测参数
            'line_response_threshold': 0.1,  # 线响应阈值
            'local_max_radius': 3,           # 局部最大值半径
            
            # 区域生长参数
            'mu_f': 0.0789,
            'sigma_f': 0.0774,  # 确保包含此参数
            'alpha': 1.0,
            
            # 形态学参数
            'morph_open_size': 2,
            'morph_close_size': 3,
            
            # 连通域过滤
            'min_area': 50,
        }
        
        self.params = params if params else self.default_params.copy()
        
        # 定义四个方向检测核
        self.kernels = {
            '45°': np.array([[-1, -1, 2],
                            [-1, 2, -1],
                            [2, -1, -1]], dtype=np.float32),
            
            '90°': np.array([[-1, 2, -1],
                            [-1, 2, -1],
                            [-1, 2, -1]], dtype=np.float32),
            
            '135°': np.array([[2, -1, -1],
                             [-1, 2, -1],
                             [-1, -1, 2]], dtype=np.float32),
            
            '180°': np.array([[-1, -1, -1],
                             [2, 2, 2],
                             [-1, -1, -1]], dtype=np.float32)
        }
    
    def image_enhancement_preprocessing(self, image_tensor):
        """4.1 图像增强预处理"""
        # 提取绿色通道
        if len(image_tensor.shape) == 3:
            green_channel = image_tensor[1]  # 绿色通道
        else:
            green_channel = image_tensor
        
        green_np = green_channel.cpu().numpy() if isinstance(green_channel, torch.Tensor) else green_channel
        
        # 创建圆形掩码
        h, w = green_np.shape
        center_y, center_x = h // 2, w // 2
        radius = min(center_y, center_x) * 0.9
        Y, X = np.ogrid[:h, :w]
        mask = ((X - center_x) ** 2 + (Y - center_y) ** 2) <= radius ** 2
        
        # 归一化处理
        normalized_image = green_np * mask.astype(np.float32)
        
        # 均值滤波(核大小100×100) → Background_Image
        kernel_size = self.params['background_kernel_size']
        background_image = cv2.blur(normalized_image, (kernel_size, kernel_size))
        
        # 增强图像 = Normalized_Image - Background_Image
        enhanced_image = normalized_image - background_image
        
        # 归一化增强图像
        if enhanced_image.max() > enhanced_image.min():
            enhanced_image = (enhanced_image - enhanced_image.min()) / (enhanced_image.max() - enhanced_image.min())
        
        return enhanced_image, mask
    
    def vessel_centerline_detection(self, enhanced_image):
        """4.2 血管中心线检测(种子点生成)"""
        responses = []
        
        # FOR 每个方向核 Kernel_i
        for direction, kernel in self.kernels.items():
            # Response_i = Enhanced_Image ⊗ Kernel_i (卷积操作)
            response = cv2.filter2D(enhanced_image, cv2.CV_32F, kernel)
            
            # 应用ReLU激活: Response_i = max(0, Response_i)
            response = np.maximum(0, response)
            responses.append(response)
        
        # 合并响应：Combined_Response = max(Response_1, Response_2, Response_3, Response_4)
        combined_response = np.maximum.reduce(responses)
        
        return combined_response
    
    def extract_seed_points(self, combined_response, enhanced_image, mask):
        """提取种子点"""
        threshold = self.params['line_response_threshold']
        radius = self.params['local_max_radius']
        
        # 动态调整阈值
        valid_response = combined_response[mask > 0]
        if len(valid_response) > 0:
            threshold = np.percentile(valid_response, 90)  # 取前10%
        
        seed_points = []
        h, w = combined_response.shape
        
        # FOR Enhanced_Image中每个像素(i,j)
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                if not mask[i, j]:  # 跳过掩码外的像素
                    continue
                
                # IF Combined_Response(i,j) > 阈值 AND 为局部最大值
                if combined_response[i, j] > threshold:
                    # 检查是否为局部最大值
                    local_region = combined_response[i-radius:i+radius+1, j-radius:j+radius+1]
                    if combined_response[i, j] == local_region.max():
                        seed_points.append((i, j))
        
        # 转换为掩码格式
        seeds_mask = np.zeros_like(combined_response, dtype=np.uint8)
        for i, j in seed_points:
            seeds_mask[i, j] = 1
        
        return seeds_mask, seed_points
    
    def compute_gradient_diffusion_matrix(self, gradient_x, gradient_y, x, y):
        """计算基于垂直梯度的扩散矩阵"""
        
        # 获取当前像素的梯度
        gx = gradient_x[x, y]
        gy = gradient_y[x, y]
        
        # 计算梯度幅值和方向
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        if gradient_magnitude < 1e-6:
            # 梯度很小时，使用更大的标准邻域
            return [(-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2),
                    (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2),
                    (0,-2), (0,-1), (0,1), (0,2),
                    (1,-2), (1,-1), (1,0), (1,1), (1,2),
                    (2,-2), (2,-1), (2,0), (2,1), (2,2)]
        
        # 计算梯度方向（弧度）
        gradient_angle = np.arctan2(gy, gx)
        
        # 计算垂直梯度方向（血管走向）
        vessel_angle = gradient_angle + np.pi/2  # 垂直于梯度方向
        
        # 基于垂直梯度方向生成扩散邻域
        diffusion_neighbors = []
        
        # 主要扩散方向：沿血管走向，更大范围
        for distance in range(1, 8):  # 扩散距离1-7，增加范围
            # 血管方向的两个方向
            dx1 = int(round(distance * np.cos(vessel_angle)))
            dy1 = int(round(distance * np.sin(vessel_angle)))
            dx2 = int(round(distance * np.cos(vessel_angle + np.pi)))
            dy2 = int(round(distance * np.sin(vessel_angle + np.pi)))
            
            diffusion_neighbors.extend([(dx1, dy1), (dx2, dy2)])
        
        # 辅助扩散方向：垂直于血管走向，增加范围
        perpendicular_angle = vessel_angle + np.pi/2
        for distance in range(1, 4):  # 垂直方向扩散距离增加到1-3
            dx1 = int(round(distance * np.cos(perpendicular_angle)))
            dy1 = int(round(distance * np.sin(perpendicular_angle)))
            dx2 = int(round(distance * np.cos(perpendicular_angle + np.pi)))
            dy2 = int(round(distance * np.sin(perpendicular_angle + np.pi)))
            
            diffusion_neighbors.extend([(dx1, dy1), (dx2, dy2)])
        
        # 添加对角线和更多邻域确保覆盖面更大
        extra_neighbors = [
            (-3, -3), (-3, 0), (-3, 3), (0, -3), (0, 3), (3, -3), (3, 0), (3, 3),
            (-2, -3), (-2, 3), (2, -3), (2, 3), (-3, -2), (-3, 2), (3, -2), (3, 2),
            (-1, -3), (-1, 3), (1, -3), (1, 3), (-3, -1), (-3, 1), (3, -1), (3, 1)
        ]
        diffusion_neighbors.extend(extra_neighbors)
        
        # 去除重复的邻域点
        unique_neighbors = list(set(diffusion_neighbors))
        
        # 如果邻域数量还是太少，确保至少有24个邻域
        if len(unique_neighbors) < 24:
            standard_large = [(-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2),
                             (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2),
                             (0,-2), (0,-1), (0,1), (0,2),
                             (1,-2), (1,-1), (1,0), (1,1), (1,2),
                             (2,-2), (2,-1), (2,0), (2,1), (2,2)]
            for neighbor in standard_large:
                if neighbor not in unique_neighbors:
                    unique_neighbors.append(neighbor)
        
        return unique_neighbors
    
    def region_iterative_growing(self, enhanced_image, seed_points, mask):
        """4.3 基于垂直梯度扩散的区域迭代生长算法"""
        h, w = enhanced_image.shape
        
        # 初始化
        segmented = np.zeros((h, w), dtype=np.uint8)  # 分割结果图
        visited = np.zeros((h, w), dtype=bool)        # 访问标记图
        
        # 计算梯度图 Gradient = √(∇x² + ∇y²)
        grad_x = cv2.Sobel(enhanced_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(enhanced_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # 设置参数 - 确保所有参数都存在
        mu_f = self.params.get('mu_f', 0.0789)
        sigma_f = self.params.get('sigma_f', 0.0774)
        alpha = self.params.get('alpha', 1.0)
        
        vessel_threshold = mu_f + alpha * sigma_f
        background_threshold = mu_f
        
        total_vessel_pixels = 0
        
        # FOR 每个种子点
        for seed_idx, (seed_x, seed_y) in enumerate(seed_points):
            if visited[seed_x, seed_y]:
                continue  # 跳过已访问的种子点
            
            # 初始化生长队列
            queue = deque([(seed_x, seed_y)])
            region_pixels = 0
            
            # WHILE Queue不为空
            while queue:
                # 取出当前像素
                x, y = queue.popleft()
                
                if visited[x, y]:
                    continue
                
                # 标记为已访问
                visited[x, y] = True
                
                # 获取梯度值
                rho = gradient[x, y]
                
                # 应用血管判断准则
                if vessel_threshold <= rho:
                    # 血管条件
                    segmented[x, y] = 255
                    region_pixels += 1
                    
                    # 计算基于垂直梯度的扩散邻域
                    diffusion_neighbors = self.compute_gradient_diffusion_matrix(
                        grad_x, grad_y, x, y
                    )
                    
                    # 将扩散邻域像素加入生长队列
                    for dx, dy in diffusion_neighbors:
                        nx, ny = x + dx, y + dy
                        
                        # IF (nx, ny)在图像范围内 AND Visited(nx, ny) == False
                        if (0 <= nx < h and 0 <= ny < w and 
                            not visited[nx, ny] and mask[nx, ny]):
                            queue.append((nx, ny))
                
                elif background_threshold >= rho:
                    # 背景条件
                    segmented[x, y] = 0
                
                else:
                    # 边界模糊区域，继续使用垂直梯度扩散
                    diffusion_neighbors = self.compute_gradient_diffusion_matrix(
                        grad_x, grad_y, x, y
                    )
                    
                    for dx, dy in diffusion_neighbors:
                        nx, ny = x + dx, y + dy
                        
                        if (0 <= nx < h and 0 <= ny < w and 
                            not visited[nx, ny] and mask[nx, ny]):
                            queue.append((nx, ny))
            
            total_vessel_pixels += region_pixels
        
        return segmented.astype(np.float32) / 255.0
    
    def post_process(self, vessels, mask):
        """后处理"""
        # 应用掩码
        vessels = vessels * mask
        
        # 形态学操作
        vessels_uint8 = (vessels * 255).astype(np.uint8)
        
        # 开运算去噪 - 使用安全的参数获取
        morph_open_size = self.params.get('morph_open_size', 2)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (morph_open_size, morph_open_size))
        cleaned = cv2.morphologyEx(vessels_uint8, cv2.MORPH_OPEN, open_kernel)
        
        # 闭运算连接 - 使用安全的参数获取
        morph_close_size = self.params.get('morph_close_size', 3)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                (morph_close_size, morph_close_size))
        connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
        
        # 连通域过滤
        labeled = measure.label(connected)
        props = measure.regionprops(labeled)
        
        filtered = np.zeros_like(connected)
        kept_regions = 0
        
        min_area = self.params.get('min_area', 50)
        for prop in props:
            if prop.area >= min_area:
                filtered[labeled == prop.label] = 255
                kept_regions += 1
  
        return filtered.astype(np.float32) / 255.0
        min_area = self.params.get('min_area', 50)
        for prop in props:
            if prop.area >= min_area:
                filtered[labeled == prop.label] = 255
                kept_regions += 1

        print(f"连通域过滤: 保留 {kept_regions}/{len(props)} 个区域")

        return filtered.astype(np.float32) / 255.0

    def segment_vessels(self, image_tensor):
        """完整的血管分割流程"""
        # 4.1 图像增强预处理
        enhanced_image, mask = self.image_enhancement_preprocessing(image_tensor)
        
        # 4.2 血管中心线检测
        combined_response = self.vessel_centerline_detection(enhanced_image)
        
        # 提取种子点
        seeds_mask, seed_points = self.extract_seed_points(combined_response, enhanced_image, mask)
        
        if len(seed_points) == 0:
            return np.zeros_like(enhanced_image), combined_response, seeds_mask
        
        # 4.3 区域迭代生长
        vessels = self.region_iterative_growing(enhanced_image, seed_points, mask)
        
        # 后处理
        final_vessels = self.post_process(vessels, mask)
        
        return final_vessels, combined_response, seeds_mask

class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def define_parameter_grid(self):
        """定义参数网格"""
        param_grid = {
            'line_response_threshold': [0.05, 0.1, 0.15],
            'local_max_radius': [2, 3, 4],
            'mu_f': [0.06, 0.0789, 0.09],
            'sigma_f': [0.0774],  # 添加sigma_f参数
            'alpha': [0.8, 1.0, 1.2],
            'background_kernel_size': [80, 100, 120]
        }
        return param_grid
    
    def grid_search(self, max_combinations=20):
        """网格搜索最佳参数"""
        param_grid = self.define_parameter_grid()
        
        # 生成所有参数组合
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        # 限制组合数量
        if len(combinations) > max_combinations:
            combinations = combinations[:max_combinations]
        
        # 获取测试图像
        image_files = [f for f in os.listdir(self.data_path) 
                      if f.lower().endswith(('.tif', '.png', '.jpg'))]
        test_files = image_files[:2]  # 使用前2张图进行参数搜索
        
        results = []
        
        for i, combination in enumerate(tqdm(combinations)):
            params = dict(zip(keys, combination))
            
            # 更新默认参数
            segmenter = DirectionalLineVesselSegmentation()
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
                    
                    vessels, response, seeds = segmenter.segment_vessels(img_tensor)
                    total_vessels += vessels.sum()
                    total_seeds += seeds.sum()
                    
                except Exception as e:
                    total_vessels = 0
                    break
            
            # 评估指标
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
        with open(os.path.join(self.output_dir, 'gradient_diffusion_grid_search_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results[0]['params'] if results else None

def process_validation_set(data_path, output_dir, best_params=None):
    """处理验证集"""
    os.makedirs(f"{output_dir}/gradient_diffusion_results", exist_ok=True)
    os.makedirs(f"{output_dir}/line_responses", exist_ok=True)
    os.makedirs(f"{output_dir}/seed_points", exist_ok=True)
    
    # 创建分割器 - 确保包含所有必要参数
    if best_params:
        # 确保所有必要参数都存在
        default_missing_params = {
            'sigma_f': 0.0774,
            'morph_open_size': 2,
            'morph_close_size': 3,
            'min_area': 50
        }
        
        for param, default_value in default_missing_params.items():
            if param not in best_params:
                best_params[param] = default_value
    
    segmenter = DirectionalLineVesselSegmentation(best_params)
    
    # 获取所有图像
    image_files = [f for f in os.listdir(data_path) 
                  if f.lower().endswith(('.tif', '.png', '.jpg'))]
    
    for i, img_file in enumerate(image_files):
        try:
            # 加载图像
            img_path = os.path.join(data_path, img_file)
            image = Image.open(img_path).convert('RGB')
            img_array = np.array(image) / 255.0
            img_tensor = torch.tensor(img_array.transpose(2, 0, 1), dtype=torch.float32)
            
            # 分割
            vessels, line_response, seeds = segmenter.segment_vessels(img_tensor)
            
            # 保存结果
            base_name = os.path.splitext(img_file)[0]
            
            # 保存血管分割结果
            vessel_img = (vessels * 255).astype(np.uint8)
            Image.fromarray(vessel_img).save(f"{output_dir}/gradient_diffusion_results/{base_name}_vessels.png")
            
            # 保存线响应
            if line_response.max() > 0:
                response_normalized = (line_response / line_response.max() * 255).astype(np.uint8)
            else:
                response_normalized = (line_response * 255).astype(np.uint8)
            Image.fromarray(response_normalized).save(f"{output_dir}/line_responses/{base_name}_response.png")
            
            # 保存种子点
            seed_img = (seeds * 255).astype(np.uint8)
            Image.fromarray(seed_img).save(f"{output_dir}/seed_points/{base_name}_seeds.png")
            
        except Exception as e:
            continue

def main():
    # 配置路径
    data_path = '/root/FSG-Net-pytorch/data/DRIVE/val/input'
    output_dir = './gradient_diffusion_vessel_results'
    
    # 1. 参数优化
    optimizer = ParameterOptimizer(data_path, output_dir)
    best_params = optimizer.grid_search(max_combinations=20)
    
    if best_params:
        # 2. 处理验证集
        process_validation_set(data_path, output_dir, best_params)

if __name__ == "__main__":
    main()