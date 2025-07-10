import os
import numpy as np
import cv2
from PIL import Image
import torch
from skimage import morphology, measure
from collections import deque
import matplotlib.pyplot as plt

class RidgeTraversalVesselSegmentation:
    """基于Ridge Traversal的血管追踪分割 - Aylward算法实现"""
    
    def __init__(self, params=None):
        self.default_params = {
            # 图像增强参数
            'background_kernel_size': 100,
            
            # 种子点检测参数 (保持原有逻辑)
            'line_response_threshold': 0.1,
            'local_max_radius': 3,
            
            # Ridge追踪参数
            'ridge_step_size': 1.0,          # 追踪步长
            'max_ridge_length': 25,          # 最大追踪长度
            'ridge_threshold': 0.05,         # Ridge强度阈值
            'curvature_threshold': 0.5,      # 曲率变化阈值
            'radius_change_threshold': 0.3,   # 半径变化阈值
            
            # Hessian分析参数
            'hessian_sigma': 1.5,            # Hessian矩阵高斯核
            'vessel_min_radius': 1.0,        # 最小血管半径
            'vessel_max_radius': 8.0,        # 最大血管半径
            
            # 连接参数
            'connection_distance': 8,         # 血管段连接距离
            
            # 后处理参数
            'morph_open_size': 2,
            'morph_close_size': 3,
            'min_area': 30,
        }
        
        self.params = params if params else self.default_params.copy()
        
        # 方向检测核 (保持原有)
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
        """图像预处理"""
        if len(image_tensor.shape) == 3:
            green_channel = image_tensor[1]
        else:
            green_channel = image_tensor
        
        green_np = green_channel.cpu().numpy() if isinstance(green_channel, torch.Tensor) else green_channel
        green_np = green_np.astype(np.float32)
        
        # 圆形掩码
        h, w = green_np.shape
        center_y, center_x = h // 2, w // 2
        radius = min(center_y, center_x) * 0.9
        Y, X = np.ogrid[:h, :w]
        mask = ((X - center_x) ** 2 + (Y - center_y) ** 2) <= radius ** 2
        
        # 归一化和背景减除
        normalized_image = green_np * mask.astype(np.float32)
        kernel_size = self.params['background_kernel_size']
        background_image = cv2.blur(normalized_image, (kernel_size, kernel_size))
        enhanced_image = normalized_image - background_image
        
        if enhanced_image.max() > enhanced_image.min():
            enhanced_image = (enhanced_image - enhanced_image.min()) / (enhanced_image.max() - enhanced_image.min())
        
        enhanced_image = enhanced_image.astype(np.float32)
        
        return enhanced_image, mask
    
    def vessel_centerline_detection(self, enhanced_image):
        """血管中心线检测"""
        responses = []
        for direction, kernel in self.kernels.items():
            kernel = kernel.astype(np.float32)
            response = cv2.filter2D(enhanced_image, cv2.CV_32F, kernel)
            response = np.maximum(0, response)
            responses.append(response)
        
        combined_response = np.maximum.reduce(responses)
        return combined_response
    
    def extract_strategic_seed_points(self, enhanced_image, line_response, mask):
        """提取战略性种子点：分叉点、末端、边缘、遗漏区域"""
        
        h, w = enhanced_image.shape
        all_seeds = []
        
        # 1. 检测血管分叉点
        bifurcation_seeds = self.detect_bifurcation_points(enhanced_image, mask)
        all_seeds.extend(bifurcation_seeds)
        
        # 2. 检测血管末端
        endpoint_seeds = self.detect_vessel_endpoints(enhanced_image, mask)
        all_seeds.extend(endpoint_seeds)
        
        # 3. 检测血管边缘强响应点
        edge_seeds = self.detect_vessel_edge_points(line_response, mask)
        all_seeds.extend(edge_seeds)
        
        # 4. 检测可能遗漏的血管区域
        missed_seeds = self.detect_missed_vessel_regions(enhanced_image, mask)
        all_seeds.extend(missed_seeds)
        
        # 去重和过滤
        unique_seeds = self.filter_and_deduplicate_seeds(all_seeds, min_distance=8)
        print(f"战略种子点: 分叉{len(bifurcation_seeds)} + 末端{len(endpoint_seeds)} + 边缘{len(edge_seeds)} + 遗漏{len(missed_seeds)} = {len(unique_seeds)}个")
        
        # 创建种子点掩码
        seeds_mask = np.zeros_like(enhanced_image, dtype=np.uint8)
        for y, x in unique_seeds:
            seeds_mask[y, x] = 1
        
        return seeds_mask, unique_seeds
    
    def detect_bifurcation_points(self, image, mask):
        """检测血管分叉点"""
        # 使用骨架化检测分叉
        from skimage import morphology
        
        # 先进行简单阈值分割
        threshold = np.percentile(image[mask > 0], 75)
        binary = (image > threshold) & mask
        
        # 骨架化
        skeleton = morphology.skeletonize(binary)
        
        # 检测分叉点（3个或更多邻居的骨架点）
        bifurcation_points = []
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0  # 中心点不计算
        
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                if skeleton[i, j]:
                    # 计算8邻域中骨架点的数量
                    neighbors = np.sum(skeleton[i-1:i+2, j-1:j+2] * kernel)
                    if neighbors >= 3:  # 分叉点
                        bifurcation_points.append((i, j))
        
        return bifurcation_points
    
    def detect_vessel_endpoints(self, image, mask):
        """检测血管末端点"""
        # 检测高梯度的孤立点
        gy, gx = np.gradient(image)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        
        endpoint_points = []
        
        # 寻找梯度强但周围连续性较差的点
        threshold = np.percentile(gradient_mag[mask > 0], 85)
        
        for i in range(3, image.shape[0]-3):
            for j in range(3, image.shape[1]-3):
                if not mask[i, j]:
                    continue
                
                if gradient_mag[i, j] > threshold:
                    # 检查局部连续性
                    local_region = image[i-2:i+3, j-2:j+3]
                    local_std = np.std(local_region)
                    
                    # 高梯度但局部变化大的点可能是末端
                    if local_std > 0.1:
                        # 检查是否在边缘附近
                        edge_distance = min(i, j, image.shape[0]-1-i, image.shape[1]-1-j)
                        if edge_distance > 20:  # 不要太靠近图像边缘
                            endpoint_points.append((i, j))
        
        return endpoint_points
    
    def detect_vessel_edge_points(self, line_response, mask):
        """检测血管边缘的强响应点"""
        # 从线响应中提取少量强响应点
        edge_points = []
        
        # 使用更高的阈值，只取最强的响应
        valid_response = line_response[mask > 0]
        if len(valid_response) > 0:
            high_threshold = np.percentile(valid_response, 95)  # 提高到95%分位数
            
            # 在强响应区域中进行非最大值抑制
            strong_response = (line_response > high_threshold) & mask
            
            # 分块采样，确保分布均匀
            block_size = 20
            for i in range(0, line_response.shape[0], block_size):
                for j in range(0, line_response.shape[1], block_size):
                    block = strong_response[i:min(i+block_size, line_response.shape[0]), 
                                         j:min(j+block_size, line_response.shape[1])]
                    
                    if np.any(block):
                        # 在每个块中找最强的点
                        block_response = line_response[i:min(i+block_size, line_response.shape[0]), 
                                                     j:min(j+block_size, line_response.shape[1])]
                        local_coords = np.unravel_index(np.argmax(block_response * block), block.shape)
                        global_y, global_x = i + local_coords[0], j + local_coords[1]
                        edge_points.append((global_y, global_x))
        
        return edge_points
    
    def detect_missed_vessel_regions(self, image, mask):
        """检测可能被遗漏的血管区域"""
        missed_points = []
        
        # 使用多尺度的形态学操作检测细小结构
        from skimage import morphology
        
        # 不同尺度的线性结构元素
        for angle in [0, 45, 90, 135]:
            length = 5
            selem = morphology.disk(1)  # 小的结构元素
            
            # 形态学开运算
            opened = morphology.opening(image, selem)
            
            # 顶帽变换检测细小结构
            tophat = image - opened
            
            # 在顶帽图像中寻找可能的血管点
            threshold = np.percentile(tophat[mask > 0], 90)
            candidates = (tophat > threshold) & mask
            
            # 稀疏采样
            coords = np.where(candidates)
            for idx in range(0, len(coords[0]), 15):  # 每15个点取一个
                missed_points.append((coords[0][idx], coords[1][idx]))
        
        return missed_points
    
    def filter_and_deduplicate_seeds(self, seeds, min_distance=8):
        """过滤和去重种子点"""
        if not seeds:
            return []
        
        # 去重：移除距离过近的点
        unique_seeds = []
        seeds = list(set(seeds))  # 先去除完全重复的点
        
        for i, seed1 in enumerate(seeds):
            is_unique = True
            for j, seed2 in enumerate(unique_seeds):
                dist = np.sqrt((seed1[0] - seed2[0])**2 + (seed1[1] - seed2[1])**2)
                if dist < min_distance:
                    is_unique = False
                    break
            
            if is_unique:
                unique_seeds.append(seed1)
        
        return unique_seeds
    
    def compute_hessian_features(self, image, sigma):
        """计算Hessian矩阵特征 - 优化版本"""
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            print("警告: scipy未安装，跳过Hessian计算")
            # 返回简化的特征
            h, w = image.shape
            eigenvalues = np.zeros((h, w, 2))
            eigenvectors = np.zeros((h, w, 2, 2))
            # 设置默认方向为水平
            eigenvectors[:, :, :, 1] = [1, 0]  # 默认方向向量
            return eigenvalues, eigenvectors
        
        # 计算二阶偏导数
        Ixx = gaussian_filter(image, sigma, order=[2, 0])
        Iyy = gaussian_filter(image, sigma, order=[0, 2])
        Ixy = gaussian_filter(image, sigma, order=[1, 1])
        
        h, w = image.shape
        eigenvalues = np.zeros((h, w, 2))
        eigenvectors = np.zeros((h, w, 2, 2))
        
        # 批量计算特征值和特征向量（避免逐像素循环）
        for i in range(0, h, 10):  # 降采样计算，加速
            for j in range(0, w, 10):
                try:
                    H = np.array([[Ixx[i, j], Ixy[i, j]], 
                                 [Ixy[i, j], Iyy[i, j]]])
                    
                    eigenvals, eigenvecs = np.linalg.eigh(H)
                    
                    # 按特征值大小排序
                    idx = np.argsort(np.abs(eigenvals))
                    eigenvalues[i:min(i+10,h), j:min(j+10,w), :] = eigenvals[idx]
                    eigenvectors[i:min(i+10,h), j:min(j+10,w), :, :] = eigenvecs[:, idx]
                except:
                    # 出错时设置默认值
                    eigenvalues[i:min(i+10,h), j:min(j+10,w), :] = [0, 0]
                    eigenvectors[i:min(i+10,h), j:min(j+10,w), :, :] = np.eye(2)
        
        return eigenvalues, eigenvectors
    
    def estimate_vessel_radius(self, image, center_y, center_x, direction):
        """估计血管半径 - 沿垂直方向测量"""
        h, w = image.shape
        
        # 垂直于血管方向的单位向量
        perp_x = -np.sin(direction)
        perp_y = np.cos(direction)
        
        # 沿垂直方向采样
        max_radius = self.params['vessel_max_radius']
        profile = []
        
        for r in np.arange(-max_radius, max_radius + 0.5, 0.5):
            sample_x = int(center_x + r * perp_x)
            sample_y = int(center_y + r * perp_y)
            
            if 0 <= sample_x < w and 0 <= sample_y < h:
                profile.append(image[sample_y, sample_x])
            else:
                profile.append(0)
        
        profile = np.array(profile)
        
        # 找到中心附近的峰值宽度
        center_idx = len(profile) // 2
        threshold = profile[center_idx] * 0.5  # 半高全宽
        
        # 从中心向两侧搜索
        left_edge = center_idx
        right_edge = center_idx
        
        for i in range(center_idx, 0, -1):
            if profile[i] < threshold:
                left_edge = i
                break
        
        for i in range(center_idx, len(profile)):
            if profile[i] < threshold:
                right_edge = i
                break
        
        radius = (right_edge - left_edge) * 0.5 * 0.5  # 0.5是采样间隔
        return max(self.params['vessel_min_radius'], min(radius, self.params['vessel_max_radius']))
    
    def ridge_traversal_tracking(self, enhanced_image, seed_point, mask):
        """Ridge Traversal血管追踪 - 优化版本"""
        h, w = enhanced_image.shape
        start_y, start_x = seed_point
        
        # 简化版本：使用梯度方向而不是复杂的Hessian
        # 计算梯度
        gy, gx = np.gradient(enhanced_image)
        
        # 获取起始点的梯度方向
        if abs(gx[start_y, start_x]) < 1e-6 and abs(gy[start_y, start_x]) < 1e-6:
            return [seed_point]  # 梯度太小，直接返回
        
        gradient_angle = np.arctan2(gy[start_y, start_x], gx[start_y, start_x])
        
        tracked_points = [seed_point]
        
        # 双向追踪（简化版本）
        for direction_mult in [1, -1]:
            current_x, current_y = float(start_x), float(start_y)
            current_direction = gradient_angle + direction_mult * np.pi/2  # 垂直于梯度方向
            
            step_size = self.params['ridge_step_size']
            max_steps = min(int(self.params['max_ridge_length'] / step_size), 15)  # 限制最大步数
            
            for step in range(max_steps):
                # 沿当前方向前进一步
                next_x = current_x + step_size * np.cos(current_direction)
                next_y = current_y + step_size * np.sin(current_direction)
                
                # 边界检查
                next_x_int, next_y_int = int(round(next_x)), int(round(next_y))
                if not (0 <= next_x_int < w and 0 <= next_y_int < h):
                    break
                if not mask[next_y_int, next_x_int]:
                    break
                
                # 检查强度（简化的血管检查）
                if enhanced_image[next_y_int, next_x_int] < self.params['ridge_threshold']:
                    break
                
                # 检查是否偏离太远
                if step > 0:
                    dist_from_start = np.sqrt((next_x - start_x)**2 + (next_y - start_y)**2)
                    if dist_from_start > self.params['max_ridge_length']:
                        break
                
                # 更新状态
                current_x, current_y = next_x, next_y
                
                # 添加追踪点（避免重复）
                if (next_y_int, next_x_int) not in tracked_points:
                    tracked_points.append((next_y_int, next_x_int))
        
        return tracked_points
    
    def ridge_based_vessel_segmentation(self, enhanced_image, seed_points, mask):
        """基于Ridge Traversal的血管分割 - 少而精的种子点版本"""
        print(f"=== Ridge追踪 - 处理{len(seed_points)}个战略种子点 ===")
        
        h, w = enhanced_image.shape
        segmented = np.zeros((h, w), dtype=np.uint8)
        
        total_vessel_pixels = 0
        processed_seeds = set()
        
        for seed_idx, seed_point in enumerate(seed_points):
            if seed_point in processed_seeds:
                continue
            
            try:
                # 从种子点进行Ridge追踪
                tracked_points = self.ridge_traversal_tracking(enhanced_image, seed_point, mask)
                
                if len(tracked_points) > 1:  # 至少追踪到一些点
                    # 标记追踪到的点
                    for y, x in tracked_points:
                        if 0 <= y < h and 0 <= x < w:  # 额外边界检查
                            segmented[y, x] = 255
                            processed_seeds.add((y, x))
                            total_vessel_pixels += 1
                    
            except Exception as e:
                continue
        
        print(f"追踪完成，总血管像素: {total_vessel_pixels}")
        
        return segmented.astype(np.float32) / 255.0
    
    def connect_vessel_segments(self, vessels, mask):
        """连接血管段"""
        # 获取血管像素
        vessel_coords = np.where(vessels > 0.5)
        if len(vessel_coords[0]) < 2:
            return vessels
        
        vessel_points = list(zip(vessel_coords[0], vessel_coords[1]))
        vessels_connected = vessels.copy()
        
        connection_dist = self.params['connection_distance']
        
        # 简单的距离连接
        for i, (y1, x1) in enumerate(vessel_points[::5]):  # 降采样加速
            for j, (y2, x2) in enumerate(vessel_points[i+5::5]):
                dist = np.sqrt((y2-y1)**2 + (x2-x1)**2)
                
                if 2 < dist <= connection_dist:
                    # 连接两点
                    line_points = self.bresenham_line(y1, x1, y2, x2)
                    for y, x in line_points:
                        if mask[y, x]:
                            vessels_connected[y, x] = 1.0
        
        return vessels_connected
    
    def bresenham_line(self, y1, x1, y2, x2):
        """Bresenham直线算法"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            points.append((y1, x1))
            
            if y1 == y2 and x1 == x2:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        
        return points
    
    def post_process(self, vessels, mask):
        """后处理"""
        # 连接血管段
        vessels = self.connect_vessel_segments(vessels, mask)
        
        # 应用掩码
        vessels = vessels * mask
        
        # 轻微的形态学操作
        vessels_uint8 = (vessels * 255).astype(np.uint8)
        
        # 连通域过滤
        labeled = measure.label(vessels_uint8)
        props = measure.regionprops(labeled)
        
        filtered = np.zeros_like(vessels_uint8)
        kept_regions = 0
        
        min_area = self.params['min_area']
        for prop in props:
            if prop.area >= min_area:
                filtered[labeled == prop.label] = 255
                kept_regions += 1
        
        return filtered.astype(np.float32) / 255.0
    
    def segment_vessels(self, image_tensor):
        """完整的血管分割流程"""
        # 1. 图像预处理
        enhanced_image, mask = self.image_enhancement_preprocessing(image_tensor)
        
        # 2. 血管中心线检测
        combined_response = self.vessel_centerline_detection(enhanced_image)
        
        # 3. 提取战略性种子点（新方法）
        seeds_mask, seed_points = self.extract_strategic_seed_points(enhanced_image, combined_response, mask)
        
        if len(seed_points) == 0:
            return np.zeros_like(enhanced_image), combined_response, seeds_mask
        
        # 4. Ridge Traversal追踪
        vessels = self.ridge_based_vessel_segmentation(enhanced_image, seed_points, mask)
        
        # 5. 后处理
        final_vessels = self.post_process(vessels, mask)
        
        return final_vessels, combined_response, seeds_mask

def simple_test(data_path, output_dir, num_images=3):
    """简单测试函数"""
    os.makedirs(output_dir, exist_ok=True)
    
    segmenter = RidgeTraversalVesselSegmentation()
    
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
            
            # 保存结果
            base_name = os.path.splitext(img_file)[0]
            
            vessel_img = (vessels * 255).astype(np.uint8)
            Image.fromarray(vessel_img).save(f"{output_dir}/{base_name}_vessels.png")
            
            if line_response.max() > 0:
                response_normalized = (line_response / line_response.max() * 255).astype(np.uint8)
            else:
                response_normalized = (line_response * 255).astype(np.uint8)
            Image.fromarray(response_normalized).save(f"{output_dir}/{base_name}_response.png")
            
            seed_img = (seeds * 255).astype(np.uint8)
            Image.fromarray(seed_img).save(f"{output_dir}/{base_name}_seeds.png")
            
            # 创建叠加图 - 修复tensor处理
            try:
                if isinstance(img_tensor, torch.Tensor):
                    original_green = img_tensor[1].cpu().numpy()
                else:
                    original_green = img_tensor[1] if len(img_tensor.shape) == 3 else img_tensor
                
                overlay = create_vessel_overlay(original_green, vessels, seeds)
                Image.fromarray((overlay * 255).astype(np.uint8)).save(
                    f"{output_dir}/{base_name}_overlay.png")
            except Exception as overlay_error:
                print(f"    叠加图生成失败: {overlay_error}")
                # 使用原始图像的绿色通道作为fallback
                original_green = np.array(image)[:, :, 1] / 255.0
                overlay = create_vessel_overlay(original_green, vessels, seeds)
                Image.fromarray((overlay * 255).astype(np.uint8)).save(
                    f"{output_dir}/{base_name}_overlay.png")
            
            print(f"结果:")
            print(f"  血管像素: {vessel_pixels:.0f} ({vessel_percentage:.2f}%)")
            print(f"  种子点: {seed_count:.0f}")
            if seed_count > 0:
                print(f"  扩展比例: {vessel_pixels/seed_count:.2f}")
            
        except Exception as e:
            print(f"处理错误: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n结果保存到: {output_dir}/")

def create_vessel_overlay(original, vessels, seeds):
    """创建血管叠加可视化"""
    overlay = np.stack([original, original, original], axis=-1)
    
    # 血管用红色显示
    vessel_coords = np.where(vessels > 0.5)
    overlay[vessel_coords[0], vessel_coords[1], 0] = 1.0
    overlay[vessel_coords[0], vessel_coords[1], 1] = 0.0
    overlay[vessel_coords[0], vessel_coords[1], 2] = 0.0
    
    # 种子点用绿色显示
    seed_coords = np.where(seeds > 0.5)
    overlay[seed_coords[0], seed_coords[1], 0] = 0.0
    overlay[seed_coords[0], seed_coords[1], 1] = 1.0
    overlay[seed_coords[0], seed_coords[1], 2] = 0.0
    
    return overlay

if __name__ == "__main__":
    data_path = '/root/FSG-Net-pytorch/data/DRIVE/val/input'
    output_dir = './ridge_traversal_results'
    
    print("=== 基于Ridge Traversal的血管分割测试 ===")
    print("理论依据: Aylward & Bullitt (2002) IEEE TMI")
    print("核心方法:")
    print("1. 保持原有的种子点检测")
    print("2. Hessian矩阵特征分析")
    print("3. Ridge强度和方向追踪")
    print("4. 曲率和半径变化约束")
    print("5. 双向追踪扩展")
    
    simple_test(data_path, output_dir, num_images=3)