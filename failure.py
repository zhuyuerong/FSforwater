import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.backbones.FSGNet import FSGNet
from PIL import Image
import os
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from scipy import ndimage
from skimage import measure, morphology, feature
import json
from collections import defaultdict, OrderedDict
from torchsummary import summary


class VesselFailureAnalyzer:
    """眼底血管分割失败模式深度分析工具"""
    
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        if model_path:
            self.model = self.load_fsg_model(model_path)
        
        # 分析结果存储
        self.analysis_results = {
            'optic_disc_analysis': [],
            'thin_vessel_analysis': [],
            'vessel_continuity_analysis': [],
            'global_statistics': {},
            'failure_patterns': [],
            'receptive_field_analysis': {}  # 新增感受野分析
        }
        
        # 感受野相关
        self.input_size = (3, 512, 512)
        self.activations = {}
        self.layer_names = []
    
    def load_fsg_model(self, model_path):
        """加载FSG模型"""
        model = FSGNet(channel=3, n_classes=1, base_c=64, depths=[3, 3, 9, 3], kernel_size=3)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 处理DataParallel保存的权重
        state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('module.FSGNet.'):
                new_key = key[len('module.FSGNet.'):]
                state_dict[new_key] = value
            elif key.startswith('module.'):
                new_key = key[len('module.'):]
                state_dict[new_key] = value
            else:
                state_dict[key] = value
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model
    
    def detect_optic_disc_advanced(self, image):
        """高级视杯检测方法"""
        if image.dtype == np.float64:
            image = image.astype(np.float32)
    
        # 如果图像值在[0,1]范围，转换到[0,255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 多尺度高斯模糊检测亮区域
        blurred_15 = cv2.GaussianBlur(gray, (15, 15), 0)
        blurred_31 = cv2.GaussianBlur(gray, (31, 31), 0)
        
        # 亮度阈值
        brightness_threshold = np.percentile(gray, 95)
        bright_mask = gray > brightness_threshold
        
        # 形态学操作保留圆形区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
        opened = cv2.morphologyEx(bright_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        # 寻找最大的圆形连通分量
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 选择最圆且面积较大的轮廓
            best_contour = None
            best_score = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1000:  # 太小的区域忽略
                    continue
                
                # 计算圆形度
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    score = area * circularity  # 面积和圆形度的综合评分
                    
                    if score > best_score:
                        best_score = score
                        best_contour = contour
            
            if best_contour is not None:
                mask = np.zeros_like(gray, dtype=np.uint8)
                cv2.fillPoly(mask, [best_contour], 1)
                mask = mask.astype(bool)

                # 获取视杯的中心和半径信息
                (x, y), radius = cv2.minEnclosingCircle(best_contour)
                center = (int(x), int(y))
                
                return mask, center, int(radius)
        
        return np.zeros_like(gray, dtype=bool), None, None
    
    def analyze_optic_disc_false_positives(self, predictions, ground_truths, original_images):
        """深度分析视杯区域假阳性"""
        optic_disc_results = []
        
        for i, (pred, gt, img) in enumerate(zip(predictions, ground_truths, original_images)):
            # 检测视杯
            od_mask, center, radius = self.detect_optic_disc_advanced(img)
            
            if od_mask.sum() == 0:
                continue
            
            # 计算假阳性
            pred_binary = pred > 0.5
            gt_binary = gt > 0.5
            fp_mask = pred_binary & (~gt_binary)
            
            # 视杯区域分析
            od_total_pixels = od_mask.sum()
            od_fp_pixels = (fp_mask & od_mask).sum()
            od_fp_rate = od_fp_pixels / od_total_pixels if od_total_pixels > 0 else 0
            
            # 非视杯区域假阳性率作为对比
            non_od_mask = ~od_mask
            non_od_total = non_od_mask.sum()
            non_od_fp = (fp_mask & non_od_mask).sum()
            non_od_fp_rate = non_od_fp / non_od_total if non_od_total > 0 else 0
            
            # 视杯亮度分析
                        # 视杯亮度分析
            if len(img.shape) == 3:
                # 确保图像格式正确用于cv2处理
                img_for_cv2 = img.copy()
                if img_for_cv2.dtype == np.float64:
                    img_for_cv2 = img_for_cv2.astype(np.float32)
                if img_for_cv2.max() <= 1.0:
                    img_for_cv2 = (img_for_cv2 * 255).astype(np.uint8)
                elif img_for_cv2.dtype != np.uint8:
                    img_for_cv2 = img_for_cv2.astype(np.uint8)

                od_brightness = np.mean(cv2.cvtColor(img_for_cv2, cv2.COLOR_RGB2GRAY)[od_mask])
            else:
                od_brightness = np.mean(img[od_mask])
            
            result = {
                'image_id': i,
                'od_center': center,
                'od_radius': radius,
                'od_area': od_total_pixels,
                'od_fp_pixels': od_fp_pixels,
                'od_fp_rate': od_fp_rate,
                'non_od_fp_rate': non_od_fp_rate,
                'od_brightness': od_brightness,
                'fp_ratio_od_vs_other': od_fp_rate / non_od_fp_rate if non_od_fp_rate > 0 else float('inf')
            }
            
            optic_disc_results.append(result)
            
        return optic_disc_results
    
    def analyze_thin_vessel_false_negatives(self, predictions, ground_truths):
        """分析细血管假阴性"""
        thin_vessel_results = []
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            pred_binary = pred > 0.5
            gt_binary = gt > 0.5
            fn_mask = (~pred_binary) & gt_binary
            
            if fn_mask.sum() == 0:
                continue
            
            # 骨架化分析血管结构
            gt_skeleton = morphology.skeletonize(gt_binary)
            
            # 计算血管宽度（距离变换）
            distance_transform = ndimage.distance_transform_edt(gt_binary)
            vessel_widths = distance_transform[gt_skeleton]
            
            # 定义细血管（宽度小于某个阈值）
            thin_threshold = np.percentile(vessel_widths[vessel_widths > 0], 30) if len(vessel_widths) > 0 else 2
            
            # 在骨架上找细血管区域
            thin_vessel_mask = gt_skeleton & (distance_transform <= thin_threshold)
            
            # 分析细血管的假阴性
            thin_vessel_fn = fn_mask & thin_vessel_mask
            thick_vessel_fn = fn_mask & gt_skeleton & (~thin_vessel_mask)
            
            # 端点分析
            endpoints = self.find_vessel_endpoints(gt_skeleton)
            endpoint_fn = fn_mask & endpoints
            
            # 分支点分析
            branch_points = self.find_branch_points(gt_skeleton)
            branch_fn = fn_mask & branch_points
            
            result = {
                'image_id': i,
                'total_fn_pixels': fn_mask.sum(),
                'thin_vessel_fn': thin_vessel_fn.sum(),
                'thick_vessel_fn': thick_vessel_fn.sum(),
                'endpoint_fn': endpoint_fn.sum(),
                'branch_fn': branch_fn.sum(),
                'thin_vessel_fn_rate': thin_vessel_fn.sum() / thin_vessel_mask.sum() if thin_vessel_mask.sum() > 0 else 0,
                'avg_thin_vessel_width': thin_threshold
            }
            
            thin_vessel_results.append(result)
            
        return thin_vessel_results
    
    def find_vessel_endpoints(self, skeleton):
        """找到血管端点"""
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        endpoints = skeleton & (neighbor_count == 1)
        return endpoints
    
    def find_branch_points(self, skeleton):
        """找到血管分支点"""
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        branch_points = skeleton & (neighbor_count >= 3)
        return branch_points
    
    def analyze_vessel_continuity(self, predictions, ground_truths):
        """分析血管连续性问题"""
        continuity_results = []
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            pred_binary = pred > 0.5
            gt_binary = gt > 0.5
            
            # 分析ground truth中的连通分量
            gt_labeled = measure.label(gt_binary)
            gt_components = np.unique(gt_labeled)[1:]
            
            disconnected_vessels = 0
            total_vessel_length = 0
            predicted_vessel_length = 0
            
            for comp_id in gt_components:
                comp_mask = (gt_labeled == comp_id)
                comp_skeleton = morphology.skeletonize(comp_mask)
                vessel_length = comp_skeleton.sum()
                total_vessel_length += vessel_length
                
                # 检查这个血管在预测中的连续性
                pred_in_vessel = pred_binary & comp_mask
                pred_labeled = measure.label(pred_in_vessel)
                pred_components = len(np.unique(pred_labeled)[1:])
                
                if pred_components > 1:
                    disconnected_vessels += 1
                
                predicted_vessel_length += pred_in_vessel.sum()
            
            # 计算连续性指标
            connectivity_ratio = predicted_vessel_length / total_vessel_length if total_vessel_length > 0 else 0
            disconnection_rate = disconnected_vessels / len(gt_components) if len(gt_components) > 0 else 0
            
            result = {
                'image_id': i,
                'total_vessels': len(gt_components),
                'disconnected_vessels': disconnected_vessels,
                'disconnection_rate': disconnection_rate,
                'connectivity_ratio': connectivity_ratio,
                'total_vessel_length': total_vessel_length,
                'predicted_vessel_length': predicted_vessel_length
            }
            
            continuity_results.append(result)
            
        return continuity_results
    
    def calculate_contrast_map(self, image):
        """计算图像对比度图"""
        # 确保图像格式正确
        if image.dtype == np.float64:
            image = image.astype(np.float32)

        # 如果图像值在[0,1]范围，转换到[0,255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 使用标准差作为局部对比度指标
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        
        # 局部均值
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # 局部标准差（对比度）
        local_sq_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        local_contrast = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        
        return local_contrast
    
    def analyze_low_contrast_regions(self, predictions, ground_truths, original_images):
        """分析低对比度区域的表现"""
        contrast_results = []
        
        for i, (pred, gt, img) in enumerate(zip(predictions, ground_truths, original_images)):
            contrast_map = self.calculate_contrast_map(img)
            
            # 定义低对比度区域
            low_contrast_threshold = np.percentile(contrast_map, 25)
            low_contrast_mask = contrast_map < low_contrast_threshold
            high_contrast_mask = contrast_map >= np.percentile(contrast_map, 75)
            
            pred_binary = pred > 0.5
            gt_binary = gt > 0.5
            fn_mask = (~pred_binary) & gt_binary
            
            # 低对比度区域的假阴性
            low_contrast_fn = (fn_mask & low_contrast_mask).sum()
            low_contrast_gt = (gt_binary & low_contrast_mask).sum()
            low_contrast_fn_rate = low_contrast_fn / low_contrast_gt if low_contrast_gt > 0 else 0
            
            # 高对比度区域的假阴性
            high_contrast_fn = (fn_mask & high_contrast_mask).sum()
            high_contrast_gt = (gt_binary & high_contrast_mask).sum()
            high_contrast_fn_rate = high_contrast_fn / high_contrast_gt if high_contrast_gt > 0 else 0
            
            result = {
                'image_id': i,
                'low_contrast_fn_rate': low_contrast_fn_rate,
                'high_contrast_fn_rate': high_contrast_fn_rate,
                'contrast_sensitivity': low_contrast_fn_rate / high_contrast_fn_rate if high_contrast_fn_rate > 0 else float('inf'),
                'avg_contrast': contrast_map.mean(),
                'contrast_std': contrast_map.std()
            }
            
            contrast_results.append(result)
            
        return contrast_results
    
    def analyze_receptive_field_vs_vessels(self, predictions, ground_truths, original_images):
        """分析感受野与血管宽度的关系"""
        print("分析感受野与血管特征的关系...")
        
        if self.model is None:
            print("警告: 没有模型，跳过感受野分析")
            return {}
        
        # 1. 计算理论感受野
        rf_info = self.calculate_theoretical_receptive_field()
        
        # 2. 分析实际血管宽度分布
        vessel_width_stats = []
        
        for i, (pred, gt, img) in enumerate(zip(predictions, ground_truths, original_images)):
            gt_binary = gt > 0.5
            pred_binary = pred > 0.5
            fn_mask = (~pred_binary) & gt_binary
            
            if gt_binary.sum() == 0:
                continue
            
            # 计算距离变换得到血管宽度
            distance_transform = ndimage.distance_transform_edt(gt_binary)
            skeleton = morphology.skeletonize(gt_binary)
            
            # 在骨架上提取血管宽度
            vessel_widths = distance_transform[skeleton] * 2  # 半径*2=直径
            vessel_widths = vessel_widths[vessel_widths > 0]
            
            if len(vessel_widths) == 0:
                continue
            
            # 分析不同宽度血管的检测成功率
            width_bins = [0, 2, 4, 6, 8, 10, 15, float('inf')]
            width_labels = ['≤2px', '2-4px', '4-6px', '6-8px', '8-10px', '10-15px', '>15px']
            
            width_detection_rates = []
            
            for j in range(len(width_bins)-1):
                min_w, max_w = width_bins[j], width_bins[j+1]
                
                # 找到这个宽度范围的血管像素
                width_mask = (distance_transform >= min_w/2) & (distance_transform < max_w/2) & gt_binary
                
                if width_mask.sum() > 0:
                    # 计算这个宽度范围的检测率
                    detected = pred_binary & width_mask
                    detection_rate = detected.sum() / width_mask.sum()
                else:
                    detection_rate = 0
                
                width_detection_rates.append(detection_rate)
            
            vessel_width_stats.append({
                'image_id': i,
                'mean_vessel_width': np.mean(vessel_widths),
                'median_vessel_width': np.median(vessel_widths),
                'min_vessel_width': np.min(vessel_widths),
                'max_vessel_width': np.max(vessel_widths),
                'width_std': np.std(vessel_widths),
                'width_bins': width_labels,
                'detection_rates_by_width': width_detection_rates,
                'total_vessel_pixels': gt_binary.sum(),
                'total_detected_pixels': pred_binary.sum()
            })
        
        # 3. 感受野与血管宽度的对比分析
        max_theoretical_rf = rf_info[-1]['receptive_field'] if rf_info else 0
        
        # 计算每个宽度范围的平均检测率
        avg_detection_by_width = {}
        if vessel_width_stats:
            for i, width_label in enumerate(width_labels):
                rates = [stat['detection_rates_by_width'][i] for stat in vessel_width_stats]
                avg_detection_by_width[width_label] = np.mean(rates)
        
        # 4. 分析感受野不足的影响
        rf_analysis = {
            'theoretical_max_rf': max_theoretical_rf,
            'vessel_width_distribution': vessel_width_stats,
            'avg_detection_by_width': avg_detection_by_width,
            'rf_coverage_analysis': {}
        }
        
        # 分析哪些血管宽度可能受感受野限制
        for width_label, detection_rate in avg_detection_by_width.items():
            # 估计这个宽度范围的代表值
            if '≤2' in width_label:
                representative_width = 1.5
            elif '2-4' in width_label:
                representative_width = 3
            elif '4-6' in width_label:
                representative_width = 5
            elif '6-8' in width_label:
                representative_width = 7
            elif '8-10' in width_label:
                representative_width = 9
            elif '10-15' in width_label:
                representative_width = 12.5
            else:
                representative_width = 20
            
            # 分析感受野是否足够
            rf_sufficient = representative_width <= max_theoretical_rf
            rf_coverage_ratio = min(1.0, max_theoretical_rf / representative_width) if representative_width > 0 else 1.0
            
            rf_analysis['rf_coverage_analysis'][width_label] = {
                'representative_width': representative_width,
                'detection_rate': detection_rate,
                'rf_sufficient': rf_sufficient,
                'rf_coverage_ratio': rf_coverage_ratio,
                'potential_rf_limitation': detection_rate < 0.7 and not rf_sufficient
            }
        
        return rf_analysis
    
    def calculate_theoretical_receptive_field(self):
        """计算理论感受野"""
        if self.model is None:
            return []
        
        rf_info = []
        current_rf = 1
        current_stride = 1
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
                
                # 计算感受野
                current_rf = current_rf + (kernel_size - 1) * current_stride
                current_stride = current_stride * stride
                
                rf_info.append({
                    'layer_name': name,
                    'layer_type': 'Conv2d',
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': padding,
                    'receptive_field': current_rf,
                    'effective_stride': current_stride
                })
                
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                kernel_size = module.kernel_size
                stride = module.stride if module.stride is not None else kernel_size
                
                current_rf = current_rf + (kernel_size - 1) * current_stride
                current_stride = current_stride * stride
                
                rf_info.append({
                    'layer_name': name,
                    'layer_type': type(module).__name__,
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': getattr(module, 'padding', 0),
                    'receptive_field': current_rf,
                    'effective_stride': current_stride
                })
        
        return rf_info
    
    def get_model_summary(self):
        """获取模型结构摘要"""
        if self.model is None:
            return "模型未加载"
        
        try:
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                summary(self.model, self.input_size, device='cpu')
            
            return f.getvalue()
        except Exception as e:
            return f"无法生成模型摘要: {e}"
    
    def comprehensive_failure_analysis(self, data_path, gt_paths, output_dir):
        """执行全面的失败模式分析"""
        print("开始执行全面失败模式分析...")
        
        # 收集所有数据
        predictions = []
        ground_truths = []
        original_images = []
        image_names = []
        
        # 查找图像文件
        image_files = [f for f in os.listdir(data_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        
        print(f"找到 {len(image_files)} 张图像")
        
        for img_file in image_files:
            try:
                # 加载原始图像
                img_path = os.path.join(data_path, img_file)
                image = Image.open(img_path).convert('RGB')
                img_array = np.array(image) / 255.0
                original_images.append(img_array)
                
                # 加载预测结果（如果模型存在）
                if self.model:
                    img_tensor = torch.tensor(img_array.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        pred_out1, _, _ = self.model(img_tensor)
                        prediction = pred_out1.squeeze().cpu().numpy()
                else:
                    # 如果没有模型，尝试从已保存的结果中加载
                    pred_path = os.path.join(output_dir, 'fsg_original', f"{os.path.splitext(img_file)[0]}_fsg.png")
                    if os.path.exists(pred_path):
                        pred_img = Image.open(pred_path).convert('L')
                        prediction = np.array(pred_img) / 255.0
                    else:
                        print(f"警告: 找不到 {img_file} 的预测结果")
                        continue
                
                predictions.append(prediction)
                
                # 查找对应的真实标签
                gt_path = self.find_ground_truth(img_file, gt_paths)
                if gt_path:
                    gt_image = Image.open(gt_path).convert('L')
                    ground_truth = np.array(gt_image) / 255.0
                    ground_truth = (ground_truth > 0.5).astype(np.float32)  # 二值化
                    ground_truths.append(ground_truth)
                    image_names.append(img_file)
                else:
                    print(f"警告: 找不到 {img_file} 的真实标签")
                    continue
                    
            except Exception as e:
                print(f"处理 {img_file} 时出错: {e}")
                continue
        
        if len(predictions) == 0:
            print("没有找到有效的预测结果和真实标签对")
            return
        
        print(f"成功加载 {len(predictions)} 对图像用于分析")
        
        # 执行各种失败模式分析
        print("1. 分析视杯区域假阳性...")
        optic_disc_results = self.analyze_optic_disc_false_positives(predictions, ground_truths, original_images)
        
        print("2. 分析细血管假阴性...")
        thin_vessel_results = self.analyze_thin_vessel_false_negatives(predictions, ground_truths)
        
        print("3. 分析血管连续性...")
        continuity_results = self.analyze_vessel_continuity(predictions, ground_truths)
        
        print("4. 分析低对比度区域表现...")
        contrast_results = self.analyze_low_contrast_regions(predictions, ground_truths, original_images)
        
        print("5. 分析感受野与血管宽度关系...")
        rf_analysis = self.analyze_receptive_field_vs_vessels(predictions, ground_truths, original_images)
        
        # 保存分析结果
        self.analysis_results = {
            'optic_disc_analysis': optic_disc_results,
            'thin_vessel_analysis': thin_vessel_results,
            'vessel_continuity_analysis': continuity_results,
            'contrast_analysis': contrast_results,
            'receptive_field_analysis': rf_analysis,  # 新增
            'image_names': image_names
        }
        
        # 生成统计报告
        self.generate_statistical_report(output_dir)
        
        # 生成可视化图表
        self.generate_visualization_report(output_dir)
        
        # 生成感受野专门报告
        if rf_analysis:
            self.generate_receptive_field_report(output_dir)
        
        print(f"分析完成！结果保存在 {output_dir}")
        
        return self.analysis_results
    
    def find_ground_truth(self, img_file, gt_paths):
        """查找对应的真实标签文件"""
        base_name = os.path.splitext(img_file)[0]
        
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
        
        for gt_dir in gt_paths:
            if os.path.exists(gt_dir):
                for possible_name in possible_names:
                    gt_path = os.path.join(gt_dir, possible_name)
                    if os.path.exists(gt_path):
                        return gt_path
        return None
    
    def generate_receptive_field_report(self, output_dir):
        """生成感受野专门报告"""
        rf_data = self.analysis_results.get('receptive_field_analysis', {})
        if not rf_data:
            return
        
        report_path = os.path.join(output_dir, 'receptive_field_analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== FSGNet 感受野与血管检测关系分析 ===\n\n")
            
            # 1. 基础感受野信息
            if 'theoretical_max_rf' in rf_data:
                f.write("1. 感受野基础信息\n")
                f.write("-" * 40 + "\n")
                f.write(f"理论最大感受野: {rf_data['theoretical_max_rf']} 像素\n\n")
            
            # 2. 不同宽度血管的检测性能
            if 'avg_detection_by_width' in rf_data:
                f.write("2. 不同宽度血管检测性能\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'血管宽度':<15} {'平均检测率':<12} {'状态':<10}\n")
                f.write("-" * 50 + "\n")
                
                for width_label, detection_rate in rf_data['avg_detection_by_width'].items():
                    status = "优秀" if detection_rate > 0.8 else "良好" if detection_rate > 0.6 else "需改进"
                    f.write(f"{width_label:<15} {detection_rate:<12.3f} {status:<10}\n")
                f.write("\n")
            
            # 3. 感受野覆盖分析
            if 'rf_coverage_analysis' in rf_data:
                f.write("3. 感受野覆盖能力分析\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'血管宽度':<15} {'覆盖率':<10} {'检测率':<10} {'潜在RF限制':<12}\n")
                f.write("-" * 60 + "\n")
                
                rf_limited_widths = []
                for width_label, analysis in rf_data['rf_coverage_analysis'].items():
                    coverage_ratio = analysis['rf_coverage_ratio']
                    detection_rate = analysis['detection_rate']
                    rf_limitation = analysis['potential_rf_limitation']
                    
                    limitation_str = "是" if rf_limitation else "否"
                    f.write(f"{width_label:<15} {coverage_ratio:<10.2f} {detection_rate:<10.3f} {limitation_str:<12}\n")
                    
                    if rf_limitation:
                        rf_limited_widths.append(width_label)
                
                f.write("\n")
                
                # 4. 关键发现和建议
                f.write("4. 关键发现和改进建议\n")
                f.write("-" * 40 + "\n")
                
                if rf_limited_widths:
                    f.write(f"⚠️  受感受野限制的血管宽度: {', '.join(rf_limited_widths)}\n\n")
                    f.write("建议的改进措施:\n")
                    f.write("1. 增加扩张卷积 (Dilated Convolution):\n")
                    f.write("   - 在网络中后期添加扩张率为2-4的卷积层\n")
                    f.write("   - 可以在不增加参数的情况下扩大感受野\n\n")
                    
                    f.write("2. 多尺度特征融合:\n")
                    f.write("   - 使用不同尺度的特征图进行融合\n")
                    f.write("   - 实现类似FPN的多尺度架构\n\n")
                    
                    f.write("3. 注意力机制:\n")
                    f.write("   - 添加空间注意力模块\n")
                    f.write("   - 帮助模型关注细血管区域\n\n")
                    
                    f.write("4. 损失函数优化:\n")
                    f.write("   - 对细血管区域增加权重\n")
                    f.write("   - 使用Focal Loss处理样本不平衡\n\n")
                else:
                    f.write("✅ 当前感受野能够很好地覆盖各种宽度的血管\n")
                    f.write("模型在感受野方面表现良好，可以专注于其他优化方向\n\n")
                
                # 5. 定量改进建议
                max_rf = rf_data.get('theoretical_max_rf', 0)
                f.write("5. 定量改进目标\n")
                f.write("-" * 40 + "\n")
                f.write(f"当前理论感受野: {max_rf} 像素\n")
                
                # 找到检测率最低的血管宽度
                worst_performance = min(rf_data['avg_detection_by_width'].items(), 
                                      key=lambda x: x[1])
                worst_width, worst_rate = worst_performance
                
                f.write(f"表现最差的血管宽度: {worst_width} (检测率: {worst_rate:.3f})\n")
                
                # 建议的感受野目标
                target_rf = max_rf * 1.5  # 建议增加50%
                f.write(f"建议目标感受野: {target_rf:.0f} 像素\n")
                f.write(f"需要增加: {target_rf - max_rf:.0f} 像素\n\n")
        
        print(f"感受野分析报告已保存: {report_path}")
    
    def generate_statistical_report(self, output_dir):
        """生成统计报告"""
        report_path = os.path.join(output_dir, 'failure_analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 眼底血管分割失败模式分析报告 ===\n\n")
            
            # 视杯分析
            if self.analysis_results['optic_disc_analysis']:
                od_data = self.analysis_results['optic_disc_analysis']
                f.write("1. 视杯区域假阳性分析\n")
                f.write("-" * 30 + "\n")
                
                avg_od_fp_rate = np.mean([x['od_fp_rate'] for x in od_data])
                avg_non_od_fp_rate = np.mean([x['non_od_fp_rate'] for x in od_data])
                avg_fp_ratio = np.mean([x['fp_ratio_od_vs_other'] for x in od_data if x['fp_ratio_od_vs_other'] != float('inf')])
                
                f.write(f"平均视杯区域假阳性率: {avg_od_fp_rate:.3f}\n")
                f.write(f"平均非视杯区域假阳性率: {avg_non_od_fp_rate:.3f}\n")
                f.write(f"视杯区域假阳性率是其他区域的 {avg_fp_ratio:.2f} 倍\n")
                f.write(f"分析图像数量: {len(od_data)}\n\n")
            
            # 细血管分析
            if self.analysis_results['thin_vessel_analysis']:
                tv_data = self.analysis_results['thin_vessel_analysis']
                f.write("2. 细血管假阴性分析\n")
                f.write("-" * 30 + "\n")
                
                avg_thin_fn_rate = np.mean([x['thin_vessel_fn_rate'] for x in tv_data])
                total_endpoint_fn = sum([x['endpoint_fn'] for x in tv_data])
                total_branch_fn = sum([x['branch_fn'] for x in tv_data])
                
                f.write(f"平均细血管假阴性率: {avg_thin_fn_rate:.3f}\n")
                f.write(f"总端点假阴性数: {total_endpoint_fn}\n")
                f.write(f"总分支点假阴性数: {total_branch_fn}\n")
                f.write(f"分析图像数量: {len(tv_data)}\n\n")
            
            # 连续性分析
            if self.analysis_results['vessel_continuity_analysis']:
                cont_data = self.analysis_results['vessel_continuity_analysis']
                f.write("3. 血管连续性分析\n")
                f.write("-" * 30 + "\n")
                
                avg_disconnection_rate = np.mean([x['disconnection_rate'] for x in cont_data])
                avg_connectivity_ratio = np.mean([x['connectivity_ratio'] for x in cont_data])
                
                f.write(f"平均血管断裂率: {avg_disconnection_rate:.3f}\n")
                f.write(f"平均连通性比率: {avg_connectivity_ratio:.3f}\n")
                f.write(f"分析图像数量: {len(cont_data)}\n\n")
            
            # 对比度分析
            if self.analysis_results['contrast_analysis']:
                contrast_data = self.analysis_results['contrast_analysis']
                f.write("4. 对比度敏感性分析\n")
                f.write("-" * 30 + "\n")
                
                avg_low_contrast_fn = np.mean([x['low_contrast_fn_rate'] for x in contrast_data])
                avg_high_contrast_fn = np.mean([x['high_contrast_fn_rate'] for x in contrast_data])
                avg_sensitivity = np.mean([x['contrast_sensitivity'] for x in contrast_data if x['contrast_sensitivity'] != float('inf')])
                
                f.write(f"低对比度区域假阴性率: {avg_low_contrast_fn:.3f}\n")
                f.write(f"高对比度区域假阴性率: {avg_high_contrast_fn:.3f}\n")
                f.write(f"对比度敏感性(低/高): {avg_sensitivity:.2f}\n")
                f.write(f"分析图像数量: {len(contrast_data)}\n\n")
            
            # 感受野分析摘要
            if 'receptive_field_analysis' in self.analysis_results and self.analysis_results['receptive_field_analysis']:
                rf_data = self.analysis_results['receptive_field_analysis']
                f.write("5. 感受野分析摘要\n")
                f.write("-" * 30 + "\n")
                
                if 'theoretical_max_rf' in rf_data:
                    f.write(f"理论最大感受野: {rf_data['theoretical_max_rf']} 像素\n")
                
                if 'avg_detection_by_width' in rf_data:
                    # 找到表现最差的血管宽度
                    worst_width = min(rf_data['avg_detection_by_width'].items(), key=lambda x: x[1])
                    best_width = max(rf_data['avg_detection_by_width'].items(), key=lambda x: x[1])
                    
                    f.write(f"检测率最低的血管宽度: {worst_width[0]} ({worst_width[1]:.3f})\n")
                    f.write(f"检测率最高的血管宽度: {best_width[0]} ({best_width[1]:.3f})\n")
                
                # 统计受感受野限制的血管类型
                if 'rf_coverage_analysis' in rf_data:
                    rf_limited = [width for width, analysis in rf_data['rf_coverage_analysis'].items() 
                                if analysis.get('potential_rf_limitation', False)]
                    if rf_limited:
                        f.write(f"可能受感受野限制的血管宽度: {', '.join(rf_limited)}\n")
                    else:
                        f.write("所有血管宽度都能被当前感受野很好覆盖\n")
                
                f.write(f"详细感受野分析请查看: receptive_field_analysis_report.txt\n\n")
        
        print(f"统计报告已保存至: {report_path}")
    
    def generate_visualization_report(self, output_dir):
        """Generate visualization report"""
        plt.style.use('default')
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Retinal Vessel Segmentation Failure Mode Analysis', fontsize=16, fontweight='bold')

        # 1. Optic disc false positive rate distribution
        if self.analysis_results['optic_disc_analysis']:
            od_data = self.analysis_results['optic_disc_analysis']
            od_fp_rates = [x['od_fp_rate'] for x in od_data]
            non_od_fp_rates = [x['non_od_fp_rate'] for x in od_data]

            axes[0, 0].hist([od_fp_rates, non_od_fp_rates], bins=20, alpha=0.7, 
                           label=['Optic Disc Region', 'Non-Optic Disc Region'], color=['red', 'blue'])
            axes[0, 0].set_title('False Positive Rate Distribution Comparison')
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Thin vessel false negative analysis
        if self.analysis_results['thin_vessel_analysis']:
            tv_data = self.analysis_results['thin_vessel_analysis']
            thin_fn_rates = [x['thin_vessel_fn_rate'] for x in tv_data]

            axes[0, 1].boxplot([thin_fn_rates], labels=['Thin Vessel FN Rate'])
            axes[0, 1].set_title('Thin Vessel False Negative Rate Distribution')
            axes[0, 1].set_ylabel('False Negative Rate')
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Vessel continuity
        if self.analysis_results['vessel_continuity_analysis']:
            cont_data = self.analysis_results['vessel_continuity_analysis']
            disconnection_rates = [x['disconnection_rate'] for x in cont_data]
            connectivity_ratios = [x['connectivity_ratio'] for x in cont_data]

            axes[0, 2].scatter(disconnection_rates, connectivity_ratios, alpha=0.6)
            axes[0, 2].set_title('Vessel Continuity Scatter Plot')
            axes[0, 2].set_xlabel('Disconnection Rate')
            axes[0, 2].set_ylabel('Connectivity Ratio')
            axes[0, 2].grid(True, alpha=0.3)

        # 4. Contrast sensitivity
        if self.analysis_results['contrast_analysis']:
            contrast_data = self.analysis_results['contrast_analysis']
            low_contrast_fn = [x['low_contrast_fn_rate'] for x in contrast_data]
            high_contrast_fn = [x['high_contrast_fn_rate'] for x in contrast_data]

            x = np.arange(len(low_contrast_fn))
            width = 0.35

            axes[1, 0].bar(x - width/2, low_contrast_fn, width, label='Low Contrast Region', alpha=0.8)
            axes[1, 0].bar(x + width/2, high_contrast_fn, width, label='High Contrast Region', alpha=0.8)
            axes[1, 0].set_title('False Negative Rate by Contrast Region')
            axes[1, 0].set_xlabel('Image Index')
            axes[1, 0].set_ylabel('False Negative Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Receptive field vs vessel width relationship
        if 'receptive_field_analysis' in self.analysis_results and self.analysis_results['receptive_field_analysis']:
            rf_data = self.analysis_results['receptive_field_analysis']

            if 'avg_detection_by_width' in rf_data:
                width_labels = list(rf_data['avg_detection_by_width'].keys())
                detection_rates = list(rf_data['avg_detection_by_width'].values())

                bars = axes[1, 1].bar(range(len(width_labels)), detection_rates, alpha=0.8, color='skyblue')
                axes[1, 1].set_title('Detection Rate by Vessel Width')
                axes[1, 1].set_xlabel('Vessel Width')
                axes[1, 1].set_ylabel('Detection Rate')
                axes[1, 1].set_xticks(range(len(width_labels)))
                axes[1, 1].set_xticklabels(width_labels, rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Excellent (0.8)')
                axes[1, 1].axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Good (0.6)')
                axes[1, 1].legend()

                # Add value annotations on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # 6. Receptive field coverage analysis
        if 'receptive_field_analysis' in self.analysis_results and self.analysis_results['receptive_field_analysis']:
            rf_data = self.analysis_results['receptive_field_analysis']

            if 'rf_coverage_analysis' in rf_data:
                coverage_data = rf_data['rf_coverage_analysis']
                width_labels = list(coverage_data.keys())
                coverage_ratios = [coverage_data[w]['rf_coverage_ratio'] for w in width_labels]
                detection_rates = [coverage_data[w]['detection_rate'] for w in width_labels]

                # Scatter plot: RF coverage ratio vs detection rate
                colors = ['red' if coverage_data[w].get('potential_rf_limitation', False) else 'blue' 
                         for w in width_labels]

                scatter = axes[1, 2].scatter(coverage_ratios, detection_rates, c=colors, alpha=0.7, s=100)
                axes[1, 2].set_title('RF Coverage Ratio vs Detection Rate')
                axes[1, 2].set_xlabel('RF Coverage Ratio')
                axes[1, 2].set_ylabel('Detection Rate')
                axes[1, 2].grid(True, alpha=0.3)

                # Add labels
                for i, label in enumerate(width_labels):
                    axes[1, 2].annotate(label, (coverage_ratios[i], detection_rates[i]), 
                                      xytext=(5, 5), textcoords='offset points', fontsize=8)

                # Add ideal line
                axes[1, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Ideal Relationship')
                axes[1, 2].legend(['Ideal Relationship', 'Normal', 'RF Limited'])

        # 7. Comprehensive performance heatmap
        if all(k in self.analysis_results for k in ['optic_disc_analysis', 'thin_vessel_analysis', 'vessel_continuity_analysis']):
            # Create comprehensive performance metric matrix
            metrics_data = []
            image_indices = []

            # Collect comprehensive metrics for all images
            max_images = min(len(self.analysis_results['optic_disc_analysis']),
                           len(self.analysis_results['thin_vessel_analysis']),
                           len(self.analysis_results['vessel_continuity_analysis']))

            for i in range(max_images):
                od_fp_rate = self.analysis_results['optic_disc_analysis'][i]['od_fp_rate']
                thin_fn_rate = self.analysis_results['thin_vessel_analysis'][i]['thin_vessel_fn_rate']
                disconnection_rate = self.analysis_results['vessel_continuity_analysis'][i]['disconnection_rate']

                metrics_data.append([od_fp_rate, thin_fn_rate, disconnection_rate])
                image_indices.append(f'Image {i+1}')

            if metrics_data:
                metrics_array = np.array(metrics_data).T
                im = axes[2, 0].imshow(metrics_array, cmap='Reds', aspect='auto')
                axes[2, 0].set_title('Comprehensive Failure Mode Heatmap')
                axes[2, 0].set_ylabel('Failure Type')
                axes[2, 0].set_xlabel('Images')
                axes[2, 0].set_yticks([0, 1, 2])
                axes[2, 0].set_yticklabels(['OD False Positive', 'Thin Vessel FN', 'Vessel Break'])

                # Show only part of image labels to avoid crowding
                step = max(1, len(image_indices) // 10)
                axes[2, 0].set_xticks(range(0, len(image_indices), step))
                axes[2, 0].set_xticklabels(image_indices[::step], rotation=45)

                plt.colorbar(im, ax=axes[2, 0], label='Failure Rate')

        # 8. Key issues summary pie chart
        if self.analysis_results['optic_disc_analysis'] and self.analysis_results['thin_vessel_analysis']:
            od_issues = len([x for x in self.analysis_results['optic_disc_analysis'] if x['od_fp_rate'] > 0.1])
            thin_issues = len([x for x in self.analysis_results['thin_vessel_analysis'] if x['thin_vessel_fn_rate'] > 0.3])
            continuity_issues = len([x for x in self.analysis_results['vessel_continuity_analysis'] if x['disconnection_rate'] > 0.2])

            issue_counts = [od_issues, thin_issues, continuity_issues]
            issue_labels = ['Severe OD FP', 'Severe Thin Vessel Miss', 'Severe Vessel Break']
            colors = ['#ff9999', '#66b3ff', '#99ff99']

            axes[2, 1].pie(issue_counts, labels=issue_labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[2, 1].set_title('Main Issues Distribution')

        # 9. Model structure summary
        if self.model:
            # Create a simple model structure text plot
            axes[2, 2].axis('off')

            # Get receptive field information
            rf_info = self.calculate_theoretical_receptive_field()
            if rf_info:
                # Show key layer receptive field development
                key_layers = rf_info[::max(1, len(rf_info)//5)]  # Take 5 key layers

                text_content = "FSGNet Key Layer RF:\n\n"
                for layer in key_layers:
                    text_content += f"{layer['layer_name'].split('.')[-1]}: {layer['receptive_field']}px\n"

                if 'receptive_field_analysis' in self.analysis_results:
                    rf_data = self.analysis_results['receptive_field_analysis']
                    if 'theoretical_max_rf' in rf_data:
                        text_content += f"\nMax RF: {rf_data['theoretical_max_rf']}px"

                axes[2, 2].text(0.1, 0.9, text_content, transform=axes[2, 2].transAxes,
                               fontsize=10, verticalalignment='top', fontfamily='monospace',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

            axes[2, 2].set_title('Model RF Information')

        plt.tight_layout()

        # Save charts
        chart_path = os.path.join(output_dir, 'comprehensive_failure_analysis.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'comprehensive_failure_analysis.pdf'), bbox_inches='tight')
        plt.show()

        print(f"Visualization charts saved to: {chart_path}")
    
    def generate_region_weight_map(self, image, analysis_results):
        """基于失败模式分析生成区域权重图"""
        h, w = image.shape[:2] if len(image.shape) == 2 else image.shape[:2]
        weight_map = np.ones((h, w), dtype=np.float32)
        
        # 检测视杯区域并降权
        od_mask, center, radius = self.detect_optic_disc_advanced(image)
        if od_mask.sum() > 0:
            weight_map[od_mask] = 0.3  # 大幅降低视杯区域权重
        
        # 计算对比度图，对低对比度区域提高权重
        contrast_map = self.calculate_contrast_map(image)
        low_contrast_threshold = np.percentile(contrast_map, 25)
        low_contrast_mask = contrast_map < low_contrast_threshold
        weight_map[low_contrast_mask] *= 1.5  # 提高低对比度区域权重
        
        # 检测可能的血管区域（使用简单的形态学方法）
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 增强细血管区域的权重
        enhanced = cv2.GaussianBlur(gray, (5, 5), 0)
        vessel_response = cv2.Laplacian(enhanced, cv2.CV_64F)
        vessel_response = np.abs(vessel_response)
        
        # 对可能的血管区域增加权重
        vessel_threshold = np.percentile(vessel_response, 85)
        potential_vessel_mask = vessel_response > vessel_threshold
        weight_map[potential_vessel_mask] *= 1.8
        
        # 归一化权重到合理范围
        weight_map = np.clip(weight_map, 0.1, 3.0)
        
        return weight_map
    
    def create_adaptive_loss_function(self):
        """创建基于失败模式分析的自适应损失函数"""
        def adaptive_weighted_loss(pred, target, original_image, loss_weights=None):
            """
            自适应加权损失函数
            
            Args:
                pred: 预测结果 [B, 1, H, W]
                target: 真实标签 [B, 1, H, W] 
                original_image: 原始图像 [B, 3, H, W]
                loss_weights: 额外的损失权重字典
            """
            if loss_weights is None:
                loss_weights = {
                    'optic_disc_weight': 0.3,      # 视杯区域降权
                    'thin_vessel_weight': 2.0,      # 细血管区域加权
                    'low_contrast_weight': 1.5,     # 低对比度区域加权
                    'focal_alpha': 0.25,            # Focal Loss参数
                    'focal_gamma': 2.0
                }
            
            batch_size = pred.shape[0]
            total_loss = 0.0
            
            for i in range(batch_size):
                # 转换为numpy进行权重计算
                img_np = original_image[i].cpu().numpy().transpose(1, 2, 0)
                
                # 生成权重图
                weight_map = self.generate_region_weight_map(img_np, self.analysis_results)
                weight_tensor = torch.from_numpy(weight_map).to(pred.device).unsqueeze(0)
                
                # 基础二元交叉熵
                pred_i = pred[i]
                target_i = target[i]
                
                # Focal Loss计算
                ce_loss = nn.functional.binary_cross_entropy_with_logits(
                    pred_i, target_i, reduction='none'
                )
                pt = torch.exp(-ce_loss)
                focal_loss = loss_weights['focal_alpha'] * (1-pt)**loss_weights['focal_gamma'] * ce_loss
                
                # 应用区域权重
                weighted_loss = focal_loss * weight_tensor
                
                # 额外的边缘损失（强调血管边界）
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
                
                target_edge_x = nn.functional.conv2d(target_i, sobel_x, padding=1)
                target_edge_y = nn.functional.conv2d(target_i, sobel_y, padding=1)
                target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2)
                
                pred_edge_x = nn.functional.conv2d(torch.sigmoid(pred_i), sobel_x, padding=1)
                pred_edge_y = nn.functional.conv2d(torch.sigmoid(pred_i), sobel_y, padding=1)
                pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2)
                
                edge_loss = nn.functional.mse_loss(pred_edge, target_edge)
                
                # 组合损失
                image_loss = weighted_loss.mean() + 0.1 * edge_loss
                total_loss += image_loss
            
            return total_loss / batch_size
        
        return adaptive_weighted_loss


def main_analysis():
    """主分析函数"""
    # 配置路径
    model_path = '/root/FSG-Net-pytorch/model_ckpts/FSG-Net-DRIVE.pt'
    
    # 数据路径
    possible_data_paths = [
        '/root/FSG-Net-pytorch/data/DRIVE/val/input',
        '/root/FSG-Net-pytorch/data/DRIVE/train/input',
    ]
    
    possible_gt_paths = [
        '/root/FSG-Net-pytorch/data/DRIVE/val/gt',
        '/root/FSG-Net-pytorch/data/DRIVE/train/gt',
        '/root/FSG-Net-pytorch/data/DRIVE/val/label',
        '/root/FSG-Net-pytorch/data/DRIVE/train/label',
    ]
    
    output_dir = './comprehensive_failure_analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 找到有效的数据路径
    data_path = None
    for path in possible_data_paths:
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            if len(files) > 0:
                data_path = path
                print(f"使用数据路径: {path}")
                break
    
    if data_path is None:
        print("未找到有效的数据路径！")
        return
    
    # 初始化分析器
    print("初始化全面失败模式分析器...")
    analyzer = VesselFailureAnalyzer(model_path)
    
    # 执行全面分析
    results = analyzer.comprehensive_failure_analysis(
        data_path=data_path,
        gt_paths=possible_gt_paths,
        output_dir=output_dir
    )
    
    if results:
        print("\n=== 分析结果总结 ===")
        
        # 视杯分析总结
        if results['optic_disc_analysis']:
            od_data = results['optic_disc_analysis']
            avg_od_fp = np.mean([x['od_fp_rate'] for x in od_data])
            avg_other_fp = np.mean([x['non_od_fp_rate'] for x in od_data])
            print(f"视杯区域平均假阳性率: {avg_od_fp:.3f}")
            print(f"其他区域平均假阳性率: {avg_other_fp:.3f}")
            print(f"视杯问题严重程度: {avg_od_fp/avg_other_fp:.2f}倍")
        
        # 细血管分析总结
        if results['thin_vessel_analysis']:
            tv_data = results['thin_vessel_analysis']
            avg_thin_fn = np.mean([x['thin_vessel_fn_rate'] for x in tv_data])
            print(f"细血管平均假阴性率: {avg_thin_fn:.3f}")
        
        # 连续性分析总结
        if results['vessel_continuity_analysis']:
            cont_data = results['vessel_continuity_analysis']
            avg_disconnection = np.mean([x['disconnection_rate'] for x in cont_data])
            print(f"平均血管断裂率: {avg_disconnection:.3f}")
        
        # 感受野分析总结
        if 'receptive_field_analysis' in results and results['receptive_field_analysis']:
            rf_data = results['receptive_field_analysis']
            if 'theoretical_max_rf' in rf_data:
                print(f"理论最大感受野: {rf_data['theoretical_max_rf']} 像素")
            
            if 'avg_detection_by_width' in rf_data:
                worst_detection = min(rf_data['avg_detection_by_width'].items(), key=lambda x: x[1])
                print(f"检测率最低的血管宽度: {worst_detection[0]} ({worst_detection[1]:.3f})")
        
        print(f"\n详细分析结果已保存至: {output_dir}")
        print("下一步建议:")
        print("1. 查看 failure_analysis_report.txt 了解详细统计")
        print("2. 查看 receptive_field_analysis_report.txt 了解感受野分析")
        print("3. 查看 comprehensive_failure_analysis.png 了解可视化结果")
        print("4. 基于分析结果优化模型架构或实施针对性改进")
        
        # 生成自适应损失函数
        adaptive_loss = analyzer.create_adaptive_loss_function()
        print("5. 使用生成的自适应损失函数重新训练模型")
        
        return analyzer, results
    
    return None, None


if __name__ == "__main__":
    analyzer, results = main_analysis()