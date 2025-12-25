"""
车牌检测模块
实现车牌区域的定位、检测和提取功能
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from .image_utils import (
    preprocess_image,
    canny_edge_detection,
    connected_components,
    calculate_gradient
)

def locate_plate_region(image: np.ndarray) -> Dict[str, Any]:
    """
    定位车牌区域（极致缩小版：进一步收紧尺寸和定位精度）
    """
    # 1. 预处理（保持不变）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_enhanced = clahe.apply(image)
    image_blurred = cv2.GaussianBlur(image_enhanced, (3, 3), 1.0)
    binary_adaptive = cv2.adaptiveThreshold(
        image_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 3, 1
    )
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary_closed = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel_close)
    binary_processed = cv2.morphologyEx(binary_closed, cv2.MORPH_OPEN, kernel_open)
    
    # 2. 垂直投影（优化峰值筛选）
    vertical_proj = np.sum(binary_processed, axis=0)
    proj_smoothed = cv2.GaussianBlur(vertical_proj.astype(np.float32), (0, 0), 1.0)
    diff_proj = np.diff(proj_smoothed)
    peak_indices = np.where((diff_proj[:-1] > 0) & (diff_proj[1:] < 0))[0]
    
    h, w = image.shape
    
    if len(peak_indices) > 0:
        # 优化：只保留前20%高值峰值，且最多5个
        peak_values = proj_smoothed[peak_indices]
        peak_threshold = np.percentile(peak_values, 80)  # 70→80
        valid_peaks = peak_indices[peak_values >= peak_threshold]
        if len(valid_peaks) == 0:
            valid_peaks = peak_indices
        # 限制峰值数量
        if len(valid_peaks) > 5:
            peak_sorted = sorted(valid_peaks, key=lambda i: proj_smoothed[i], reverse=True)
            valid_peaks = np.array(peak_sorted[:5])
        
        # 取投影值最大的峰值作为中心
        mid = valid_peaks[np.argmax(proj_smoothed[valid_peaks])]
        
        # 3. 尺寸计算：极致缩小
        width = int(w * 0.20)   # 25%→20%
        width = max(width, int(w * 0.15))  # 20%→18%
        width = min(width, int(w * 0.20))  # 30%→25%
        
        height = int(width / 2.2)  # 3.5→3.8
        height = max(height, int(h * 0.08)) # 10%→8%
        height = min(height, int(h * 0.15)) # 20%→15%
        
        # 4. 垂直边界优化：取高值行中点
        horizontal_proj = np.sum(binary_processed, axis=1)
        # 新增：筛选高值行，取中点
        horiz_threshold = np.percentile(horizontal_proj, 70)
        high_value_rows = np.where(horizontal_proj >= horiz_threshold)[0]
        if len(high_value_rows) > 0:
            y_peak = int((high_value_rows[0] + high_value_rows[-1]) / 2)
        else:
            y_peak = np.argmax(horizontal_proj)
        
        x1 = max(0, mid - width // 2)
        x2 = min(w, mid + width // 2)
        y1 = max(0, y_peak - height // 2)
        y2 = min(h, y_peak + height // 2)
        
        print(f"[DEBUG] 极致缩小定位: mid={mid}, width={width}, height={height}")
        print(f"[DEBUG] 极致缩小边界: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
        
    else:
        # 强制裁剪：进一步缩小
        width = int(w * 0.12)  # 15%→12%
        height = int(width / 2.2)  # 3.5→3.8
        x1 = int(w * 0.25)     # 20%→25%
        x2 = int(w * 0.75)     # 80%→75%
        y1 = int(h * 0.35)     # 30%→35%
        y2 = int(h * 0.65)     # 70%→65%
        
        print(f"[DEBUG] 极致缩小强制定位: width={width}, height={height}")
        print(f"[DEBUG] 极致缩小强制边界: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
    
    # 区域尺寸验证：降低阈值+缩小兜底尺寸
    if (x2 - x1) < 25 or (y2 - y1) < 12:  # 30→25, 15→12
        width = min(w, 40)  # 60→50
        height = min(h, 15) # 20→18
        x1 = max(0, w // 2 - width // 2)
        x2 = min(w, w // 2 + width // 2)
        y1 = max(0, h // 2 - height // 2)
        y2 = min(h, h // 2 + height // 2)
        
        print(f"[DEBUG] 区域过小，极致缩小默认尺寸: width={width}, height={height}")
    
    # 提取并优化车牌区域（保持不变）
    plate_image = image[y1:y2, x1:x2]
    if plate_image.size > 0:
        plate_enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(plate_image)
        plate_normalized = cv2.normalize(plate_enhanced, None, 0, 255, cv2.NORM_MINMAX)
        plate_image = plate_normalized
    
    return {
        'bbox': (x1, y1, x2 - x1, y2 - y1),
        'image': plate_image
    }

def detect_plate_count(image: np.ndarray) -> int:
    """
    检测图像中车牌区域的数量
    对应MATLAB: detect_plate_count(img)
    
    Args:
        image: 输入灰度图像
        
    Returns:
        车牌区域数量
    """
    # 二值化处理
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 连通域分析
    num_regions, _, stats_dict = connected_components(binary)
    
    # 计算图像总面积
    h, w = image.shape
    img_area = h * w
    
    # 过滤可能是车牌的区域
    min_area = img_area * 0.05
    max_area = img_area * 0.5
    
    valid_count = 0
    for _, stats in stats_dict.items():
        if min_area <= stats['area'] <= max_area:
            valid_count += 1
    
    # 限制最大数量
    return min(valid_count, 5)

def detect_all_plates(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    检测图像中所有可能的车牌区域
    对应MATLAB: detect_all_plates(img)
    
    Args:
        image: 输入灰度图像
        
    Returns:
        车牌区域列表
    """
    # 二值化处理
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 连通域分析
    num_regions, labels, stats_dict = connected_components(binary)
    
    # 计算图像总面积
    h, w = image.shape
    img_area = h * w
    
    plate_regions = []
    
    for _, stats in stats_dict.items():
        area = stats['area']
        bbox = stats['bbox']
        x, y, width, height = bbox
        
        if height > 0:
            aspect_ratio = width / height
            # 计算紧凑度（面积与边界框面积的比值）
            bbox_area = width * height
            extent = area / bbox_area if bbox_area > 0 else 0
            
            # 过滤条件：面积、宽高比、紧凑度
            if (img_area * 0.05 <= area <= img_area * 0.5 and
                1.5 <= aspect_ratio <= 6 and
                extent > 0.5):
                
                plate_regions.append({
                    'bbox': bbox,
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'extent': extent
                })
    
    # 按面积降序排序
    plate_regions.sort(key=lambda x: x['area'], reverse=True)
    
    return plate_regions

def select_target_plate(plate_regions: List[Dict[str, Any]], 
                       image: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    选择目标车牌（选择最大或最清晰的车牌）
    对应MATLAB: select_target_plate(plate_regions, img)
    
    Args:
        plate_regions: 车牌区域列表
        image: 输入灰度图像
        
    Returns:
        目标车牌区域信息
    """
    if not plate_regions:
        return None
    
    if len(plate_regions) == 1:
        return plate_regions[0]
    
    # 计算每个区域的清晰度得分
    scores = []
    for region in plate_regions:
        x, y, width, height = region['bbox']
        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height
        
        if x2 > x1 and y2 > y1:
            # 提取区域
            region_img = image[y1:y2, x1:x2]
            
            # 计算梯度方差作为清晰度指标
            gradient_magnitude = calculate_gradient(region_img)
            std_gradient = np.std(gradient_magnitude)
            
            # 得分 = 清晰度 × 面积
            score = std_gradient * region['area']
            scores.append(score)
        else:
            # 无效区域，仅使用面积
            scores.append(region['area'])
    
    # 选择得分最高的区域
    best_index = np.argmax(scores)
    return plate_regions[best_index]

def extract_plate_region(image, plate_region):
    """
    提取车牌区域
    
    Args:
        image: 输入图像（彩色或灰度）
        plate_region: 车牌区域信息
        
    Returns:
        提取的车牌区域图像
    """
    x, y, width, height = plate_region['bbox']
    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height
    
    # 处理彩色和灰度图像
    if len(image.shape) == 3:
        h, w, _ = image.shape
    else:
        h, w = image.shape
        
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    if x2 > x1 and y2 > y1:
        return image[y1:y2, x1:x2]
    else:
        # 如果区域无效，返回整个图像
        return image

def detect_image_type(image: np.ndarray) -> str:
    """
    检测图像类型：正常、倾斜、干扰、多车牌
    对应MATLAB: detect_image_type(img)
    
    Args:
        image: 输入灰度图像
        
    Returns:
        图像类型字符串 ('normal', 'tilted', 'interfered', 'multiple')
    """
    from .image_utils import detect_tilt_angle, detect_interference_level
    
    # 检测倾斜角度
    angle = detect_tilt_angle(image)
    
    # 检测干扰程度
    interference_level = detect_interference_level(image)
    
    # 检测车牌数量
    plate_count = detect_plate_count(image)
    
    # 根据特征判断图像类型
    if plate_count > 1:
        return 'multiple'
    elif abs(angle) > 2 and abs(angle) <= 15:
        return 'tilted'
    elif interference_level > 0.3:
        return 'interfered'
    else:
        return 'normal'

def remove_interference(image: np.ndarray) -> np.ndarray:
    """
    去除图像中的干扰文字和图案
    对应MATLAB: remove_interference(img)
    
    Args:
        image: 输入灰度图像
        
    Returns:
        去干扰后的图像
    """
    # 二值化处理
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 1. 形态学操作去除小干扰（开运算）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 2. 使用连通域分析去除孤立的小区域
    num_regions, labels, stats_dict = connected_components(opened)
    
    if num_regions > 0:
        # 计算面积阈值（小于平均面积的区域可能是干扰）
        areas = [stats['area'] for stats in stats_dict.values()]
        mean_area = np.mean(areas)
        min_area = mean_area * 0.1
        
        # 创建掩码
        mask = np.zeros_like(labels, dtype=np.uint8)
        for i, stats in stats_dict.items():
            if stats['area'] >= min_area:
                mask[labels == i] = 255
        
        # 应用掩码
        cleaned = cv2.bitwise_and(opened, mask)
    else:
        cleaned = opened
    
    # 3. 中值滤波进一步平滑
    cleaned = cv2.medianBlur(cleaned, 3)
    
    return cleaned

def enhance_after_cleaning(image: np.ndarray) -> np.ndarray:
    """
    去干扰后的图像增强
    对应MATLAB: enhance_after_cleaning(img)
    
    Args:
        image: 输入二值图像
        
    Returns:
        增强后的图像
    """
    # 1. 对比度增强（直方图均衡化）
    if image.max() > 0:
        # 确保图像是uint8类型
        if image.dtype != np.uint8:
            image = np.uint8(image)
        
        # 直方图均衡化
        enhanced = cv2.equalizeHist(image)
    else:
        enhanced = image
    
    # 2. 形态学闭运算连接字符
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    
    return enhanced

def enhance_normal_plate(image: np.ndarray) -> np.ndarray:
    """
    增强正常车牌图像
    对应MATLAB: enhance_normal_plate(img)
    
    Args:
        image: 输入灰度图像
        
    Returns:
        增强后的图像
    """
    # 确保图像是uint8类型
    if image.dtype != np.uint8:
        image = np.uint8(image)
    
    # 1. 对比度增强
    enhanced = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    # 2. 轻微锐化
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced