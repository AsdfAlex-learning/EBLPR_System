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
    定位车牌区域（垂直投影法）
    对应MATLAB: locate_plate_region(img)
    
    Args:
        image: 输入灰度图像
        
    Returns:
        包含车牌区域信息的字典 {
            'bbox': (x, y, width, height),
            'image': 车牌区域图像
        }
    """
    # 二值化处理
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 垂直投影
    vertical_proj = np.sum(binary, axis=0)
    
    # 平滑投影曲线
    kernel = np.ones(5, dtype=np.float32) / 5
    proj_smoothed = np.convolve(vertical_proj, kernel, mode='same')
    
    # 计算差分找峰值
    diff_proj = np.diff(proj_smoothed)
    peak_indices = np.where((diff_proj[:-1] > 0) & (diff_proj[1:] < 0))[0]
    
    h, w = image.shape
    
    if len(peak_indices) > 0:
        # 计算峰值中心
        mid = int(np.mean(peak_indices))
        # 估计车牌宽度（图像宽度的30%）
        width = int(w * 0.3)
        # 估计车牌高度（宽高比约5:1）
        height = int(width / 5)
        
        # 计算边界
        x1 = max(0, mid - width // 2)
        x2 = min(w, mid + width // 2)
        y1 = max(0, h // 2 - height // 2)
        y2 = min(h, h // 2 + height // 2)
    else:
        # 强制裁剪中间区域
        x1 = int(w * 0.2)
        x2 = int(w * 0.8)
        y1 = int(h * 0.3)
        y2 = int(h * 0.7)
    
    # 提取车牌区域
    plate_image = image[y1:y2, x1:x2]
    
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

def extract_plate_region(image: np.ndarray, 
                        plate_region: Dict[str, Any]) -> np.ndarray:
    """
    提取车牌区域
    对应MATLAB: extract_plate_region(img, plate_region)
    
    Args:
        image: 输入灰度图像
        plate_region: 车牌区域信息
        
    Returns:
        提取的车牌区域图像
    """
    x, y, width, height = plate_region['bbox']
    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height
    
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