"""
图像处理工具模块
实现原MATLAB图像处理函数的OpenCV等价实现
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """
    将彩色图像转换为灰度图像
    对应MATLAB: rgb2gray(img)
    
    Args:
        image: 输入彩色图像 (BGR格式)
        
    Returns:
        灰度图像
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    中值滤波去噪
    对应MATLAB: medfilt2(img, [3,3])
    
    Args:
        image: 输入图像
        kernel_size: 滤波核大小
        
    Returns:
        滤波后的图像
    """
    return cv2.medianBlur(image, kernel_size)

def adjust_contrast(image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
    """
    调整图像对比度和亮度
    对应MATLAB: imadjust(img)
    
    Args:
        image: 输入图像
        alpha: 对比度增益 (1.0-3.0)
        beta: 亮度偏移 (0-100)
        
    Returns:
        调整后的图像
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    直方图均衡化增强对比度
    对应MATLAB: histeq(img)
    
    Args:
        image: 输入灰度图像
        
    Returns:
        均衡化后的图像
    """
    return cv2.equalizeHist(image)

def canny_edge_detection(image: np.ndarray, threshold1: int = 50, threshold2: int = 150) -> np.ndarray:
    """
    Canny边缘检测
    对应MATLAB: edge(img, 'Canny')
    
    Args:
        image: 输入灰度图像
        threshold1: 第一阈值
        threshold2: 第二阈值
        
    Returns:
        边缘图像
    """
    return cv2.Canny(image, threshold1, threshold2)

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    旋转图像
    对应MATLAB: imrotate(img, angle, 'bilinear', 'crop')
    
    Args:
        image: 输入图像
        angle: 旋转角度 (度)
        
    Returns:
        旋转后的图像
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 执行旋转
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    return rotated

def morphological_open(image: np.ndarray, kernel_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
    """
    形态学开运算（先腐蚀后膨胀）
    对应MATLAB: imopen(img, se)
    
    Args:
        image: 输入二值图像
        kernel_size: 结构元素大小
        
    Returns:
        开运算后的图像
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def morphological_close(image: np.ndarray, kernel_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
    """
    形态学闭运算（先膨胀后腐蚀）
    对应MATLAB: imclose(img, se)
    
    Args:
        image: 输入二值图像
        kernel_size: 结构元素大小
        
    Returns:
        闭运算后的图像
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def connected_components(image: np.ndarray) -> Tuple[int, np.ndarray, Dict[int, Dict[str, Any]]]:
    """
    连通域分析
    对应MATLAB: bwconncomp(img) + regionprops
    
    Args:
        image: 输入二值图像（0-255）
        
    Returns:
        (num_labels, labels, stats_dict)
        num_labels: 连通域数量
        labels: 标记图像
        stats_dict: 每个连通域的统计信息
    """
    # 确保输入是二值图像
    if image.dtype != np.uint8:
        image = np.uint8(image)
    
    # 执行连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 8, cv2.CV_32S)
    
    # 构建统计信息字典
    stats_dict = {}
    for i in range(1, num_labels):  # 跳过背景
        stats_dict[i] = {
            'area': stats[i, cv2.CC_STAT_AREA],
            'bbox': (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT]
            ),
            'centroid': (centroids[i, 0], centroids[i, 1])
        }
    
    return num_labels - 1, labels, stats_dict  # 减去背景

def binarize(image: np.ndarray, threshold: Optional[int] = None) -> np.ndarray:
    """
    图像二值化
    对应MATLAB: imbinarize(img)
    
    Args:
        image: 输入灰度图像
        threshold: 阈值（None表示自动计算）
        
    Returns:
        二值图像（0和255）
    """
    if threshold is None:
        # 自动计算阈值（OTSU方法）
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    
    return binary

def remove_small_objects(image: np.ndarray, min_size: int) -> np.ndarray:
    """
    移除小对象
    对应MATLAB: bwareaopen(img, min_size)
    
    Args:
        image: 输入二值图像
        min_size: 最小对象大小
        
    Returns:
        处理后的图像
    """
    # 使用连通域分析移除小对象
    num_labels, labels, stats_dict = connected_components(image)
    
    # 创建掩码
    mask = np.zeros_like(labels, dtype=np.uint8)
    
    # 保留大于min_size的对象
    for i, stats in stats_dict.items():
        if stats['area'] >= min_size:
            mask[labels == i] = 255
    
    return mask

def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    调整图像大小
    对应MATLAB: imresize(img, size)
    
    Args:
        image: 输入图像
        size: 目标大小 (height, width)
        
    Returns:
        调整大小后的图像
    """
    return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    图像预处理（组合操作）
    对应MATLAB: preprocess_image(img)
    
    Args:
        image: 输入图像
        
    Returns:
        预处理后的图像
    """
    # 1. 转换为灰度图
    gray = rgb_to_gray(image)
    
    # 2. 中值滤波去噪
    filtered = median_filter(gray, 3)
    
    # 3. 对比度拉伸
    enhanced = adjust_contrast(filtered, alpha=1.5, beta=0)
    
    return enhanced

def calculate_gradient(image: np.ndarray) -> np.ndarray:
    """
    计算图像梯度
    对应MATLAB: gradient(double(img))
    
    Args:
        image: 输入灰度图像
        
    Returns:
        梯度幅值图像
    """
    # 转换为浮点数
    img_float = image.astype(np.float64)
    
    # 计算x和y方向梯度
    sobelx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    return gradient_magnitude

def hough_lines(image: np.ndarray, threshold: int = 100) -> List[Tuple[float, float]]:
    """
    Hough变换检测直线
    对应MATLAB: hough(img)
    
    Args:
        image: 输入边缘图像
        threshold: 阈值
        
    Returns:
        直线列表 [(rho, theta), ...]
    """
    # 执行Hough变换
    lines = cv2.HoughLines(image, 1, np.pi/180, threshold)
    
    if lines is None:
        return []
    
    return [(float(line[0][0]), float(line[0][1])) for line in lines]

def detect_tilt_angle(image: np.ndarray) -> float:
    """
    检测图像倾斜角度
    对应MATLAB: detect_tilt_angle(img)
    
    Args:
        image: 输入灰度图像
        
    Returns:
        倾斜角度（度）
    """
    # 边缘检测
    edges = canny_edge_detection(image, 50, 150)
    
    # Hough变换检测直线
    lines = hough_lines(edges, 50)
    
    if not lines:
        return 0.0
    
    # 计算所有直线的角度
    angles = []
    for rho, theta in lines:
        angle = theta * 180 / np.pi - 90  # 转换为角度并调整
        # 过滤接近水平或垂直的线
        if 5 < abs(angle) < 85:
            angles.append(angle)
    
    if not angles:
        return 0.0
    
    # 返回平均角度
    return float(np.mean(angles))

def detect_interference_level(image: np.ndarray) -> float:
    """
    检测图像中的干扰程度
    对应MATLAB: detect_interference(img)
    
    Args:
        image: 输入灰度图像
        
    Returns:
        干扰程度（0-1）
    """
    # 计算梯度
    gradient_magnitude = calculate_gradient(image)
    
    # 计算梯度方差和最大值
    std_gradient = np.std(gradient_magnitude)
    max_gradient = np.max(gradient_magnitude)
    
    if max_gradient == 0:
        return 0.0
    
    # 计算干扰程度
    level = std_gradient / max_gradient
    
    # 归一化到0-1范围
    return float(min(level, 1.0))