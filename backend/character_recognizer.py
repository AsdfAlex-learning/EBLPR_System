"""
字符识别模块
实现车牌字符的分割和识别功能
"""
import cv2
import numpy as np
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# 全局变量存储字符模板和标签
_char_templates = []
_char_labels = []

def load_char_templates(template_dir: str = None) -> bool:
    """
    加载字符模板库
    对应MATLAB: load_char_templates()
    
    Args:
        template_dir: 模板目录路径
        
    Returns:
        是否加载成功
    """
    global _char_templates, _char_labels
    
    # 清空全局变量
    _char_templates = []
    _char_labels = []
    
    # 确定模板目录
    if not template_dir:
        # 默认路径
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent
        template_dir = str(project_root / "opencv_core" / "char_templates")
    
    if not os.path.isdir(template_dir):
        print(f"字符模板文件夹不存在: {template_dir}")
        return False
    
    # 定义字符标签顺序（包含中文省份/城市，移除车牌无的I/O）
    labels = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
        'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
        'W', 'X', 'Y', 'Z', '粤', '广', '州', '佛', '山'
    ]
    
    # 加载每个字符的模板
    for label in labels:
        template_path = os.path.join(template_dir, f"{label}.png")
        
        if os.path.exists(template_path):
            try:
                # 读取模板图像
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                
                if template is not None:
                    # 二值化
                    _, template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # 统一尺寸为40x20
                    template = cv2.resize(template, (20, 40), interpolation=cv2.INTER_NEAREST)
                    
                    _char_templates.append(template)
                    _char_labels.append(label)
                else:
                    print(f"无法读取模板文件: {template_path}")
                    
            except Exception as e:
                print(f"加载模板 {label} 时出错: {e}")
        else:
            print(f"模板文件缺失: {template_path}")
    
    if not _char_templates:
        print("未加载到任何字符模板")
        return False
    
    print(f"成功加载 {len(_char_templates)} 个字符模板")
    return True

def segment_characters(plate_region: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    分割车牌字符
    对应MATLAB: segment_characters(plate_region)
    
    Args:
        plate_region: 包含车牌区域信息的字典
        
    Returns:
        字符列表，每个字符是一个字典 {
            'image': 字符图像,
            'position': 位置信息
        }
    """
    plate_image = plate_region['image']
    
    # 确保图像是灰度图
    if len(plate_image.shape) == 3:
        plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    h, w = plate_image.shape
    
    # 电动自行车车牌共7个字符，按宽度均分（固定位置）
    char_width = w // 7
    characters = []
    
    for i in range(7):
        start_col = i * char_width
        end_col = min((i + 1) * char_width, w)
        
        # 提取字符
        char_img = plate_image[:, start_col:end_col]
        
        # 二值化
        _, char_img_bin = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 去除边缘噪点
        char_img_clean = cv2.morphologyEx(char_img_bin, cv2.MORPH_OPEN, 
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        
        characters.append({
            'image': char_img_clean,
            'position': (start_col, 0, end_col - start_col, h)
        })
    
    return characters

def match_character(char_img: np.ndarray) -> str:
    """
    匹配单个字符（模板匹配）
    对应MATLAB: match_character(char_img)
    
    Args:
        char_img: 字符图像
        
    Returns:
        匹配的字符标签，匹配失败返回 '?'
    """
    global _char_templates, _char_labels
    
    # 确保模板已加载
    if not _char_templates:
        if not load_char_templates():
            return '?'
    
    # 确保字符图像是二值图像
    if char_img.max() > 0:
        if char_img.dtype != np.uint8:
            char_img = np.uint8(char_img)
        
        # 二值化
        _, char_img_bin = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY)
    else:
        return '?'
    
    # 去除孤立小噪点
    char_img_clean = cv2.morphologyEx(char_img_bin, cv2.MORPH_OPEN, 
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    
    # 形态学闭运算，修复字符断裂
    char_img_close = cv2.morphologyEx(char_img_clean, cv2.MORPH_CLOSE, 
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
    
    # 强制统一尺寸（与模板严格对齐）
    char_img_final = cv2.resize(char_img_close, (20, 40), interpolation=cv2.INTER_NEAREST)
    
    # 计算与每个模板的汉明距离
    min_dist = float('inf')
    best_label = '?'
    
    for i, template in enumerate(_char_templates):
        # 确保模板是二值图像
        if template.dtype != np.uint8:
            template = np.uint8(template)
        
        _, template_bin = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY)
        
        # 计算汉明距离（不同像素的比例）
        diff = np.logical_xor(char_img_final, template_bin)
        dist = np.sum(diff) / diff.size
        
        # 降低匹配阈值到0.3，提升识别成功率
        if dist < min_dist and dist < 0.3:
            min_dist = dist
            best_label = _char_labels[i]
    
    return best_label

def recognize_characters(characters: List[Dict[str, Any]]) -> str:
    """
    识别车牌字符
    对应MATLAB: recognize_characters(characters)
    
    Args:
        characters: 字符列表
        
    Returns:
        识别结果字符串
    """
    if not characters:
        return '识别失败：未检测到字符'
    
    # 初始化结果
    result_text = ''
    
    # 逐个识别字符
    for char_info in characters:
        char_img = char_info['image']
        char_label = match_character(char_img)
        result_text += char_label
    
    # 检查识别结果
    if not result_text or all(c == '?' for c in result_text):
        return '识别失败：未匹配到有效字符'
    
    return result_text

def recognize_normal_plate(img: np.ndarray) -> str:
    """
    正常车牌识别算法
    对应MATLAB: recognize_normal_plate(img)
    
    Args:
        img: 预处理后的图像
        
    Returns:
        识别结果
    """
    from .plate_detector import locate_plate_region
    
    # 1. 车牌区域定位
    plate_region = locate_plate_region(img)
    
    # 2. 字符分割
    characters = segment_characters(plate_region)
    
    # 3. 字符识别
    result = recognize_characters(characters)
    
    return result

def recognize_tilted_plate(img: np.ndarray) -> str:
    """
    倾斜车牌矫正识别算法
    对应MATLAB: recognize_tilted_plate(img)
    
    Args:
        img: 输入图像
        
    Returns:
        识别结果
    """
    from .image_utils import detect_tilt_angle, rotate_image, preprocess_image
    
    # 1. 检测倾斜角度
    angle = detect_tilt_angle(img)
    
    # 限制角度范围
    if abs(angle) > 15:
        angle = np.sign(angle) * 15
    
    # 2. 图像旋转矫正
    rotated = rotate_image(img, -angle)
    
    # 3. 重新预处理
    img_processed = preprocess_image(rotated)
    
    # 4. 使用正常识别流程
    return recognize_normal_plate(img_processed)

def recognize_interfered_plate(img: np.ndarray) -> str:
    """
    文字干扰车牌处理算法
    对应MATLAB: recognize_interfered_plate(img)
    
    Args:
        img: 输入图像
        
    Returns:
        识别结果
    """
    from .plate_detector import remove_interference, enhance_after_cleaning
    
    # 1. 去除干扰
    img_cleaned = remove_interference(img)
    
    # 2. 增强图像
    img_enhanced = enhance_after_cleaning(img_cleaned)
    
    # 3. 使用正常识别流程
    return recognize_normal_plate(img_enhanced)

def recognize_multiple_plates(img: np.ndarray) -> str:
    """
    多车牌场景识别算法
    对应MATLAB: recognize_multiple_plates(img)
    
    Args:
        img: 输入图像
        
    Returns:
        识别结果
    """
    from .plate_detector import detect_all_plates, select_target_plate, extract_plate_region
    
    # 1. 检测所有车牌区域
    plate_regions = detect_all_plates(img)
    
    if not plate_regions:
        return '未检测到车牌'
    
    # 2. 选择目标车牌
    target_plate = select_target_plate(plate_regions, img)
    
    if not target_plate:
        return '未找到有效车牌区域'
    
    # 3. 提取目标车牌区域
    img_plate = extract_plate_region(img, target_plate)
    
    # 4. 对目标车牌进行识别
    return recognize_normal_plate(img_plate)