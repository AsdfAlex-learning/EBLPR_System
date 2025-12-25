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
        # 修正路径：使用正确的相对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(current_dir, "..", "opencv_core", "char_templates")
        template_dir = os.path.abspath(template_dir)
    
    print(f"尝试加载模板目录: {template_dir}")
    
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
    loaded_count = 0
    for label in labels:
        template_path = os.path.join(template_dir, f"{label}.png")
        
        # 检查文件是否存在
        if os.path.exists(template_path):
            # 使用Python文件操作读取图像，避免OpenCV中文文件名问题
            try:
                with open(template_path, 'rb') as f:
                    file_bytes = np.frombuffer(f.read(), np.uint8)
                    template = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            except Exception as e:
                print(f"读取模板文件 {label} 时出错: {e}")
                template = None
        else:
            print(f"模板文件不存在: {template_path}")
            template = None
        
        if template is not None:
            # 二值化
            _, template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 统一尺寸为40x20
            template = cv2.resize(template, (20, 40), interpolation=cv2.INTER_NEAREST)
            
            _char_templates.append(template)
            _char_labels.append(label)
            loaded_count += 1
            print(f"  已加载模板: {label}")
        else:
            print(f"无法读取模板文件: {template_path}")
            
    if not _char_templates:
        print("未加载到任何字符模板")
        return False
    
    print(f"成功加载 {len(_char_templates)} 个字符模板")
    
    # 检查是否有汉字模板
    chinese_chars = [label for label in _char_labels if len(label) == 1 and '\u4e00' <= label <= '\u9fff']
    print(f"汉字模板数量: {len(chinese_chars)}")
    print(f"汉字模板标签: {chinese_chars}")
    
    return True

def segment_characters(plate_region: Dict[str, Any], debug: bool = False) -> List[Dict[str, Any]]:
    """
    分割车牌字符（改进版本，专门处理小尺寸车牌图像）
    对应MATLAB: segment_characters(plate_region)
    
    Args:
        plate_region: 包含车牌区域信息的字典
        debug: 是否打印调试信息
        
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
    
    if debug:
        print(f"车牌区域尺寸: {h}x{w}")
    
    # 改进字符分割方法 - 专门处理小尺寸车牌
    # 1. 图像预处理：增强对比度
    # 使用直方图均衡化增强对比度
    plate_enhanced = cv2.equalizeHist(plate_image)
    
    # 2. 自适应阈值二值化（更适合小尺寸图像）
    plate_binary = cv2.adaptiveThreshold(plate_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 5, 2)
    
    # 3. 反转二值化结果，使字符为白色，背景为黑色
    plate_binary = cv2.bitwise_not(plate_binary)
    
    # 4. 形态学操作去除小噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    plate_binary = cv2.morphologyEx(plate_binary, cv2.MORPH_OPEN, kernel)
    
    # 5. 垂直投影法精确定位字符边界
    vertical_proj = np.sum(plate_binary, axis=0)
    
    # 对于小尺寸图像，使用更小的平滑核
    kernel_size = max(1, w // 15)  # 动态调整核大小
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    proj_smoothed = np.convolve(vertical_proj, kernel, mode='same')
    
    # 计算差分找峰值（字符边界）
    diff_proj = np.diff(proj_smoothed)
    
    # 找到字符边界（从负变正的位置是字符开始，从正变负是字符结束）
    char_starts = np.where((diff_proj[:-1] < 0) & (diff_proj[1:] > 0))[0] + 1
    char_ends = np.where((diff_proj[:-1] > 0) & (diff_proj[1:] < 0))[0] + 1
    
    # 确保字符边界数量匹配
    if len(char_starts) > len(char_ends):
        char_starts = char_starts[:len(char_ends)]
    elif len(char_ends) > len(char_starts):
        char_ends = char_ends[:len(char_starts)]
    
    characters = []
    
    # 对于小尺寸车牌，使用更宽松的字符数量判断
    if len(char_starts) >= 5:  # 至少找到5个字符边界
        # 取前7个字符（如果不足7个，则取所有）
        char_starts = char_starts[:min(7, len(char_starts))]
        char_ends = char_ends[:min(7, len(char_ends))]
        
        for i, (start, end) in enumerate(zip(char_starts, char_ends)):
            # 提取字符区域（稍微扩大边界）
            char_width = end - start
            expand = max(1, char_width // 4)  # 扩大25%的宽度
            
            start_col = max(0, start - expand)
            end_col = min(w, end + expand)
            
            # 提取字符区域
            char_region = plate_binary[:, start_col:end_col]
            
            # 如果字符区域太小，使用固定宽度
            if char_region.shape[1] < 3:
                char_width = max(3, w // 8)
                start_col = max(0, start - char_width // 2)
                end_col = min(w, start + char_width // 2)
                char_region = plate_binary[:, start_col:end_col]
            
            # 调整字符大小到标准尺寸
            if char_region.size > 0:
                # 对于小尺寸图像，使用更精细的插值
                char_img = cv2.resize(char_region, (20, 40), interpolation=cv2.INTER_CUBIC)
            else:
                # 创建空白图像
                char_img = np.zeros((40, 20), dtype=np.uint8)
            
            characters.append({
                'image': char_img,
                'position': (start_col, 0, end_col - start_col, h)
            })
    else:
        # 如果垂直投影法失败，使用固定分割
        if debug:
            print("垂直投影法失败，使用固定分割")
        
        char_width = max(3, w // 7)  # 确保最小宽度
        for i in range(7):
            start_col = i * char_width
            end_col = min((i + 1) * char_width, w)
            
            # 提取字符区域
            char_region = plate_binary[:, start_col:end_col]
            
            # 调整字符大小
            if char_region.size > 0:
                char_img = cv2.resize(char_region, (20, 40), interpolation=cv2.INTER_CUBIC)
            else:
                char_img = np.zeros((40, 20), dtype=np.uint8)
            
            characters.append({
                'image': char_img,
                'position': (start_col, 0, end_col - start_col, h)
            })
    
    if debug:
        print(f"成功分割出 {len(characters)} 个字符")
        for i, char_info in enumerate(characters):
            print(f"  字符{i}: 位置 {char_info['position']}, 尺寸 {char_info['image'].shape}")
    
    return characters

def match_character(char_img: np.ndarray, debug: bool = False) -> str:
    """
    匹配单个字符（模板匹配）
    对应MATLAB: match_character(char_img)
    
    Args:
        char_img: 字符图像
        debug: 是否打印调试信息
        
    Returns:
        匹配的字符标签，匹配失败返回 '?'
    """
    global _char_templates, _char_labels
    
    # 确保模板已加载（只加载一次）
    if not _char_templates:
        if not load_char_templates():
            if debug:
                print("模板未加载")
            return '?'
    
    # 检查字符图像
    if char_img is None or char_img.size == 0:
        if debug:
            print("字符图像为空")
        return '?'
    
    # 确保字符图像是二值图像
    if char_img.max() > 0:
        if char_img.dtype != np.uint8:
            char_img = np.uint8(char_img)
    else:
        if debug:
            print("字符图像全黑")
        return '?'
    
    # 改进字符预处理
    # 1. 确保字符图像是二值图像
    if char_img.max() > 1:  # 如果是灰度图像
        _, char_img_bin = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        char_img_bin = char_img
    
    # 2. 膨胀操作增强字符
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    char_img_dilated = cv2.dilate(char_img_bin, kernel, iterations=1)
    
    # 3. 去除孤立小噪点
    char_img_clean = cv2.morphologyEx(char_img_dilated, cv2.MORPH_OPEN, 
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
    
    # 4. 形态学闭运算，修复字符断裂
    char_img_close = cv2.morphologyEx(char_img_clean, cv2.MORPH_CLOSE, 
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
    
    # 5. 强制统一尺寸（与模板严格对齐）
    char_img_final = cv2.resize(char_img_close, (20, 40), interpolation=cv2.INTER_CUBIC)
    
    # 6. 最终二值化
    _, char_img_final = cv2.threshold(char_img_final, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 计算与每个模板的相似度
    max_similarity = 0
    best_label = '?'
    best_similarity = 0
    
    for i, template in enumerate(_char_templates):
        # 确保模板是二值图像
        if template.dtype != np.uint8:
            template = np.uint8(template)
        
        _, template_bin = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY)
        
        # 改进相似度计算：使用多种相似度指标
        # 1. 计算重叠像素比例（Jaccard相似系数）
        intersection = np.logical_and(char_img_final, template_bin)
        union = np.logical_or(char_img_final, template_bin)
        
        if np.sum(union) > 0:
            jaccard_similarity = np.sum(intersection) / np.sum(union)
        else:
            jaccard_similarity = 0
        
        # 2. 计算像素匹配度（相同像素比例）
        same_pixels = np.sum(char_img_final == template_bin)
        pixel_similarity = same_pixels / char_img_final.size
        
        # 3. 计算结构相似度（考虑字符结构）
        # 计算字符的重心位置
        char_moments = cv2.moments(char_img_final)
        template_moments = cv2.moments(template_bin)
        
        if char_moments['m00'] > 0 and template_moments['m00'] > 0:
            char_cx = char_moments['m10'] / char_moments['m00']
            char_cy = char_moments['m01'] / char_moments['m00']
            template_cx = template_moments['m10'] / template_moments['m00']
            template_cy = template_moments['m01'] / template_moments['m00']
            
            # 重心位置差异（归一化）
            centroid_diff = np.sqrt((char_cx - template_cx)**2 + (char_cy - template_cy)**2)
            centroid_similarity = 1 - min(1.0, centroid_diff / 10)  # 最大允许10像素差异
        else:
            centroid_similarity = 0
        
        # 综合相似度（加权平均）
        similarity = 0.5 * jaccard_similarity + 0.3 * pixel_similarity + 0.2 * centroid_similarity
        
        # 记录最佳匹配
        if similarity > max_similarity:
            max_similarity = similarity
            best_label = _char_labels[i]
            best_similarity = similarity
    
    # 即使相似度较低，也返回最佳匹配
    if debug:
        print(f"最佳匹配: {best_label}, 相似度: {best_similarity:.3f}")
    
    # 调整匹配阈值到0.5，使用综合相似度
    if best_similarity < 0.5:
        return '?'
    
    return best_label

def recognize_characters(characters: List[Dict[str, Any]], debug: bool = False) -> str:
    """
    识别所有字符
    对应MATLAB: recognize_characters(characters)
    
    Args:
        characters: 字符列表
        debug: 是否打印调试信息
        
    Returns:
        识别结果字符串
    """
    if not characters:
        if debug:
            print("没有字符可识别")
        return ""
    
    result_text = ""
    
    for i, char_info in enumerate(characters):
        if debug:
            print(f"\n识别字符 {i}:")
        
        char_img = char_info['image']
        recognized_char = match_character(char_img, debug=debug)
        result_text += recognized_char
    
    return result_text

from .image_utils import preprocess_image
from .plate_detector import locate_plate_region

def recognize_normal_plate(image_input, debug: bool = False) -> str:
    """
    识别正常车牌
    对应MATLAB: recognize_normal_plate(image_path)
    
    Args:
        image_input: 图像路径或numpy数组
        debug: 是否打印调试信息
        
    Returns:
        识别结果
    """
    # 读取图像（如果输入是路径）
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            if debug:
                print(f"无法读取图像: {image_input}")
            return ""
    else:
        # 输入已经是numpy数组
        image = image_input
    
    # 1. 图像预处理
    preprocessed = preprocess_image(image)
    
    # 2. 定位车牌区域
    plate_region = locate_plate_region(preprocessed)
    
    # 3. 分割字符
    characters = segment_characters(plate_region, debug=debug)
    
    # 4. 识别字符
    result = recognize_characters(characters, debug=debug)
    
    if debug:
        print(f"normal_plate识别结果: {result}")
    
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