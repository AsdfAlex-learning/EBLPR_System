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
    电动车牌字符分割专属函数（基于模板比例定位的精准分割）
    核心逻辑：模板比例映射 + 相对位置计算，解决边框截取不精准问题
    
    Args:
        plate_region: 包含车牌区域信息的字典
        debug: 是否打印调试信息
        
    Returns:
        字符列表，每个字符是一个字典 {
            'image': 字符图像,
            'position': 位置信息
        }
    """
    
    # ========== 常量定义（电动车牌标准化参数） ==========
    # 电动车牌字符数量：8个字符（汉字2 + 字母1 + 数字5）
    EXPECTED_CHAR_COUNT = 8
    
    # 模板参考尺寸（基于1mm=2像素映射的标准化尺寸）
    TEMPLATE_WIDTH = 183    # 91.5mm × 2 = 183像素
    TEMPLATE_HEIGHT = 362   # 181mm × 2 = 362像素
    
    # 电动车牌字符模板坐标（基于相对比例重新设计）
    CHAR_TEMPLATES = [
        # 汉字区域（"广""州"）- 2个字符，水平间距为字母数字间距的3倍
        # 汉字中心点与字母数字区域中心点垂直对齐
        {'name': '汉字1（广）', 'x': 50, 'y': 150, 'w': 50, 'h': 70},  # 字符1
        {'name': '汉字2（州）', 'x': 200, 'y': 150, 'w': 50, 'h': 70},  # 字符2
        # 字母数字区域（P53283）- 6个字符，水平间距为图像宽度的3%-8%
        {'name': '字母1（P）', 'x': 50, 'y': 150, 'w': 40, 'h': 60},   # 字符3
        {'name': '数字1（5）', 'x': 100, 'y': 150, 'w': 40, 'h': 60},  # 字符4
        {'name': '数字2（3）', 'x': 150, 'y': 150, 'w': 40, 'h': 60},  # 字符5
        {'name': '数字3（2）', 'x': 200, 'y': 150, 'w': 40, 'h': 60},  # 字符6
        {'name': '数字4（8）', 'x': 250, 'y': 150, 'w': 40, 'h': 60},  # 字符7
        {'name': '数字5（3）', 'x': 300, 'y': 150, 'w': 40, 'h': 60}   # 字符8
    ]
    
    # ========== 子函数：图像预处理 ==========
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """
        图像预处理：灰度转换 → 自适应二值化 → 形态学闭运算
        目标：字符为白色，背景为黑色，去除噪点，补字符缝隙
        """
        # 1. 转灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 2. 自适应二值化（字符为白色，背景黑色）
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # 3. 形态学闭运算（kernel=(2,2)）去噪、补字符缝隙
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary_closed
    
    # ========== 子函数：计算比例缩放因子 ==========
    def calculate_scaling_factors(plate_width: int, plate_height: int) -> tuple:
        """
        计算实际车牌图像与模板之间的缩放比例
        返回：(scale_x, scale_y) - 水平和垂直方向的缩放因子
        """
        # 基于模板的整体尺寸和实际车牌尺寸计算缩放因子
        # 确保所有字符都能在图像范围内
        
        # 模板的整体尺寸（基于字符位置和尺寸计算）
        template_max_x = max(template['x'] + template['w'] for template in CHAR_TEMPLATES)
        template_max_y = max(template['y'] + template['h'] for template in CHAR_TEMPLATES)
        
        # 计算水平和垂直方向的缩放因子
        # 确保缩放后的字符不会超出图像边界
        scale_x = plate_width / template_max_x * 0.9  # 留10%的边距
        scale_y = plate_height / template_max_y * 0.9  # 留10%的边距
        
        # 使用较小的缩放因子，确保所有字符都能在图像内
        scale_factor = min(scale_x, scale_y)
        
        if debug:
            print(f"模板最大边界: x={template_max_x}, y={template_max_y}")
            print(f"车牌图像尺寸: {plate_width}x{plate_height}")
            print(f"计算缩放因子: scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")
            print(f"最终缩放因子: {scale_factor:.3f}")
        
        return scale_factor, scale_factor
    
    # ========== 主函数逻辑 ==========
    plate_image = plate_region['image']
    h, w = plate_image.shape
    
    if debug:
        print(f"输入车牌图像尺寸: {h}x{w}")
        print(f"像素范围: {plate_image.min()}-{plate_image.max()}")
    
    # 步骤1：图像预处理
    plate_binary = preprocess_image(plate_image)
    
    if debug:
        print(f"二值化后像素范围: {plate_binary.min()}-{plate_binary.max()}")
    
    # 步骤2：计算比例缩放因子
    scale_x, scale_y = calculate_scaling_factors(w, h)
    
    # 步骤3：基于模板比例的精确定位（核心逻辑）
    characters = []
    
    for i, template in enumerate(CHAR_TEMPLATES):
        # 3.1 根据缩放因子计算实际字符位置和尺寸
        real_x = int(template['x'] * scale_x)
        real_y = int(template['y'] * scale_y)
        real_w = int(template['w'] * scale_x)
        real_h = int(template['h'] * scale_y)
        
        # 3.2 确保字符尺寸在合理范围内
        # 字符宽度：车牌宽度的1/8到1/12
        min_char_width = w // 12
        max_char_width = w // 8
        real_w = max(min_char_width, min(real_w, max_char_width))
        
        # 字符高度：车牌高度的1/3到1/2
        min_char_height = h // 3
        max_char_height = h // 2
        real_h = max(min_char_height, min(real_h, max_char_height))
        
        # 3.3 处理坐标越界
        real_x = max(0, real_x)
        real_y = max(0, real_y)
        
        # 确保宽度和高度不超出图像边界
        if real_x + real_w > w:
            real_w = w - real_x
        if real_y + real_h > h:
            real_h = h - real_y
        
        # 3.4 提取字符区域（从原始车牌图）
        if real_w > 0 and real_h > 0:
            char_region = plate_image[real_y:real_y+real_h, real_x:real_x+real_w]
            
            # 调整到标准尺寸（20×40像素）
            if char_region.size > 0:
                char_img = cv2.resize(char_region, (20, 40), interpolation=cv2.INTER_CUBIC)
            else:
                char_img = np.zeros((40, 20), dtype=np.uint8)
            
            characters.append({
                'image': char_img,
                'position': (real_x, real_y, real_w, real_h)
            })
            
            if debug:
                print(f"  字符{i+1}({template['name']}): ")
                print(f"    模板位置: ({template['x']}, {template['y']}, {template['w']}, {template['h']})")
                print(f"    实际位置: ({real_x}, {real_y}, {real_w}, {real_h})")
        else:
            if debug:
                print(f"  字符{i+1}({template['name']}): 坐标越界，跳过")
    
    # 步骤4：验证分割结果
    if len(characters) < EXPECTED_CHAR_COUNT:
        if debug:
            print(f"警告: 只分割出 {len(characters)} 个字符，期望 {EXPECTED_CHAR_COUNT} 个")
    
    if debug:
        print(f"成功分割出 {len(characters)} 个字符")
        
        # 计算字符间距分布
        if len(characters) > 1:
            spacings = []
            for i in range(1, len(characters)):
                prev_char_end = characters[i-1]['position'][0] + characters[i-1]['position'][2]
                curr_char_start = characters[i]['position'][0]
                spacing = curr_char_start - prev_char_end
                spacings.append(spacing)
            
            if spacings:
                avg_spacing = sum(spacings) / len(spacings)
                std_spacing = (sum((s - avg_spacing) ** 2 for s in spacings) / len(spacings)) ** 0.5
                print(f"平均字符间距: {avg_spacing:.1f} 像素")
                print(f"间距标准差: {std_spacing:.1f} 像素")
                if std_spacing < 10:
                    print("✓ 字符分布均匀")
                else:
                    print("⚠ 字符分布不均匀")
    
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