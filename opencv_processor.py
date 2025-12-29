"""
OpenCV主处理模块
整合所有图像处理和识别功能，提供统一的处理接口
"""
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import os
from pathlib import Path

from .image_utils import rgb_to_gray, preprocess_image
from .plate_detector import detect_image_type, locate_plate_region
from .character_recognizer import (
    load_char_templates,
    recognize_normal_plate,
    recognize_tilted_plate,
    recognize_interfered_plate,
    recognize_multiple_plates
)

# 确保字符模板已加载
load_char_templates()

def detect_image_type(img: np.ndarray) -> str:
    """
    检测图像类型：正常、倾斜、干扰、多车牌
    对应MATLAB: detect_image_type(img)
    
    Args:
        img: 输入灰度图像
        
    Returns:
        图像类型字符串 ('normal', 'tilted', 'interfered', 'multiple')
    """
    from .image_utils import detect_tilt_angle, detect_interference_level
    from .plate_detector import detect_plate_count
    
    # 检测倾斜角度
    angle = detect_tilt_angle(img)
    
    # 检测干扰程度
    interference_level = detect_interference_level(img)
    
    # 检测车牌数量
    plate_count = detect_plate_count(img)
    
    # 根据特征判断图像类型
    if plate_count > 1:
        return 'multiple'
    elif abs(angle) > 2 and abs(angle) <= 15:
        return 'tilted'
    elif interference_level > 0.3:
        return 'interfered'
    else:
        return 'normal'

def process_image(image_path: str) -> Tuple[str, str]:
    """
    车牌预处理+字符识别主函数
    对应MATLAB: process_image(imagePath)
    
    Args:
        image_path: 原始图像文件路径
        
    Returns:
        (plate_path, plate_number)
        plate_path: 裁剪后的车牌图像文件路径
        plate_number: 识别出的车牌号码
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像文件: {image_path}")
        
        # 转换为灰度图
        img_gray = rgb_to_gray(img)
        
        # 预处理图像
        img_processed = preprocess_image(img_gray)
        
        # 定位车牌区域
        plate_region = locate_plate_region(img_processed)
        bbox = plate_region['bbox']
        x1, y1, width, height = bbox
        x2 = x1 + width
        y2 = y1 + height
        
        h, w = img_gray.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # 提取车牌区域
        if x2 > x1 and y2 > y1:
            plate_img = img_gray[y1:y2, x1:x2]
        else:
            plate_img = img_gray
        
        # 保存裁剪后的车牌图像
        folder = os.path.dirname(image_path)
        name = os.path.splitext(os.path.basename(image_path))[0]
        plate_path = os.path.join(folder, f"{name}_plate.png")
        
        cv2.imwrite(plate_path, plate_img)
        print(f"OpenCV 预处理完成，车牌子图保存到: {plate_path}")
        
        # 检测图像类型并选择对应算法
        img_type = detect_image_type(img_processed)
        print(f"检测到图像类型: {img_type}")
        
        # 根据图像类型选择识别算法
        if img_type == 'normal':
            plate_number = recognize_normal_plate(img_processed)
        elif img_type == 'tilted':
            plate_number = recognize_tilted_plate(img_processed)
        elif img_type == 'interfered':
            plate_number = recognize_interfered_plate(img_processed)
        elif img_type == 'multiple':
            plate_number = recognize_multiple_plates(img_processed)
        else:
            plate_number = '识别失败：未知图像类型'
        
        print(f"OpenCV 字符识别完成，车牌号码：{plate_number}")
        
        return plate_path, plate_number
        
    except Exception as e:
        print(f"处理图像时出错: {e}")
        raise

def process_image_array(image: np.ndarray, output_dir: str = None) -> Tuple[Optional[str], str]:
    """
    处理图像数组（直接输入图像数据）
    
    Args:
        image: 输入图像数组 (BGR格式)
        output_dir: 输出目录
        
    Returns:
        (plate_path, plate_number)
        plate_path: 裁剪后的车牌图像文件路径（如果保存成功）
        plate_number: 识别出的车牌号码
    """
    try:
        # 转换为灰度图
        img_gray = rgb_to_gray(image)
        
        # 预处理图像
        img_processed = preprocess_image(img_gray)
        
        # 定位车牌区域
        plate_region = locate_plate_region(img_processed)
        bbox = plate_region['bbox']
        x1, y1, width, height = bbox
        x2 = x1 + width
        y2 = y1 + height
        
        h, w = img_gray.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # 提取车牌区域
        if x2 > x1 and y2 > y1:
            plate_img = img_gray[y1:y2, x1:x2]
        else:
            plate_img = img_gray
        
        # 保存裁剪后的车牌图像（如果指定了输出目录）
        plate_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plate_path = os.path.join(output_dir, f"plate_{np.random.randint(10000)}.png")
            cv2.imwrite(plate_path, plate_img)
            print(f"OpenCV 预处理完成，车牌子图保存到: {plate_path}")
        
        # 检测图像类型并选择对应算法
        img_type = detect_image_type(img_processed)
        print(f"检测到图像类型: {img_type}")
        
        # 根据图像类型选择识别算法
        if img_type == 'normal':
            plate_number = recognize_normal_plate(img_processed)
        elif img_type == 'tilted':
            plate_number = recognize_tilted_plate(img_processed)
        elif img_type == 'interfered':
            plate_number = recognize_interfered_plate(img_processed)
        elif img_type == 'multiple':
            plate_number = recognize_multiple_plates(img_processed)
        else:
            plate_number = '识别失败：未知图像类型'
        
        print(f"OpenCV 字符识别完成，车牌号码：{plate_number}")
        
        return plate_path, plate_number
        
    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None, f"识别失败：{str(e)}"