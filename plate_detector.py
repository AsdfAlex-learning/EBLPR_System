"""
车牌检测模块
实现车牌区域的定位、检测和提取功能
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import math

# ==========================================
# [配置参数] 倾斜校正相关
# ==========================================
SKEW_CONFIG = {
    'resize_max_dim': 600,         # 缩小处理以加快速度
    'scan_angle_range': 45,        # 扫描范围 +/- 45 度
    'scan_step': 1.0,              # 粗扫步长
    'refine_step': 0.1,            # 精扫步长
    'skew_threshold': 3.0,         # 触发旋转的最小角度阈值
    'initial_search_step': 2.0,    # 初始搜索步长
    'min_search_step': 0.5,        # 最小搜索步长
    'max_iterations': 20,          # 最大搜索迭代次数
}

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """旋转图像，背景填充白色"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated

def detect_skew_angle_radon(image: np.ndarray) -> float:
    """
    使用 Radon 变换思想（投影方差最大化）检测倾斜角。
    """
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > SKEW_CONFIG['resize_max_dim']:
        scale = SKEW_CONFIG['resize_max_dim'] / max(h, w)
        img_small = cv2.resize(image, None, fx=scale, fy=scale)
    else:
        img_small = image.copy()
        
    # 边缘检测预处理
    img_edges = cv2.Canny(img_small, 100, 200, apertureSize=3)
    
    best_angle = 0.0
    max_variance = -1.0
    
    # 1. 粗略扫描
    angles = np.arange(-SKEW_CONFIG['scan_angle_range'], SKEW_CONFIG['scan_angle_range'], SKEW_CONFIG['scan_step'])
    center = (img_small.shape[1] // 2, img_small.shape[0] // 2)
    
    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_edges, M, (img_small.shape[1], img_small.shape[0]), flags=cv2.INTER_NEAREST)
        proj = np.sum(rotated, axis=1)
        var = np.var(proj)
        
        if var > max_variance:
            max_variance = var
            best_angle = angle
            
    # 2. 精细扫描
    fine_angles = np.arange(best_angle - 2.0, best_angle + 2.0, SKEW_CONFIG['refine_step'])
    for angle in fine_angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_edges, M, (img_small.shape[1], img_small.shape[0]), flags=cv2.INTER_NEAREST)
        proj = np.sum(rotated, axis=1)
        var = np.var(proj)
        
        if var > max_variance:
            max_variance = var
            best_angle = angle
            
    return best_angle

def evaluate_plate_confidence(plate_img: np.ndarray) -> float:
    """
    评估车牌图像的质量/置信度（基于垂直投影方差）。
    """
    if plate_img is None or plate_img.size == 0:
        return 0.0
        
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img
        
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v_proj = np.sum(binary, axis=0)
    v_proj = v_proj / 255.0
    
    variance = np.var(v_proj)
    
    mean_val = np.mean(v_proj)
    crossings = 0
    for i in range(1, len(v_proj)):
        if (v_proj[i-1] < mean_val and v_proj[i] >= mean_val) or \
           (v_proj[i-1] >= mean_val and v_proj[i] < mean_val):
            crossings += 1
            
    if crossings < 10 or crossings > 30:
        variance *= 0.1
        
    return variance

def _locate_plate_core(image: np.ndarray, debug: bool = False, debug_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    定位车牌区域（增强版 - 双向Sobel + 强宽高比筛选）
    
    Args:
        image: 输入灰度图像
        debug: 是否保存调试图像
        debug_dir: 调试图像保存目录
    
    Returns:
        包含 'bbox' (x, y, w, h) 和 'image' (裁剪出的车牌图像) 的字典
    """
    
    h_orig, w_orig = image.shape
    
    TARGET_WIDTH = 800
    CROP_RATIO_W = 0.20         # 左右各裁剪掉 20%
    CROP_RATIO_H = 0.20         # 上下各裁剪掉 20%
    
    # 形态学参数
    MORPH_KERNEL_WIDTH = 25     # 较小的核，顺着边框连接
    MORPH_KERNEL_HEIGHT = 5
    
    # 筛选参数
    MIN_AR = 1.2                # 双层车牌较方
    MAX_AR = 5.0                # 允许单层车牌
    MIN_AREA = 2000             # 稍微提高阈值
    TARGET_AR = 181.0 / 91.5    # 1.98
    
    # 精修参数
    REFINE_PAD_W = 0.2
    REFINE_PAD_H = 0.5
    
    scale = TARGET_WIDTH / w_orig
    target_height = int(h_orig * scale)
    resized = cv2.resize(image, (TARGET_WIDTH, target_height))
    
    if debug and debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "00_original_resized.png"), resized)

    crop_x = int(TARGET_WIDTH * CROP_RATIO_W)
    crop_y = int(target_height * CROP_RATIO_H)
    crop_w = TARGET_WIDTH - 2 * crop_x
    crop_h = target_height - 2 * crop_y
    
    cropped = resized[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    
    if debug and debug_dir:
        cv2.imwrite(str(debug_dir / "01_center_crop.png"), cropped)

    blurred = cv2.GaussianBlur(cropped, (5, 5), 0)
    
    # Sobel X (垂直边缘)
    sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    
    # Sobel Y (水平边缘) - 捕捉汉字横向笔画
    sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    
    # 混合边缘 (各占 50%)
    combined_sobel = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    
    if debug and debug_dir:
        cv2.imwrite(str(debug_dir / "02_sobel.png"), combined_sobel)

    ret, binary = cv2.threshold(combined_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if debug and debug_dir:
        cv2.imwrite(str(debug_dir / "03_binary.png"), binary)
        
    # 预膨胀：强化弱边缘
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, kernel_dilate, iterations=1)

    kernel_fix = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_WIDTH, MORPH_KERNEL_HEIGHT))
    
    kernel_fix = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_border = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_fix)
    
    if debug and debug_dir:
        cv2.imwrite(str(debug_dir / "04_border_fix.png"), binary_border)

    contours, hierarchy = cv2.findContours(binary_border, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    if debug and debug_dir:
        debug_contours = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
        
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            x_global = x + crop_x
            y_global = y + crop_y
            
            aspect_ratio = w / float(h)
            area = w * h
            
            has_child = hierarchy[i][2] != -1
            
            if debug and debug_dir:
                cv2.rectangle(debug_contours, (x, y), (x+w, y+h), (200, 200, 200), 1)

            keep = True
            
            if not (MIN_AR < aspect_ratio < MAX_AR):
                keep = False
            elif area < MIN_AREA:
                keep = False
                
            if keep:
                cx = x + w / 2
                cy = y + h / 2
                img_cx, img_cy = crop_w / 2, crop_h / 2
                dist = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
                
                candidates.append({
                    'bbox': (x_global, y_global, w, h),
                    'aspect_ratio': aspect_ratio,
                    'area': area,
                    'has_child': has_child,
                    'dist_from_center': dist
                })
                
                if debug and debug_dir:
                    cv2.rectangle(debug_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if debug and debug_dir:
        cv2.imwrite(str(debug_dir / "05_filtered_contours.png"), debug_contours)

    if not candidates:
        if debug:
            print("[Warning] No candidates found. Fallback to center crop.")
        cx, cy = w_orig // 2, h_orig // 2
        w, h = int(w_orig * 0.6), int(h_orig * 0.3)
        x, y = cx - w // 2, cy - h // 2
        return _pack_result(image, x, y, w, h)

    candidates_with_child = [c for c in candidates if c['has_child']]
    pool = candidates_with_child if candidates_with_child else candidates
    
    best_candidate = None
    
    if len(pool) == 1:
        best_candidate = pool[0]
    else:
        # 多候选评分
        best_score = -999.0
        
        for c in pool:
            # 1. 宽高比评分 (权重 0.7)
            ar_diff = abs(c['aspect_ratio'] - TARGET_AR)
            score_ar = 1.0 - min(1.0, ar_diff / 0.5)
            
            # 2. 距离评分 (权重 0.3)
            max_dist_ref = np.sqrt(crop_w**2 + crop_h**2) / 2
            norm_dist = c['dist_from_center'] / (max_dist_ref * 0.6)
            score_center = max(0.0, 1.0 - norm_dist)
            
            total_score = score_ar * 0.7 + score_center * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_candidate = c
                
    if not best_candidate:
         # 理论上不应该到这里，除非 pool 为空
        cx, cy = w_orig // 2, h_orig // 2
        w, h = int(w_orig * 0.6), int(h_orig * 0.3)
        x, y = cx - w // 2, cy - h // 2
        return _pack_result(image, x, y, w, h)

    x_res, y_res, w_res, h_res = best_candidate['bbox']
    
    # 还原到原始图像坐标
    x = int(x_res / scale)
    y = int(y_res / scale)
    w = int(w_res / scale)
    h = int(h_res / scale)
    
    # 稍微扩大一点区域进行精修
    pad_w = int(w * REFINE_PAD_W)
    pad_h = int(h * REFINE_PAD_H)
    crop_x1 = max(0, x - pad_w)
    crop_y1 = max(0, y - pad_h)
    crop_x2 = min(w_orig, x + w + pad_w)
    crop_y2 = min(h_orig, y + h + pad_h)
    
    rough_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    
    if debug and debug_dir:
        cv2.imwrite(str(debug_dir / "07_rough_crop.png"), rough_crop)

    # 保留原有的精修逻辑
    roi_sobel = cv2.Sobel(rough_crop, cv2.CV_8U, 1, 0, ksize=3)
    _, roi_edges = cv2.threshold(roi_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if debug and debug_dir:
        cv2.imwrite(str(debug_dir / "08_roi_edges.png"), roi_edges)

    h_crop, w_crop = rough_crop.shape
    
    # 水平投影
    h_proj = np.sum(roi_edges, axis=1)
    max_val = np.max(h_proj)
    thresh_h = max_val * 0.2
    
    center_y = h_crop // 2
    top_limit = 0
    bottom_limit = h_crop
    
    for i in range(center_y, 0, -1):
        if h_proj[i] < thresh_h:
            if i > 2 and np.mean(h_proj[i-2:i]) < thresh_h:
                top_limit = i
                break
                
    for i in range(center_y, h_crop):
        if h_proj[i] < thresh_h:
            if i < h_crop - 2 and np.mean(h_proj[i:i+2]) < thresh_h:
                bottom_limit = i
                break
                
    # 垂直投影
    roi_mid = roi_edges[top_limit:bottom_limit, :]
    if roi_mid.shape[0] > 0:
        v_proj = np.sum(roi_mid, axis=0)
        max_v = np.max(v_proj)
        thresh_v = max_v * 0.15
        
        center_x = w_crop // 2
        left_limit = 0
        right_limit = w_crop
        
        gap_tolerance = int(w_crop * 0.10) 
        
        zero_count = 0
        for i in range(center_x, 0, -1):
            if v_proj[i] < thresh_v:
                zero_count += 1
                if zero_count > gap_tolerance:
                    left_limit = i + zero_count
                    break
            else:
                zero_count = 0
        
        zero_count = 0
        for i in range(center_x, w_crop):
            if v_proj[i] < thresh_v:
                zero_count += 1
                if zero_count > gap_tolerance:
                    right_limit = i - zero_count
                    break
            else:
                zero_count = 0
                
        refined_x = crop_x1 + left_limit
        refined_y = crop_y1 + top_limit
        refined_w = right_limit - left_limit
        refined_h = bottom_limit - top_limit
        
        # 验证精修结果
        rough_ar = w / h
        min_h_ratio = 0.6
        if rough_ar < 2.5: 
             min_h_ratio = 0.3
        
        if refined_w > w * 0.5 and refined_h > h * min_h_ratio:
            pad = 5
            final_x = max(0, refined_x - pad)
            final_y = max(0, refined_y - pad)
            final_w = min(w_orig - final_x, refined_w + 2*pad)
            final_h = min(h_orig - final_y, refined_h + 2*pad)
            
            return _pack_result(image, final_x, final_y, final_w, final_h)
        else:
            if debug:
                print(f"[Info] Refinement rejected: w={refined_w}, h={refined_h}")
            
    if debug:
        print("[Info] Refinement failed, using rough crop.")
        
    return _pack_result(image, x, y, w, h)

def locate_plate_region(image: np.ndarray, debug: bool = False, debug_dir: Optional[Path] = None, recognizer: Optional[Any] = None) -> Dict[str, Any]:
    """
    车牌定位入口函数（支持自动倾斜校正）
    
    Args:
        image: 输入灰度图像
        debug: 是否保存调试图像
        debug_dir: 调试目录
        recognizer: 字符识别器实例（用于辅助倾斜校正裁判）。如果不传，则只进行基础定位。
        
    Returns:
        Dict 包含:
        - bbox: (x, y, w, h) 在 rotated_image_full 上的坐标
        - image: 裁剪出的车牌图像
        - rotation_angle: 旋转角度
        - rotated_image_full: 旋转后的完整图像（如果未旋转则为原图）
    """
    
    # 1. 基础定位 (0度)
    res_0 = _locate_plate_core(image, debug, debug_dir)
    
    # 默认结果封装
    res_0['rotation_angle'] = 0.0
    res_0['rotated_image_full'] = image
    
    # 如果没有识别器，或者不需要高级校正，直接返回
    if recognizer is None:
        return res_0
        
    # ==========================================
    # [高级倾斜校正流程]
    # ==========================================
    
    # 2. 检测倾斜角
    detected_angle = detect_skew_angle_radon(image)
    
    if abs(detected_angle) < SKEW_CONFIG['skew_threshold']:
        return res_0
        
    # 3. 爬山法搜索最佳角度
    current_step = SKEW_CONFIG['initial_search_step']
    min_step = SKEW_CONFIG['min_search_step']
    current_angle = detected_angle
    
    visited_scores = {}
    best_result_global = None
    best_score_global = -1.0
    best_angle_global = 0.0
    
    iteration = 0
    max_iterations = SKEW_CONFIG['max_iterations']
    
    while current_step >= min_step and iteration < max_iterations:
        iteration += 1
        candidates_angles = [current_angle, current_angle - current_step, current_angle + current_step]
        
        neighborhood_best_score = -1.0
        neighborhood_best_angle = None
        
        for ang in candidates_angles:
            ang = round(ang, 1)
            
            if ang in visited_scores:
                score = visited_scores[ang]
            else:
                if abs(ang) < 0.1:
                    rot_gray = image
                else:
                    rot_gray = rotate_image(image, ang)
                    
                res = _locate_plate_core(rot_gray)
                
                score = 0.0
                plate_crop = res.get('image')
                if plate_crop is not None:
                    score = evaluate_plate_confidence(plate_crop)
                    
                visited_scores[ang] = score
                
                if score > best_score_global:
                    best_score_global = score
                    best_result_global = res
                    best_angle_global = ang
                    best_result_global['rotation_angle'] = ang
                    best_result_global['rotated_image_full'] = rot_gray # 暂时只存 gray
            
            if score > neighborhood_best_score:
                neighborhood_best_score = score
                neighborhood_best_angle = ang
                
        if neighborhood_best_angle is not None:
            if neighborhood_best_angle == current_angle:
                current_step = current_step / 2.0
            else:
                current_angle = neighborhood_best_angle

    # 4. 最终裁判：基于字符识别结果对比
    
    # (A) 0度得分
    score_rec_0 = 0.0
    text_0 = ""
    if res_0.get('image') is not None:
        # 使用 return_confidence=True 需要 CharacterRecognizer 支持
        # 这里假设 recognizer 已经更新
        try:
            text_0, score_rec_0 = recognizer.recognize(res_0['image'], return_confidence=True)
        except:
             text_0 = recognizer.recognize(res_0['image'])
             score_rec_0 = 0.0 # 无法获取置信度时回退
             
    # (B) 最佳旋转角度得分
    score_rec_best = 0.0
    text_best = ""
    
    if abs(best_angle_global) < 0.1:
        # 最佳就是0度
        return res_0
        
    if best_result_global is None:
        return res_0
        
    if best_result_global.get('image') is not None:
        try:
            text_best, score_rec_best = recognizer.recognize(best_result_global['image'], return_confidence=True)
        except:
            text_best = recognizer.recognize(best_result_global['image'])
            score_rec_best = 0.0

    if debug:
        print(f"[SkewCorrection] 0°: '{text_0}' ({score_rec_0:.4f}) vs {best_angle_global}°: '{text_best}' ({score_rec_best:.4f})")

    # (C) 决策
    # 只有当旋转后的识别分数 显著高于 原图时 (或者原图很烂)，才采用旋转
    confirmed = False
    if score_rec_best > score_rec_0 + 0.05:
        confirmed = True
    elif "?" in text_0 and "?" not in text_best and score_rec_best > 0.2:
        confirmed = True
        
    if confirmed:
        # 重新生成一份高质量的旋转图 (如果之前只存了gray)
        # 实际上我们在 loop 里存的是 rot_gray。如果外部需要 color，外部自己旋转?
        # 或者我们在这里返回 rotated_image_full 就是 gray 也没关系，因为 bbox 主要是为了定位
        # 但 run.py 需要 color image 来画图。
        # 这里 image 传入的是 gray (根据 type hint 和 usage)。
        # wait, run.py 传入的是 gray。
        # run.py: plate_result = locate_plate_region(gray, debug=False)
        # 所以 image 是 gray。
        # run.py 也有 color image 变量。
        
        return best_result_global
    else:
        return res_0

def _pack_result(image, x, y, w, h):
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    
    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > image.shape[1]: w = image.shape[1] - x
    if y + h > image.shape[0]: h = image.shape[0] - y
    
    plate_img = image[y:y+h, x:x+w]
    
    return {
        'bbox': (x, y, w, h),
        'image': plate_img
    }

def detect_plate_count(image: np.ndarray) -> int:
    return 1

def detect_all_plates(image: np.ndarray) -> List[Dict[str, Any]]:
    res = locate_plate_region(image)
    return [res]
