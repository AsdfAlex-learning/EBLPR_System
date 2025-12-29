"""
车牌检测模块
实现车牌区域的定位、检测和提取功能
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path

def locate_plate_region(image: np.ndarray, debug: bool = False, debug_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    定位车牌区域（增强版 - 双向Sobel + 强宽高比筛选）
    策略：Center Crop -> Dual Sobel -> Pre-dilate -> Small Kernel Close -> RETR_TREE -> High Weight AR Score -> Projection Refine
    
    Args:
        image: 输入灰度图像
        debug: 是否保存调试图像
        debug_dir: 调试图像保存目录
    
    Returns:
        包含 'bbox' (x, y, w, h) 和 'image' (裁剪出的车牌图像) 的字典
    """
    h_orig, w_orig = image.shape
    
    # ==========================================
    # 参数设置 (源自 debug_plate_detector.py)
    # ==========================================
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
    
    # ==========================================
    # Step 0: 归一化
    # ==========================================
    scale = TARGET_WIDTH / w_orig
    target_height = int(h_orig * scale)
    resized = cv2.resize(image, (TARGET_WIDTH, target_height))
    
    if debug and debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "00_original_resized.png"), resized)

    # ==========================================
    # Step 1: 中心裁剪
    # ==========================================
    crop_x = int(TARGET_WIDTH * CROP_RATIO_W)
    crop_y = int(target_height * CROP_RATIO_H)
    crop_w = TARGET_WIDTH - 2 * crop_x
    crop_h = target_height - 2 * crop_y
    
    cropped = resized[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    
    if debug and debug_dir:
        cv2.imwrite(str(debug_dir / "01_center_crop.png"), cropped)

    # ==========================================
    # Step 2: 边缘检测 (双向 Sobel)
    # ==========================================
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

    # ==========================================
    # Step 3: 二值化 & 预膨胀
    # ==========================================
    ret, binary = cv2.threshold(combined_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if debug and debug_dir:
        cv2.imwrite(str(debug_dir / "03_binary.png"), binary)
        
    # 预膨胀：强化弱边缘
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, kernel_dilate, iterations=1)

    # ==========================================
    # Step 4: 边框修复 (形态学闭运算)
    # ==========================================
    # 使用较小的核连接边框断点
    kernel_fix = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_WIDTH, MORPH_KERNEL_HEIGHT))
    # 注意：这里 debug 脚本用的是 kernel_fix=3x3, 但参数区定义了 25x5。
    # 用户说 "debug已经足够好了"，debug脚本实际代码行170用的是 (3,3)。
    # 但参数区写的是 25, 5。
    # 让我仔细看 debug_plate_detector.py 的第 170 行。
    # "kernel_fix = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))"
    # 原来 debug 脚本里写死了 (3,3)！而没有使用 MORPH_KERNEL_WIDTH/HEIGHT。
    # 既然用户觉得好，我就应该用 debug 脚本实际执行的逻辑，也就是 (3,3)。
    # 可是 (3,3) 真的能连上车牌吗？
    # 再次检查 debug 脚本... 
    # 行46: MORPH_KERNEL_WIDTH = 25
    # 行170: kernel_fix = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 行172: binary_border = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_fix)
    # 这意味着 debug 脚本实际上只用了 3x3 的闭运算。
    # 为什么效果好？因为 Step 05 用了 RETR_TREE 找边框。
    # 既然用户确认效果好，我严格遵照 debug 脚本的代码逻辑 (3,3)。
    
    kernel_fix = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_border = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_fix)
    
    if debug and debug_dir:
        cv2.imwrite(str(debug_dir / "04_border_fix.png"), binary_border)

    # ==========================================
    # Step 5: 轮廓提取与特征筛选
    # ==========================================
    contours, hierarchy = cv2.findContours(binary_border, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    if debug and debug_dir:
        debug_contours = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
        
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # 映射回 resized 全局坐标
            x_global = x + crop_x
            y_global = y + crop_y
            
            aspect_ratio = w / float(h)
            area = w * h
            
            has_child = hierarchy[i][2] != -1
            
            if debug and debug_dir:
                cv2.rectangle(debug_contours, (x, y), (x+w, y+h), (200, 200, 200), 1)

            # 筛选逻辑
            keep = True
            
            # 1. 宽高比
            if not (MIN_AR < aspect_ratio < MAX_AR):
                keep = False
            # 2. 面积
            elif area < MIN_AREA:
                keep = False
                
            if keep:
                # 计算中心位置距离
                cx = x + w / 2
                cy = y + h / 2
                img_cx, img_cy = crop_w / 2, crop_h / 2
                dist = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
                
                candidates.append({
                    'bbox': (x_global, y_global, w, h), # 存储 resized 全局坐标
                    'aspect_ratio': aspect_ratio,
                    'area': area,
                    'has_child': has_child,
                    'dist_from_center': dist
                })
                
                if debug and debug_dir:
                    cv2.rectangle(debug_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if debug and debug_dir:
        cv2.imwrite(str(debug_dir / "05_filtered_contours.png"), debug_contours)

    # ==========================================
    # Step 6: 候选评分与最终决策
    # ==========================================
    # 兜底逻辑：如果没有候选，回退到中心裁剪
    if not candidates:
        if debug:
            print("[Warning] No candidates found. Fallback to center crop.")
        cx, cy = w_orig // 2, h_orig // 2
        w, h = int(w_orig * 0.6), int(h_orig * 0.3)
        x, y = cx - w // 2, cy - h // 2
        return _pack_result(image, x, y, w, h)

    # 优先选择有子轮廓的
    candidates_with_child = [c for c in candidates if c['has_child']]
    pool = candidates_with_child if candidates_with_child else candidates
    
    best_candidate = None
    
    # 单候选直接选中
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

    # ==========================================
    # Step 7: 坐标还原与粗剪裁
    # ==========================================
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

    # ==========================================
    # Step 9: 精细定位 (Projection Refinement)
    # ==========================================
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
