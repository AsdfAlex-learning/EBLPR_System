"""
车牌倾斜校正测试脚本 (Test Skew Correction) - 增强版

算法逻辑：
1. 倾斜检测 (Skew Detection):
   使用投影轮廓法 (Projection Profile) 检测图像的主要倾斜角。
   相比霍夫变换，此方法对文本行更加鲁棒，不易受噪点影响。

2. 迭代校正 (Iterative Correction):
   - 初步检测出倾斜角 theta。
   - 如果 theta 较小 (<3度)，视为未倾斜。
   - 如果检测到倾斜，按用户建议的"约7度"步长或直接尝试目标角度及周边角度。
   - 对每次旋转后的图像进行车牌定位。

3. 结果置信度评估 (Confidence Evaluation):
   - 对定位出的车牌区域进行垂直投影分析。
   - "清晰且正"的车牌，其垂直投影会有明显的波峰波谷（字符与间隙）。
   - 计算投影方差作为置信度指标。

"""

import cv2
import numpy as np
import math
from pathlib import Path
import shutil
from typing import Optional, Tuple, List, Dict

# 引用现有定位函数
from plate_detector import locate_plate_region

# ==========================================
# [CONFIG] 参数配置
# ==========================================
CONFIG = {
    'debug_output_dir': Path("output_test_skew_v2"),
    'resize_max_dim': 600,         # 缩小处理以加快速度
    'scan_angle_range': 45,        # 扫描范围 +/- 45 度
    'scan_step': 1.0,              # 粗扫步长
    'refine_step': 0.1,            # 精扫步长
    'skew_threshold': 3.0,         # 触发旋转的最小角度阈值
    'initial_search_step': 2.0,    # 初始搜索步长 (调小以避免跳过真值)
    'min_search_step': 0.5,        # 最小搜索步长 (自适应精修终止条件)
}

from character_recognizer import CharacterRecognizer

# 初始化识别器
recognizer = CharacterRecognizer()

# ==========================================
# [Core Logic] 核心算法
# ==========================================

def calculate_recognition_confidence(plate_img: np.ndarray) -> float:
    """
    使用字符识别结果作为置信度评分。
    识别出的字符置信度总和越高，说明矫正效果越好。
    """
    if plate_img is None or plate_img.size == 0:
        return 0.0
    
    # 临时调用 recognize 方法获取详细置信度
    # 我们需要修改 recognize 方法或者直接调用其内部逻辑
    # 为了不侵入 CharacterRecognizer 太多，我们这里模拟调用
    
    # 注意：recognizer.recognize 返回的是字符串，我们需要置信度
    # 变通方法：我们可以给 CharacterRecognizer 增加一个返回详细信息的模式
    # 或者直接在这里复用其预处理和分割逻辑（略显重复）
    
    # 最佳方案：修改 CharacterRecognizer.recognize 使其能返回详细数据
    # 但为了不破坏现有接口，我们扩展一个新的方法或使用私有方法
    
    # 这里为了快速验证，我们假设 recognizer 有一个 recognize_with_confidence 方法
    # 既然没有，我们利用现有的 recognize 方法的副作用 —— 它会打印详细信息
    # 但我们需要程序能读到。
    
    # 让我们直接使用 recognizer 的内部逻辑进行快速评估
    # 1. 预处理
    target_h = 140
    h_raw, w_raw = plate_img.shape[:2]
    scale = target_h / h_raw
    target_w = int(w_raw * scale)
    if target_w <= 0: return 0.0
    
    img_resized = cv2.resize(plate_img, (target_w, target_h))
    
    # 转灰度
    if len(img_resized.shape) == 3:
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_resized
        
    # 识别并获取总分
    # 我们临时修改 recognize 接口太麻烦，不如直接实例化一个临时的识别逻辑
    # 或者，我们信任 evaluate_plate_confidence (投影法) 作为初筛
    # 而只在最终结果对比时使用识别分。
    
    # 但用户的意思是：用识别结果来决定是否旋转。
    # 这意味着我们需要对 "原图" 和 "旋转后的图" 都做一次识别，谁分高信谁。
    
    # 让我们在 process_single_image 中实现这个对比逻辑。
    # 这里只保留一个空的占位，实际逻辑写在 process_single_image 中
    return 0.0

def detect_skew_angle_radon(image: np.ndarray, debug_name: str = "") -> float:
    """
    使用 Radon 变换思想（投影方差最大化）检测倾斜角。
    """
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > CONFIG['resize_max_dim']:
        scale = CONFIG['resize_max_dim'] / max(h, w)
        img_small = cv2.resize(image, None, fx=scale, fy=scale)
    else:
        img_small = image.copy()
        
    # 边缘检测预处理
    # 提高阈值，只保留明显的强边缘（车牌边界），过滤文字纹理
    img_edges = cv2.Canny(img_small, 100, 200, apertureSize=3)
    
    best_angle = 0.0
    max_variance = -1.0
    
    # 1. 粗略扫描
    angles = np.arange(-CONFIG['scan_angle_range'], CONFIG['scan_angle_range'], CONFIG['scan_step'])
    
    scores = []
    
    center = (img_small.shape[1] // 2, img_small.shape[0] // 2)
    
    for angle in angles:
        # 旋转图像
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # 只需要旋转边缘图
        rotated = cv2.warpAffine(img_edges, M, (img_small.shape[1], img_small.shape[0]), flags=cv2.INTER_NEAREST)
        
        # 计算水平投影方差
        proj = np.sum(rotated, axis=1)
        var = np.var(proj)
        scores.append(var)
        
        if var > max_variance:
            max_variance = var
            best_angle = angle
            
    # 2. 精细扫描 (在最佳角度附近 +/- 2度)
    fine_angles = np.arange(best_angle - 2.0, best_angle + 2.0, CONFIG['refine_step'])
    for angle in fine_angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_edges, M, (img_small.shape[1], img_small.shape[0]), flags=cv2.INTER_NEAREST)
        proj = np.sum(rotated, axis=1)
        var = np.var(proj)
        
        if var > max_variance:
            max_variance = var
            best_angle = angle
            
    if debug_name:
        print(f"  [SkewDetect] Best angle: {best_angle:.2f} deg (Score: {max_variance:.2f})")
        
    return best_angle

def evaluate_plate_confidence(plate_img: np.ndarray) -> float:
    """
    评估车牌图像的质量/置信度。
    结合投影方差和峰值数量检查。
    """
    if plate_img is None or plate_img.size == 0:
        return 0.0
        
    # 转灰度
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img
        
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 垂直投影 (按列求和)
    v_proj = np.sum(binary, axis=0)
    
    # 归一化
    v_proj = v_proj / 255.0
    
    # 1. 计算方差 (基础分)
    variance = np.var(v_proj)
    
    # 2. 峰值计数 (惩罚项)
    # 计算穿过均值的次数，近似代表字符数量
    mean_val = np.mean(v_proj)
    crossings = 0
    for i in range(1, len(v_proj)):
        if (v_proj[i-1] < mean_val and v_proj[i] >= mean_val) or \
           (v_proj[i-1] >= mean_val and v_proj[i] < mean_val):
            crossings += 1
            
    # 一个字符通常有2个穿过点(上坡下坡)，车牌约7-8字符，加上边框，穿过点应在 10-25 之间
    # 如果穿过点太少（说明只有几条大粗线）或太多（噪点），则惩罚
    if crossings < 10 or crossings > 30:
        variance *= 0.1  # 强惩罚
        
    return variance

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """旋转图像，背景填充白色"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated

def process_single_image(img_path: Path, output_dir: Path):
    print(f"\nProcessing: {img_path.name}")
    
    # 读取图像
    img_np = np.fromfile(str(img_path), dtype=np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if image is None:
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. 检测倾斜角
    detected_angle = detect_skew_angle_radon(gray, img_path.stem)
    
    if abs(detected_angle) < CONFIG['skew_threshold']:
        print(f"  -> Angle too small ({detected_angle:.2f}), skipping rotation.")
        # 仍然测试 0 度
        res = locate_plate_region(gray)
        score = 0
        if res.get('image') is not None:
             score = evaluate_plate_confidence(res['image'])
        print(f"    Angle 0.0° -> Score: {score:.2f}")
        return

    # 2. 爬山法搜索最佳角度 (Hill Climbing Search)
    # 自适应步长策略：
    # 1. 使用大步长快速找到局部峰值
    # 2. 在峰值处缩小步长进行精修，直到步长小于阈值
    
    current_step = CONFIG['initial_search_step']
    min_step = CONFIG['min_search_step']
    current_angle = detected_angle
    
    # 记录已访问的角度及其分数
    visited_scores = {}
    
    best_result_global = None
    best_score_global = -1.0
    best_angle_global = 0.0
    
    vis_images = []
    
    iteration = 0
    max_iterations = 20  # 增加迭代次数上限以支持精修
    
    while current_step >= min_step and iteration < max_iterations:
        iteration += 1
        
        # 定义本次要测试的邻域 (中心 + 左右)
        candidates_angles = [current_angle, current_angle - current_step, current_angle + current_step]
        
        # 这一轮邻域内的最佳分数和角度
        neighborhood_best_score = -1.0
        neighborhood_best_angle = None
        
        for ang in candidates_angles:
            ang = round(ang, 1)
            
            # 如果已经计算过，直接取值
            if ang in visited_scores:
                score = visited_scores[ang]
                # print(f"    [Iter {iteration}] Angle {ang:5.1f}° -> Score: {score:.2f} (Cached)")
            else:
                # 旋转 & 定位
                if abs(ang) < 0.1:
                    rot_gray = gray.copy()
                    rot_color = image.copy()
                else:
                    rot_gray = rotate_image(gray, ang)
                    rot_color = rotate_image(image, ang)
                    
                res = locate_plate_region(rot_gray)
                
                # 评分
                score = 0.0
                plate_crop = res.get('image')
                if plate_crop is not None:
                    score = evaluate_plate_confidence(plate_crop)
                    
                print(f"    [Iter {iteration}] Angle {ang:5.1f}° -> Score: {score:.2f}")
                
                # 记录缓存
                visited_scores[ang] = score
                
                # 保存可视化
                vis = rot_color.copy()
                x, y, w, h = res['bbox']
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(vis, f"Ang: {ang:.1f} Sc: {score:.0f}", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                vis_images.append(vis)
                
                # 更新全局最佳
                if score > best_score_global:
                    best_score_global = score
                    best_result_global = res
                    best_angle_global = ang
            
            # 更新本轮邻域最佳
            if score > neighborhood_best_score:
                neighborhood_best_score = score
                neighborhood_best_angle = ang
        
        # 决策逻辑
        if neighborhood_best_angle is not None:
            if neighborhood_best_angle == current_angle:
                # 当前中心就是最佳 -> 已经是局部峰值 -> 缩小步长精修
                new_step = current_step / 2.0
                print(f"      >>> Peak found at {current_angle} with step {current_step}. Refining step to {new_step}...")
                current_step = new_step
            else:
                # 邻居更好 -> 移动中心 -> 保持步长
                print(f"      >>> Moving center: {current_angle} -> {neighborhood_best_angle} (Score: {neighborhood_best_score:.2f})")
                current_angle = neighborhood_best_angle

    # 生成对比图
    
    # [新增] 最终裁判：基于字符识别结果对比
    # 分别对 原始图像(0度) 和 最佳旋转图像(best_angle) 进行识别
    
    # 1. 原始图像 (0度)
    print("  [Validation] Checking 0 degree (original)...")
    res_0 = locate_plate_region(gray)
    score_rec_0 = 0.0
    text_0 = ""
    if res_0.get('image') is not None:
        text_0, score_rec_0 = recognizer.recognize(res_0['image'], return_confidence=True)
    
    # 2. 最佳旋转图像
    print(f"  [Validation] Checking best angle {best_angle_global:.1f} degree...")
    # 如果最佳角度就是0，不需要额外识别
    if abs(best_angle_global) < 0.1:
        score_rec_best = score_rec_0
        text_best = text_0
        best_result_global = res_0
    else:
        # 确保 best_result_global 是对应的
        if best_result_global is None: # 防御性编程
             best_result_global = locate_plate_region(rotate_image(gray, best_angle_global))
             
        score_rec_best = 0.0
        text_best = ""
        if best_result_global.get('image') is not None:
             text_best, score_rec_best = recognizer.recognize(best_result_global['image'], return_confidence=True)
    
    print(f"  [Validation Result]")
    print(f"    Original (0°): Text='{text_0}', Score={score_rec_0:.4f}")
    print(f"    Rotated ({best_angle_global:.1f}°): Text='{text_best}', Score={score_rec_best:.4f}")
    
    final_angle = best_angle_global
    final_res = best_result_global
    final_score = best_score_global
    
    # 决策逻辑：只有当旋转后的识别分数 显著高于 原图时，才采用旋转
    # 阈值设定：比如提高 10% 或者 原图有问号而旋转后没有
    if score_rec_best > score_rec_0 + 0.05: # 简单的阈值
        print(f"  => CONFIRMED: Rotation improves recognition. ({score_rec_0:.4f} -> {score_rec_best:.4f})")
    elif "?" in text_0 and "?" not in text_best:
        print(f"  => CONFIRMED: Rotation fixes unknown chars. ('{text_0}' -> '{text_best}')")
    else:
        print(f"  => REJECTED: Rotation does not significantly improve recognition. Reverting to 0°.")
        final_angle = 0.0
        final_res = res_0
        final_score = score_rec_0 # 这里其实用回原图的投影分也没关系，或者不更新
    
    # 绘制最终结果
    h, w = image.shape[:2]
    comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
    comparison[:, :w] = image
    
    # 绘制旋转后的图 (使用 final_angle)
    if abs(final_angle) > 0.1:
        final_vis = rotate_image(image, final_angle)
    else:
        final_vis = image.copy()
        
    if final_res and final_res.get('bbox'):
        x, y, w_box, h_box = final_res['bbox']
        cv2.rectangle(final_vis, (x, y), (x+w_box, y+h_box), (0, 0, 255), 3)
    
    comparison[:, w:] = final_vis
    
    # 添加文字说明
    cv2.putText(comparison, f"Original (0 deg)", (20, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(comparison, f"Corrected ({final_angle:.1f} deg)", (w + 20, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
               
    # 叠加识别结果
    cv2.putText(comparison, f"Text: {text_0}", (20, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # 如果最终选择了旋转，显示旋转后的文字；否则显示原图文字
    display_text = text_best if abs(final_angle) > 0.1 else text_0
    cv2.putText(comparison, f"Text: {display_text}", (w + 20, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    output_path = output_dir / f"result_{img_path.name}"
    cv2.imwrite(str(output_path), comparison)
    print(f"  -> Saved comparison to {output_path}")
    print(f"  => Final Winner: Angle {final_angle} (Rec Score: {max(score_rec_0, score_rec_best):.4f})")
    print("")

# ==========================================
# [Main]
# ==========================================

def main():
    # 测试所有 input_images 下的图片 (包括原始正常图片和倾斜图片)
    input_dir = Path("input_images")
    output_dir = CONFIG['debug_output_dir']
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"[*] Starting Skew Correction Test V2 - Full Dataset")
    
    for img_path in input_dir.rglob("*"):
        if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']:
            process_single_image(img_path, output_dir)

if __name__ == "__main__":
    main()
