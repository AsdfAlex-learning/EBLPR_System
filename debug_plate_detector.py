"""
调试专用脚本：车牌定位参数调优 (Debug Plate Detector)

功能：
    此脚本将 `plate_detector.py` 中的核心逻辑“平铺直叙”地展示出来。
    你可以直接在这里修改参数（如核大小、权重、阈值），运行脚本后，
    在 `debug_sandbox` 目录下查看每一步的中间结果图像。
    
    调试满意后，请将调整好的参数回填到 `plate_detector.py` 中。

使用方法：
    1. 修改 `IMAGE_PATH` 指定要测试的图片。
    2. 修改下面的 [参数设置区域] 中的数值。
    3. 运行此脚本。
    4. 观察控制台输出的评分细节和 `debug_sandbox` 目录下的图像。

基础薄弱者调试指南：
    - 如果车牌断裂成几截：增大 `MORPH_KERNEL_WIDTH` (形态学宽度)。
    - 如果车牌和背景粘连在一起：减小 `MORPH_KERNEL_WIDTH` 或 `MORPH_KERNEL_HEIGHT`。
    - 如果车牌没被选中：检查控制台输出的 "Rejected"，看是因为 宽高比(AR) 还是 面积(Area) 不达标。
    - 如果选中了错误的区域：调整评分权重 (WEIGHT_*)，比如增加位置权重，强迫选择中心的框。
"""

import cv2
import numpy as np
from pathlib import Path
import shutil

# ==========================================
# [参数设置区域] - 在这里调整参数！
# ==========================================

# 1. 输入图片路径 (建议使用绝对路径或相对路径)
IMAGE_PATH = r"d:\Digital_Image_Processing\FinalProject\plate_recog\input_images\normal_plate\normal_plate_3.jpg"

# 2. 图像预处理参数
TARGET_WIDTH = 800          # 归一化宽度，所有处理都在这个尺度下进行
CROP_RATIO_W = 0.20         # 左右各裁剪掉 10% (保留中间 80%)
CROP_RATIO_H = 0.20         # 上下各裁剪掉 20% (保留中间 60%)

# 3. 形态学操作参数 (最关键！)
# 作用：将破碎的字符边缘连成一个整体矩形
# 策略调整：既然现在的边缘检测已经能提取出车牌的【白色边框】，
# 我们就不再需要巨大的核去强行“糊”字符了，而是应该用小一点的核，
# 顺着这个边框把它连接闭合即可。这样可以避免把周围的背景也粘进来，结果会更精准！
MORPH_KERNEL_WIDTH = 25     # (原 55 -> 25) 只要能连上断裂的边框即可
MORPH_KERNEL_HEIGHT = 5     # (原 15 -> 5)  只要能保持上下连通即可

# 4. 候选框筛选参数 (硬筛选)
MIN_AR = 1.5                # 最小宽高比 (Aspect Ratio)
MAX_AR = 6.0                # 最大宽高比
MIN_AREA = 600              # 最小面积 (像素)
MAX_AREA_RATIO = 0.4        # 最大面积占全图比例

# 5. 评分权重参数 (软选择)
# 总分 = AR得分*权重 + 密度得分*权重 + 跳变得分*权重 + 位置得分*权重
TARGET_AR = 181.0 / 91.5    # 理想宽高比 (约 1.98)
WEIGHT_AR = 0.3             # 宽高比权重
WEIGHT_DENSITY = 0.2        # 纹理密度权重
WEIGHT_TRANSITIONS = 0.1    # 边缘跳变权重
WEIGHT_CENTER = 0.4         # 中心位置权重 (越重要设得越大)

# 6. 精修参数
REFINE_PAD_W = 0.2          # 精修前左右扩充比例
REFINE_PAD_H = 0.5          # 精修前上下扩充比例

# ==========================================
# [代码执行逻辑] - 以下逻辑模拟 plate_detector.py
# ==========================================

def run_debug():
    # 0. 准备输出目录
    output_dir = Path("debug_sandbox")
    if output_dir.exists():
        try:
            shutil.rmtree(output_dir)
        except:
            pass
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] 调试结果将保存至: {output_dir.absolute()}")

    # 1. 读取图像
    print(f"[*] 读取图像: {IMAGE_PATH}")
    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("[Error] 无法读取图像，请检查路径！")
        return

    h_orig, w_orig = image.shape
    
    # ---------------------------------------------------------
    # 步骤 00: 归一化 (Resize)
    # ---------------------------------------------------------
    scale = TARGET_WIDTH / w_orig
    target_height = int(h_orig * scale)
    resized = cv2.resize(image, (TARGET_WIDTH, target_height))
    
    cv2.imwrite(str(output_dir / "00_original_resized.png"), resized)
    print(f"[Step 00] 归一化完成: {resized.shape}")

    # ---------------------------------------------------------
    # 步骤 01: 中心裁剪 (Center Crop)
    # ---------------------------------------------------------
    crop_x = int(TARGET_WIDTH * CROP_RATIO_W)
    crop_y = int(target_height * CROP_RATIO_H)
    crop_w = TARGET_WIDTH - 2 * crop_x
    crop_h = target_height - 2 * crop_y
    
    cropped = resized[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    
    cv2.imwrite(str(output_dir / "01_center_crop.png"), cropped)
    print(f"[Step 01] 中心裁剪完成: 保留区域 x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")

    # ---------------------------------------------------------
    # 步骤 02: 边缘检测 (Sobel) - 针对电动车双层车牌优化
    # ---------------------------------------------------------
    # 问题分析：
    #   用户反馈车牌是双层结构：
    #   - 上层：汉字 (笔画少，垂直边缘弱)
    #   - 下层：数字字母 (笔画多，垂直边缘强且密集)
    #   - 现状：Sobel X 只检测垂直边缘，导致下层很亮，上层很暗。闭运算后，上层汉字直接丢了，只剩下一半车牌。
    # 
    # 解决方案：
    #   1. 引入 Y 方向边缘：汉字有很多横向笔画，Sobel Y 能很好地捕捉它们。
    #      混合权重：X方向 (0.5) + Y方向 (0.5)。
    #   2. 增强弱边缘：在二值化前或后，对整体亮度做均衡，或者在形态学之前先进行一次“膨胀”，把弱边缘强行放大。
    
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(cropped, (5, 5), 0)
    
    # Sobel X (垂直边缘)
    sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    
    # Sobel Y (水平边缘) - 新增！为了捕捉汉字的横向笔画
    sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    
    # 混合边缘 (各占 50%)
    combined_sobel = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    
    cv2.imwrite(str(output_dir / "02_sobel.png"), combined_sobel)
    print(f"[Step 02] 边缘检测完成 (已启用 X+Y 双向检测)")

    # ---------------------------------------------------------
    # 步骤 03: 二值化 (Otsu Threshold)
    # ---------------------------------------------------------
    ret, binary = cv2.threshold(combined_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    cv2.imwrite(str(output_dir / "03_binary.png"), binary)
    print(f"[Step 03] 二值化完成 (阈值={ret})")
    
    # ---------------------------------------------------------
    # 步骤 03.5: 预膨胀 (Pre-Dilation) - 新增！
    # ---------------------------------------------------------
    # 目的：在进行闭运算（大融合）之前，先让微弱的汉字笔画“变粗”一点，
    # 这样它们更有机会在后续的闭运算中被连成一片，而不是被视作噪点过滤掉。
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, kernel_dilate, iterations=1)
    print(f"[Step 03.5] 预膨胀完成 (强化弱边缘)")

    # ---------------------------------------------------------
    # 步骤 04: 边框修复 (Border Fix) - 替代原来的“闭运算”
    # ---------------------------------------------------------
    # 用户新策略：
    #   Step 03 的二值图已经有了清晰的“白色边框矩形”。
    #   我们不再需要大核闭运算去“糊”字符，而是直接利用这个边框。
    #   这里只做极其微小的连接，防止边框线条断裂。
    
    kernel_fix = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 仅闭运算一次，连接细微断点
    binary_border = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_fix)
    
    cv2.imwrite(str(output_dir / "04_border_fix.png"), binary_border)
    print(f"[Step 04] 边框修复完成 (微小闭运算, Kernel=3x3)")

    # ---------------------------------------------------------
    # 步骤 05: 轮廓提取与特征筛选 (Contour Extraction & Filtering)
    # ---------------------------------------------------------
    # 关键点：
    #   1. 使用 RETR_TREE：因为车牌边框（父轮廓）内部通常包含字符（子轮廓）。
    #      我们可以利用“有子轮廓”作为车牌边框的强特征。
    #   2. 针对双层车牌的几何特征进行筛选。
    
    contours, hierarchy = cv2.findContours(binary_border, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    debug_contours_all = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    candidates = []
    
    print(f"\n[Step 05] 开始基于边框特征筛选 (总轮廓数: {len(contours)})...")
    
    if hierarchy is not None:
        # hierarchy 结构: [Next, Previous, First_Child, Parent]
        hierarchy = hierarchy[0]
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = w * h
            
            # 特征 1: 是否有子轮廓 (First_Child != -1)
            # 车牌边框里面肯定有字符，所以应该有子轮廓
            has_child = hierarchy[i][2] != -1
            
            # 特征 2: 父轮廓索引 (用于判断是否是顶层轮廓)
            parent_idx = hierarchy[i][3]
            
            # 绘制所有轮廓(灰色细线)
            cv2.rectangle(debug_contours_all, (x, y), (x+w, y+h), (200, 200, 200), 1)
            
            # --- 筛选逻辑 ---
            keep = True
            reason = ""
            
            # 1. 宽高比 (Aspect Ratio)
            # 双层车牌比较方，大概在 1.2 到 3.0 之间 (正常单层是 3.0-5.0)
            if not (1.2 < aspect_ratio < 3.0):
                keep = False
                reason = f"AR({aspect_ratio:.2f})不符双层特征"
            
            # 2. 面积 (Area)
            # 边框不会太小，至少要能包住几个字
            elif area < 2000: # 稍微提高阈值，因为现在找的是整个外框
                keep = False
                reason = f"面积({area})太小"
                
            # 3. 边框完整性 (Solidity/Extent)
            # 如果是空心边框，contourArea 应该是边框线本身的面积，而 w*h 是包围盒面积
            # 车牌边框通常是一个闭合的矩形环
            
            # 4. (可选) 必须有子轮廓？
            # 有时候边框太粗把字符吞了，或者二值化太强，这里先作为加分项，不强制
            # if not has_child and area > 5000:
            #    keep = False
            #    reason = "无内部字符(无子轮廓)"
            
            if keep:
                # 计算中心位置得分
                cx = x + w / 2
                cy = y + h / 2
                img_cx, img_cy = crop_w / 2, crop_h / 2
                dist = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
                
                candidates.append({
                    'idx': i,
                    'bbox': (x, y, w, h),
                    'aspect_ratio': aspect_ratio,
                    'area': area,
                    'has_child': has_child,
                    'dist_from_center': dist
                })
                # 画出通过筛选的(绿色粗线)
                cv2.rectangle(debug_contours_all, (x, y), (x+w, y+h), (0, 255, 0), 2)
                child_status = "有子轮廓" if has_child else "无子轮廓"
                print(f"  -> 候选 {i}: 通过! AR={aspect_ratio:.2f}, Area={area}, {child_status}")
            else:
                if area > 1000:
                    print(f"  -> 候选 {i}: 拒绝 - {reason}")
    
    cv2.imwrite(str(output_dir / "05_filtered_contours.png"), debug_contours_all)
    
    # ---------------------------------------------------------
    # 步骤 06: 候选评分与最终决策 (Candidate Scoring & Final Decision)
    # ---------------------------------------------------------
    # 这一步是从 [Step 05] 筛选出的所有“潜在车牌框”中，
    # 挑出那个“最像车牌”的唯一真理。
    #
    # 筛选逻辑 (Score越小越好)：
    #   Total_Cost = (距离惩罚 * 0.4) + (面积惩罚 * 0.6)
    #   1. 距离惩罚：越靠近图像中心，得分越高（Cost越低）。
    #   2. 面积惩罚：面积越小越紧致，得分越高（Cost越低）。
    #      (避免选中那种把整个车头都框进去的巨大误检)
    #
    # 特殊规则：
    #   如果候选池里【只有 1 个】候选框，别犹豫，直接选它！
    #   (这时候通常说明算法非常确信，不需要再进行复杂的比对了)
    
    print(f"\n[Step 06] 选择最佳候选...")
    
    best_candidate = None
    min_score = 99999.0 
    
    # 1. 优先选择有子轮廓的 (说明里面有字)
    candidates_with_child = [c for c in candidates if c['has_child']]
    
    # 逻辑修正：
    # 如果找到了有子轮廓的候选，就只在这些优质候选中挑。
    # 如果【没有】找到任何有子轮廓的候选（可能是二值化把字符糊住了），
    # 那也不能直接报错，而是应该退而求其次，在所有候选里挑一个最好的。
    # 
    # 原来的代码： pool = candidates_with_child if candidates_with_child else candidates
    # 这行代码本身没问题，但如果 `candidates_with_child` 为空，
    # 而 `candidates` 也为空（Step 05 就全被刷掉了），那就会报 Error。
    #
    # 但你说的情况是“明明有一个解”，说明 candidates 不为空。
    # 那么唯一可能的原因是：那个唯一的解没有子轮廓 (has_child=False)，
    # 导致 candidates_with_child 为空，然后 pool 变成了 candidates。
    #
    # 让我们打印一下调试信息，看看发生了什么。
    
    pool = []
    if candidates_with_child:
        pool = candidates_with_child
        print(f"[Info] 发现 {len(pool)} 个带子轮廓的优质候选，仅在其中筛选。")
    else:
        pool = candidates
        print(f"[Info] 未发现带子轮廓的候选，降级在所有 {len(pool)} 个候选中筛选。")
    
    if not pool:
        print("[Error] 没有找到任何有效候选框！(Step 05 筛选后列表为空)")
        return

    # --- 新增逻辑：单候选直接保留 ---
    if len(pool) == 1:
        best_candidate = pool[0]
        print(f"[*] 命中特殊规则：仅有一个候选框，直接选中！")
        print(f"    Index={best_candidate['idx']}, AR={best_candidate['aspect_ratio']:.2f}")
    else:
        # --- 多候选PK逻辑 ---
        best_score = -999.0
        
        for c in pool:
            # 1. 宽高比评分 (重中之重)
            # 用户指示：越接近181:91.5的优先级越高
            target_ar = TARGET_AR
            ar_diff = abs(c['aspect_ratio'] - target_ar)
            # 收紧评分曲线：差异超过 0.5 分数就归零
            score_ar = 1.0 - min(1.0, ar_diff / 0.5)
            
            # 2. 距离评分 (越靠近中心越好)
            max_dist_ref = np.sqrt(crop_w**2 + crop_h**2) / 2
            # 放宽距离限制 (0.4 -> 0.6)
            norm_dist = c['dist_from_center'] / (max_dist_ref * 0.6)
            score_center = max(0.0, 1.0 - norm_dist)
            
            # 3. 面积评分 (可选，这里简化，只用 AR 和 Center)
            # 在 plate_detector 中还有 density 和 transitions，这里仅作几何筛选演示
            
            # 综合评分：大幅提升 AR 权重
            total_score = score_ar * 0.7 + score_center * 0.3
            
            print(f"  -> 候选 {c['idx']}: Score={total_score:.3f} (AR_Score={score_ar:.2f}, Center_Score={score_center:.2f})")
            
            if total_score > best_score:
                best_score = total_score
                best_candidate = c
            
    if best_candidate:
        if len(pool) > 1:
            print(f"[*] 最佳候选: Index={best_candidate['idx']}, AR={best_candidate['aspect_ratio']:.2f}, HasChild={best_candidate['has_child']}")
        
        # 映射回原图坐标 (bbox_global)
        bx, by, bw, bh = best_candidate['bbox']
        x_global = bx + crop_x
        y_global = by + crop_y
        best_candidate['bbox_global'] = (x_global, y_global, bw, bh)
        
        # 在 debug 图上标出最终选择 (红色)
        cv2.rectangle(debug_contours_all, (bx, by), (bx+bw, by+bh), (0, 0, 255), 3)
        cv2.imwrite(str(output_dir / "06_final_selection.png"), debug_contours_all)
    else:
        print("[Error] 无法确定最佳候选。")
        return

    # ---------------------------------------------------------
    # 步骤 07: 裁剪结果 (Crop)
    # ---------------------------------------------------------
    # 映射回 resized 原图坐标
    x_res, y_res, w_res, h_res = best_candidate['bbox_global']
    
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
    cv2.imwrite(str(output_dir / "07_rough_crop.png"), rough_crop)
    print(f"[Step 07] 粗略剪裁完成: {rough_crop.shape}")

    # ---------------------------------------------------------
    # 步骤 08: 边缘精修 (Projection Refinement)
    # ---------------------------------------------------------
    # 这里为了简化展示，只显示提取出的边缘图，实际 plate_detector 还有复杂的投影逻辑
    roi_sobel = cv2.Sobel(rough_crop, cv2.CV_8U, 1, 0, ksize=3)
    _, roi_edges = cv2.threshold(roi_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    cv2.imwrite(str(output_dir / "08_refine_edges.png"), roi_edges)
    print(f"[Step 08] 精修边缘图生成 (后续会进行投影切割)")
    
    # 在原图上画出最终的大致位置
    final_debug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(final_debug, (x, y), (x+w, y+h), (0, 0, 255), 3)
    cv2.imwrite(str(output_dir / "09_final_rough_location.png"), final_debug)
    
    print("\n[Done] 所有步骤执行完毕！请查看 debug_sandbox 文件夹。")

if __name__ == "__main__":
    run_debug()
