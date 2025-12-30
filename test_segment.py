"""
调试专用脚本：车牌字符按比例分割 (Debug Character Segmentation)

功能：
    此脚本用于测试和调整车牌字符的分割参数。
    它首先使用 plate_detector 定位车牌，然后在车牌区域上根据预设的【百分比坐标】
    绘制出 8 个字符的包围框（Bounding Boxes）。
    
    你可以通过调整下方的 [参数设置区域] 来精确控制每个字符框的位置和大小。
    所有参数都是相对于车牌宽高的百分比 (0.0 - 1.0)，这使得参数能适应不同像素大小的车牌。

使用方法：
    1. 修改 `IMAGE_PATH` 指定要测试的图片。
    2. 在 `CHAR_CONFIGS` 中调整每个字符的 `center_x`, `center_y`, `w`。
    3. 运行此脚本。
    4. 检查 `segment_debug` 目录下的 `plate_with_boxes.png` 查看效果。
    5. 检查 `segment_debug` 目录下的 `char_*.png` 查看单独切割出的字符。

"""

import cv2
import numpy as np
from pathlib import Path
import shutil
from plate_detector import locate_plate_region

# ==========================================
# [参数设置区域] - 在这里调整参数！
# ==========================================

# 1. 输入图片路径
IMAGE_PATH = r"d:\Digital_Image_Processing\FinalProject\plate_recog\input_images\normal_plate\normal_plate_1.jpg"

# [新增参数] 字符框宽高比设置
# 这里的倍率决定了框有多高 (Height = Width * Ratio)
# 下层字符 (数字/字母) 比例：1.55 (用户指定)
RATIO_BOTTOM = 1.55
# 上层字符 (汉字/字母) 比例：0.95 (默认方形略扁)
RATIO_TOP = 0.95

# 2. 字符分割配置 (双层车牌 8 字符模板)
# 
# 参数说明：
#   name:     字符名称 (用于文件名)
#   center_x: 中心点 X 坐标 (百分比, 0.0=左边, 1.0=右边)
#   center_y: 中心点 Y 坐标 (百分比, 0.0=顶边, 1.0=底边)
#   w:        框宽度 (百分比)
#
# 提示：
#   - 如果觉得框偏左了，增加 center_x
#   - 如果觉得框太窄了，增加 w
#   - 如果是双层车牌：
#       第一层通常有 2 个字符 (如 "苏E")
#       第二层通常有 5-6 个字符 (如 "88888" 或 "123456")

CHAR_CONFIGS = [
    # --- 第一行 (Top Row) ---
    # 字符 1 (汉字，如 "苏")
    {'name': '01_top_1', 'center_x': 0.37, 'center_y': 0.25, 'w': 0.13},
    # 字符 2 (字母，如 "E")
    {'name': '02_top_2', 'center_x': 0.62, 'center_y': 0.25, 'w': 0.13},
    
    # --- 第二行 (Bottom Row) ---
    # 字符 3
    {'name': '03_bottom_1', 'center_x': 0.1, 'center_y': 0.70, 'w': 0.15},
    # 字符 4
    {'name': '04_bottom_2', 'center_x': 0.26, 'center_y': 0.70, 'w': 0.15},
    # 字符 5
    {'name': '05_bottom_3', 'center_x': 0.42, 'center_y': 0.70, 'w': 0.15},
    # 字符 6
    {'name': '06_bottom_4', 'center_x': 0.58, 'center_y': 0.70, 'w': 0.15},
    # 字符 7
    {'name': '07_bottom_5', 'center_x': 0.74, 'center_y': 0.70, 'w': 0.15},
    # 字符 8
    {'name': '08_bottom_6', 'center_x': 0.902, 'center_y': 0.70, 'w': 0.15},
]

# ==========================================
# [代码执行逻辑]
# ==========================================

def run_segment_test():
    # 0. 准备输出目录
    output_dir = Path("segment_debug")
    if output_dir.exists():
        try:
            shutil.rmtree(output_dir)
        except:
            pass
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] 调试结果将保存至: {output_dir.absolute()}")

    # 1. 读取图像
    print(f"[*] 读取图像: {IMAGE_PATH}")
    image = cv2.imread(IMAGE_PATH) # 读取彩色图以便画彩色框
    if image is None:
        print("[Error] 无法读取图像，请检查路径！")
        return

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 定位车牌
    print("[*] 正在定位车牌区域...")
    # 使用 debug=False
    plate_result = locate_plate_region(image_gray, debug=False)
    
    # 获取车牌图像 (灰度)
    plate_img_gray = plate_result['image']
    
    # 这里的 bbox 是在原图上的坐标 (x, y, w, h)
    bx, by, bw, bh = plate_result['bbox']
    print(f"[*] 车牌定位成功: x={bx}, y={by}, w={bw}, h={bh}")

    # 为了画图好看，我们把车牌切出来的图转成 BGR
    # 注意：这里需要从原图 image 切割，因为 locate_plate_region 返回的是 gray
    plate_img_color = image[by:by+bh, bx:bx+bw]

    # ==========================================
    # [新增] 3. 图像预处理与归一化
    # ==========================================
    print("[*] 进行预处理与归一化...")
    
    # 3.1 尺寸归一化 (固定高度为 140px，宽度按比例缩放)
    target_h = 140
    h, w = plate_img_gray.shape
    scale = target_h / h
    target_w = int(w * scale)
    
    plate_img_gray = cv2.resize(plate_img_gray, (target_w, target_h))
    plate_img_color = cv2.resize(plate_img_color, (target_w, target_h))
    
    print(f"    -> 归一化后尺寸: {target_w} x {target_h}")

    # 3.2 图像增强 (高斯模糊 + 二值化 + 形态学)
    # 高斯模糊
    blurred = cv2.GaussianBlur(plate_img_gray, (3, 3), 0)
    
    # 二值化 (使用固定阈值 140，与 character_recognizer 保持一致)
    _, plate_binary = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)
    
    # [新增] 智能背景反转：确保是黑底白字
    h_bin, w_bin = plate_binary.shape
    border_mean = (np.mean(plate_binary[0, :]) + np.mean(plate_binary[h_bin-1, :]) + 
                   np.mean(plate_binary[:, 0]) + np.mean(plate_binary[:, w_bin-1])) / 4.0
                   
    if border_mean > 127:
        print("    -> 检测到白底，自动反转为黑底白字...")
        plate_binary = cv2.bitwise_not(plate_binary)
    
    # 形态学操作 (开运算去除噪点)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    plate_processed = cv2.morphologyEx(plate_binary, cv2.MORPH_OPEN, kernel)
    
    # 保存处理后的二值图
    cv2.imwrite(str(output_dir / "plate_processed_binary.png"), plate_processed)

    # 4. 根据参数绘制 8 个框
    print("[*] 开始根据百分比参数分割字符...")
    
    plate_h, plate_w = plate_processed.shape
    
    for cfg in CHAR_CONFIGS:
        name = cfg['name']
        cx_pct = cfg['center_x']
        cy_pct = cfg['center_y']
        w_pct = cfg['w']
        
        # 确定使用哪个宽高比
        if 'top' in name:
            ratio = RATIO_TOP
        else:
            ratio = RATIO_BOTTOM
        
        # 计算像素坐标
        # 宽度 (基于车牌宽度)
        box_w = int(plate_w * w_pct)
        # 高度 (基于宽度 * 倍率)
        box_h = int(box_w * ratio)
        
        # 中心点像素坐标
        center_x = int(plate_w * cx_pct)
        center_y = int(plate_h * cy_pct)
        
        # 左上角坐标
        x1 = center_x - box_w // 2
        y1 = center_y - box_h // 2
        
        # 边界检查 (防止越界)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(plate_w, x1 + box_w)
        y2 = min(plate_h, y1 + box_h)
        
        # 实际的 w, h 可能因为越界而被截断
        real_w = x2 - x1
        real_h = y2 - y1
        
        # 4.1 保存单独的字符图片 (使用处理后的二值图)
        if real_w > 0 and real_h > 0:
            char_img = plate_processed[y1:y2, x1:x2]
            char_filename = output_dir / f"char_{name}.png"
            cv2.imwrite(str(char_filename), char_img)
            # print(f"  -> 保存字符: {name} ({real_w}x{real_h})") # 用户要求不需要展示处理步骤
        
        # 4.2 在总览图上画框 (彩色图)
        color = (0, 255, 0) 
        thickness = 2
        cv2.rectangle(plate_img_color, (x1, y1), (x2, y2), color, thickness)
        
        # 画中心点
        cv2.circle(plate_img_color, (center_x, center_y), 2, (0, 0, 255), -1)
        
        # 标文字
        cv2.putText(plate_img_color, name.split('_')[-1], (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # 5. 保存总览图
    overview_path = output_dir / "plate_with_boxes.png"
    cv2.imwrite(str(overview_path), plate_img_color)
    print(f"[*] 分割效果总览图已保存: {overview_path}")
    print("\n[Done] 调试完成！请打开 segment_debug 文件夹查看效果。")
    print("      如果不满意，请修改脚本顶部的 RATIO_BOTTOM 或 CHAR_CONFIGS 参数。")

if __name__ == "__main__":
    run_segment_test()