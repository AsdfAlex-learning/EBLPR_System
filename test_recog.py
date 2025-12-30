"""
识别功能验证脚本 (Recognition Test)

功能：
    此脚本用于验证 character_recognizer.py 的最终识别效果。
    它整合了定位、分割和识别的全流程。
    你可以像在 test_segment.py 中一样调整 CHAR_CONFIGS 和 RATIO 参数，
    这些参数会直接传递给识别器，方便快速调优。

使用方法：
    1. 修改 `IMAGE_PATH` 指定要测试的图片。
    2. 调整下方的 `CHAR_CONFIGS` 或 `RATIO` 参数。
    3. 运行脚本，查看控制台输出的识别结果。
    4. 检查 `recog_debug` 目录下的调试图片。
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
import sys

# 添加backend目录到Python路径 (如果需要)
# sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from plate_detector import locate_plate_region
from character_recognizer import CharacterRecognizer

# ==========================================
# [参数设置区域] - 这里的参数会覆盖 character_recognizer.py 中的默认值
# ==========================================

IMAGE_PATH = r"d:\Digital_Image_Processing\FinalProject\plate_recog\input_images\normal_plate\normal_plate_1.jpg"

# 下层字符 (数字/字母) 比例
RATIO_BOTTOM = 1.55 
# 上层字符 (汉字/字母) 比例
RATIO_TOP = 0.95

# 字符分割配置 (双层车牌 8 字符模板) - 从 test_segment.py 同步过来的最佳参数
CHAR_CONFIGS = [
    # --- 第一行 (Top Row) ---
    # 字符 1 (汉字，如 "苏")
    {'name': '01_top_1', 'center_x': 0.37, 'center_y': 0.25, 'w': 0.18},
    # 字符 2 (字母，如 "E")
    {'name': '02_top_2', 'center_x': 0.62, 'center_y': 0.25, 'w': 0.18},
    
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

# 注意：为了让 bottom 的字符更紧凑，我这里手动根据之前 test_segment 的经验微调了一下
# 您可以随时改回 character_recognizer.py 里的默认值，或者在这里继续调。

def run_recog_test():
    # 0. 准备输出目录
    output_dir = Path("recog_debug")
    if output_dir.exists():
        try:
            shutil.rmtree(output_dir)
        except:
            pass
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] 调试结果将保存至: {output_dir.absolute()}")

    # 1. 初始化识别器
    print("[*] 初始化字符识别器...")
    recognizer = CharacterRecognizer(template_dir="char_templates")
    
    # 2. 读取图像
    print(f"[*] 读取图像: {IMAGE_PATH}")
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("[Error] 无法读取图像，请检查路径！")
        return
        
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. 定位车牌
    print("[*] 正在定位车牌区域...")
    plate_result = locate_plate_region(image_gray, debug=True, debug_dir=output_dir)
    
    if not plate_result:
        print("[Error] 未找到车牌区域！")
        return

    # 获取车牌图像 (灰度)
    plate_img_gray = plate_result['image']
    bx, by, bw, bh = plate_result['bbox']
    print(f"[*] 车牌定位成功: x={bx}, y={by}, w={bw}, h={bh}")

    # 4. 执行识别
    print("[*] 开始识别...")
    
    # 将我们在上面定义的参数传递给 recognize 函数
    plate_number = recognizer.recognize(
        plate_image=plate_img_gray,
        debug_output_dir=output_dir,
        configs=CHAR_CONFIGS,
        ratio_bottom=RATIO_BOTTOM,
        ratio_top=RATIO_TOP
    )
    
    print("\n" + "="*40)
    print(f" >>> 最终识别结果: {plate_number}")
    print("="*40 + "\n")
    
    # 在原图上画结果
    result_vis = image.copy()
    cv2.rectangle(result_vis, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
    cv2.putText(result_vis, plate_number, (bx, by-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
    cv2.imwrite(str(output_dir / "final_result.png"), result_vis)
    print(f"[*] 最终效果图已保存: {output_dir / 'final_result.png'}")

if __name__ == "__main__":
    run_recog_test()
