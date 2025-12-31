"""
车牌识别批量处理脚本 (Batch Processing Script)

功能：
    遍历 input_images 目录下的所有图片，执行车牌定位与识别。
    生成包含以下内容的组合结果图：
    1. 原图 (带车牌定位框)
    2. 车牌定位图 (原始裁剪)
    3. 字符分割与识别调试图
    4. 最终识别文字结果
    
    结果保存至 output_image 目录，保持原有目录结构。
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
import sys
import os
from PIL import Image, ImageDraw, ImageFont

from plate_detector import locate_plate_region, rotate_image
from character_recognizer import CharacterRecognizer

# ==========================================
# [配置参数]
# ==========================================
INPUT_ROOT_DIR = Path("input_images")
OUTPUT_ROOT_DIR = Path("output_image")

# 下层字符 (数字/字母) 比例
RATIO_BOTTOM = 1.55 
# 上层字符 (汉字/字母) 比例
RATIO_TOP = 0.95

# 字符分割配置 (双层车牌 8 字符模板)
CHAR_CONFIGS = [
    {'name': '01_top_1', 'center_x': 0.37, 'center_y': 0.25, 'w': 0.13},
    {'name': '02_top_2', 'center_x': 0.62, 'center_y': 0.25, 'w': 0.13},
    {'name': '03_bottom_1', 'center_x': 0.1, 'center_y': 0.70, 'w': 0.15},
    {'name': '04_bottom_2', 'center_x': 0.26, 'center_y': 0.70, 'w': 0.15},
    {'name': '05_bottom_3', 'center_x': 0.42, 'center_y': 0.70, 'w': 0.15},
    {'name': '06_bottom_4', 'center_x': 0.58, 'center_y': 0.70, 'w': 0.15},
    {'name': '07_bottom_5', 'center_x': 0.74, 'center_y': 0.70, 'w': 0.15},
    {'name': '08_bottom_6', 'center_x': 0.902, 'center_y': 0.70, 'w': 0.15},
]

def create_composite_result(original_img, bbox, plate_raw, recog_vis, result_text, save_path):
    """
    创建组合结果图
    布局：
    [ 原图 (带框) ]
    [ 车牌原图 ] [ 识别调试图 ]
    [ 识别结果文字 ]
    """
    bx, by, bw, bh = bbox
    
    # 1. 在原图上画框
    img_with_box = original_img.copy()
    cv2.rectangle(img_with_box, (bx, by), (bx+bw, by+bh), (0, 255, 0), 5)
    
    # 转换为 BGR (如果输入是灰度)
    if len(plate_raw.shape) == 2:
        plate_raw_bgr = cv2.cvtColor(plate_raw, cv2.COLOR_GRAY2BGR)
    else:
        plate_raw_bgr = plate_raw
        
    if len(recog_vis.shape) == 2:
        recog_vis_bgr = cv2.cvtColor(recog_vis, cv2.COLOR_GRAY2BGR)
    else:
        recog_vis_bgr = recog_vis

    # 2. 计算尺寸
    # 设定目标宽度为 1000 像素 (或原图宽度，取大者)
    target_w = max(1000, img_with_box.shape[1])
    
    # 缩放原图
    scale_orig = target_w / img_with_box.shape[1]
    h_orig = int(img_with_box.shape[0] * scale_orig)
    img_disp = cv2.resize(img_with_box, (target_w, h_orig))
    
    # 车牌区域宽度 (左右各一半)
    plate_disp_w = target_w // 2
    
    # 缩放车牌原图
    scale_p1 = plate_disp_w / plate_raw_bgr.shape[1]
    h_p1 = int(plate_raw_bgr.shape[0] * scale_p1)
    plate_raw_disp = cv2.resize(plate_raw_bgr, (plate_disp_w, h_p1))
    
    # 缩放识别调试图
    scale_p2 = plate_disp_w / recog_vis_bgr.shape[1]
    h_p2 = int(recog_vis_bgr.shape[0] * scale_p2)
    plate_recog_disp = cv2.resize(recog_vis_bgr, (plate_disp_w, h_p2))
    
    # 车牌行高度
    row2_h = max(h_p1, h_p2)
    
    # 文字区域
    text_h = 120
    
    # 3. 创建画布
    total_h = h_orig + row2_h + text_h
    canvas = np.zeros((total_h, target_w, 3), dtype=np.uint8)
    
    # 4. 填充内容
    # Top: 原图
    canvas[:h_orig, :] = img_disp
    
    # Middle: 车牌对比
    # 左边放原车牌，右边放识别过程图
    canvas[h_orig:h_orig+h_p1, :plate_disp_w] = plate_raw_disp
    canvas[h_orig:h_orig+h_p2, plate_disp_w:] = plate_recog_disp
    
    # Bottom: 文字
    # 白色背景
    canvas[h_orig+row2_h:, :] = (255, 255, 255) # White background for text
    
    text = f"Result: {result_text}"
    
    # 使用 PIL 绘制中文
    # 将 OpenCV 图片 (BGR) 转换为 PIL 图片 (RGB)
    canvas_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas_pil)
    
    # 加载字体 (使用 Windows 系统自带的 SimHei)
    font_path = "C:/Windows/Fonts/simhei.ttf"
    if not os.path.exists(font_path):
        # 备选字体
        font_path = "C:/Windows/Fonts/msyh.ttc"
        
    try:
        font = ImageFont.truetype(font_path, 60)
    except:
        print("[Warn] 无法加载中文字体，使用默认字体")
        font = ImageFont.load_default()
    
    # 计算文字位置
    # PIL 的 textbbox 或 textsize
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        fw, fh = right - left, bottom - top
    except AttributeError:
        # 旧版 Pillow
        fw, fh = draw.textsize(text, font=font)
    
    tx = (target_w - fw) // 2
    ty = h_orig + row2_h + (text_h - fh) // 2
    
    draw.text((tx, ty), text, font=font, fill=(0, 0, 0))
    
    # 转回 OpenCV 格式
    canvas = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)
    
    # 5. 保存
    cv2.imwrite(str(save_path), canvas)

def process_single_image(img_path: Path, input_root: Path, output_root: Path, recognizer: CharacterRecognizer):
    # 计算相对路径，用于保持目录结构
    try:
        rel_path = img_path.relative_to(input_root)
    except ValueError:
        rel_path = img_path.name
        
    output_dir = output_root / rel_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Processing: {img_path.name} ...")
    
    # 临时目录，用于存放 CharacterRecognizer 生成的调试图
    temp_debug_dir = output_dir / f"temp_{img_path.stem}"
    if temp_debug_dir.exists():
        shutil.rmtree(temp_debug_dir)
    temp_debug_dir.mkdir()
    
    try:
        # 读取图像
        # imread 不支持中文路径，使用 imdecode
        img_np = np.fromfile(str(img_path), dtype=np.uint8)
        image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        if image is None:
            print(f"  [Error] 无法读取图像: {img_path}")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. 定位 (传入 recognizer 开启倾斜校正)
        plate_result = locate_plate_region(gray, debug=False, recognizer=recognizer)
        
        if not plate_result:
            print(f"  [Warn] 未找到车牌: {img_path.name}")
            return

        # 检查是否进行了旋转校正
        if plate_result.get('rotation_angle', 0.0) != 0.0:
            angle = plate_result['rotation_angle']
            print(f"  -> Applied Skew Correction: {angle:.1f} deg")
            # 更新用于显示的原图 (Color)
            image = rotate_image(image, angle)
            # 更新 gray 图 (虽然识别已经用过了，但后续可能用到?)
            # plate_result['rotated_image_full'] 已经是旋转后的 gray
            
        bbox = plate_result['bbox'] # (x, y, w, h)
        plate_img_gray = plate_result['image']

        # 2. 识别
        plate_number = recognizer.recognize(
            plate_image=plate_img_gray,
            debug_output_dir=temp_debug_dir,
            configs=CHAR_CONFIGS,
            ratio_bottom=RATIO_BOTTOM,
            ratio_top=RATIO_TOP
        )
        print(f"  -> Result: {plate_number}")
        
        # 3. 获取生成的调试图 (分割图)
        recog_vis_path = temp_debug_dir / "plate_recog_debug_vis.png"
        if recog_vis_path.exists():
            recog_vis_np = np.fromfile(str(recog_vis_path), dtype=np.uint8)
            recog_vis = cv2.imdecode(recog_vis_np, cv2.IMREAD_COLOR)
        else:
            # 如果没生成调试图，就用灰度图代替
            recog_vis = plate_img_gray
            
        # 4. 生成组合结果
        save_path = output_dir / f"{img_path.stem}_result.jpg"
        create_composite_result(image, bbox, plate_img_gray, recog_vis, plate_number, save_path)
        print(f"  -> Saved to: {save_path}")

    except Exception as e:
        print(f"  [Error] 处理出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理临时目录
        if temp_debug_dir.exists():
            try:
                shutil.rmtree(temp_debug_dir)
            except:
                pass

def main():
    if not INPUT_ROOT_DIR.exists():
        print(f"❌ 输入目录不存在: {INPUT_ROOT_DIR}")
        return

    # 清空并重建输出目录
    if OUTPUT_ROOT_DIR.exists():
        shutil.rmtree(OUTPUT_ROOT_DIR)
    OUTPUT_ROOT_DIR.mkdir()
    
    print("==========================================")
    print("      车牌识别批量处理任务开始")
    print(f" 输入目录: {INPUT_ROOT_DIR.absolute()}")
    print(f" 输出目录: {OUTPUT_ROOT_DIR.absolute()}")
    print("==========================================\n")
    
    recognizer = CharacterRecognizer(template_dir="char_templates")
    
    # 递归遍历所有图片
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    count = 0
    
    for img_path in INPUT_ROOT_DIR.rglob("*"):
        if img_path.suffix.lower() in image_extensions:
            process_single_image(img_path, INPUT_ROOT_DIR, OUTPUT_ROOT_DIR, recognizer)
            count += 1
            
    print("\n==========================================")
    print(f" 任务完成! 共处理 {count} 张图片")
    print(f" 结果已保存至: {OUTPUT_ROOT_DIR.absolute()}")
    print("==========================================")

if __name__ == "__main__":
    main()
