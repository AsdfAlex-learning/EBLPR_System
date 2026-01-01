import cv2
import numpy as np
from pathlib import Path
import shutil
import sys
import os
from PIL import Image, ImageDraw, ImageFont

from plate_detector import locate_plate_region, rotate_image
from character_recognizer import CharacterRecognizer

INPUT_ROOT_DIR = Path("input_images")
OUTPUT_ROOT_DIR = Path("output_image")

RATIO_BOTTOM = 1.55 
RATIO_TOP = 0.95

# 字符位置配置
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
   
    # 创建合成结果图 (原图+车牌+识别过程+文本)。
    bx, by, bw, bh = bbox
    
    # 绘制边界框
    img_with_box = original_img.copy()
    cv2.rectangle(img_with_box, (bx, by), (bx+bw, by+bh), (0, 255, 0), 5)
    
    if len(plate_raw.shape) == 2:
        plate_raw_bgr = cv2.cvtColor(plate_raw, cv2.COLOR_GRAY2BGR)
    else:
        plate_raw_bgr = plate_raw
        
    if len(recog_vis.shape) == 2:
        recog_vis_bgr = cv2.cvtColor(recog_vis, cv2.COLOR_GRAY2BGR)
    else:
        recog_vis_bgr = recog_vis

    # 计算布局
    target_w = max(1000, img_with_box.shape[1])
    scale_orig = target_w / img_with_box.shape[1]
    h_orig = int(img_with_box.shape[0] * scale_orig)
    img_disp = cv2.resize(img_with_box, (target_w, h_orig))
    
    plate_disp_w = target_w // 2
    scale_p1 = plate_disp_w / plate_raw_bgr.shape[1]
    h_p1 = int(plate_raw_bgr.shape[0] * scale_p1)
    plate_raw_disp = cv2.resize(plate_raw_bgr, (plate_disp_w, h_p1))
    
    scale_p2 = plate_disp_w / recog_vis_bgr.shape[1]
    h_p2 = int(recog_vis_bgr.shape[0] * scale_p2)
    plate_recog_disp = cv2.resize(recog_vis_bgr, (plate_disp_w, h_p2))
    
    row2_h = max(h_p1, h_p2)
    text_h = 120
    
    total_h = h_orig + row2_h + text_h
    canvas = np.zeros((total_h, target_w, 3), dtype=np.uint8)
    
    # 图像拼接
    canvas[:h_orig, :] = img_disp
    canvas[h_orig:h_orig+h_p1, :plate_disp_w] = plate_raw_disp
    canvas[h_orig:h_orig+h_p2, plate_disp_w:] = plate_recog_disp
    canvas[h_orig+row2_h:, :] = (255, 255, 255) 
    
    text = f"Result: {result_text}"
    
    # 绘制文本
    canvas_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas_pil)
    
    # 加载中文字体
    font_path = "C:/Windows/Fonts/simhei.ttf"
    if not os.path.exists(font_path):
        font_path = "C:/Windows/Fonts/msyh.ttc"
        
    try:
        font = ImageFont.truetype(font_path, 60)
    except:
        print("中文字体加载失败，使用默认字体")
        font = ImageFont.load_default()
    
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        fw, fh = right - left, bottom - top
    except AttributeError:
        fw, fh = draw.textsize(text, font=font)
    
    tx = (target_w - fw) // 2
    ty = h_orig + row2_h + (text_h - fh) // 2
    
    draw.text((tx, ty), text, font=font, fill=(0, 0, 0))
    canvas = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(str(save_path), canvas)

def process_single_image(img_path: Path, input_root: Path, output_root: Path, recognizer: CharacterRecognizer):

    try:
        rel_path = img_path.relative_to(input_root)
    except ValueError:
        rel_path = img_path.name
        
    output_dir = output_root / rel_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"正在处理: {img_path.name}")
    
    temp_debug_dir = output_dir / f"temp_{img_path.stem}"
    if temp_debug_dir.exists():
        shutil.rmtree(temp_debug_dir)
    temp_debug_dir.mkdir()
    
    try:
        img_np = np.fromfile(str(img_path), dtype=np.uint8)
        image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        if image is None:
            print(f"图像解码失败: {img_path}")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. 车牌定位
        plate_result = locate_plate_region(gray, debug=False, recognizer=recognizer)
        
        if not plate_result:
            print(f"未在图片中找到车牌区域: {img_path.name}")
            return (img_path.name, "Not Found", 0.0)

        if plate_result.get('rotation_angle', 0.0) != 0.0:
            angle = plate_result['rotation_angle']
            print(f"应用倾斜矫正: {angle:.1f} 度")
            image = rotate_image(image, angle)
            
        bbox = plate_result['bbox']
        plate_img_gray = plate_result['image']

        # 2. 字符识别
        plate_number, confidence = recognizer.recognize(
            plate_image=plate_img_gray,
            debug_output_dir=temp_debug_dir,
            configs=CHAR_CONFIGS,
            ratio_bottom=RATIO_BOTTOM,
            ratio_top=RATIO_TOP,
            return_confidence=True
        )
        print(f"识别结果: {plate_number} (置信度: {confidence:.4f})")
        
        recog_vis_path = temp_debug_dir / "plate_recog_debug_vis.png"
        if recog_vis_path.exists():
            recog_vis_np = np.fromfile(str(recog_vis_path), dtype=np.uint8)
            recog_vis = cv2.imdecode(recog_vis_np, cv2.IMREAD_COLOR)
        else:
            recog_vis = plate_img_gray
            
        # 3. 生成并保存结果图
        save_path = output_dir / f"{img_path.stem}_result.jpg"
        create_composite_result(image, bbox, plate_img_gray, recog_vis, plate_number, save_path)
        print(f"结果已保存至: {save_path}")
        
        return (img_path.name, plate_number, confidence)

    except Exception as e:
        print(f"处理 {img_path.name} 时发生异常: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        if temp_debug_dir.exists():
            try:
                shutil.rmtree(temp_debug_dir)
            except:
                pass

def main():
    if not INPUT_ROOT_DIR.exists():
        print(f"输入目录不存在: {INPUT_ROOT_DIR}")
        return

    if OUTPUT_ROOT_DIR.exists():
        shutil.rmtree(OUTPUT_ROOT_DIR)
    OUTPUT_ROOT_DIR.mkdir()
    
    print("批量车牌识别任务开始")
    print(f"输入目录: {INPUT_ROOT_DIR.absolute()}")
    print(f"输出目录: {OUTPUT_ROOT_DIR.absolute()}")
    
    recognizer = CharacterRecognizer(template_dir="char_templates")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_results = []
    
    for img_path in INPUT_ROOT_DIR.rglob("*"):
        if img_path.suffix.lower() in image_extensions:
            res = process_single_image(img_path, INPUT_ROOT_DIR, OUTPUT_ROOT_DIR, recognizer)
            if res:
                all_results.append(res)
            
    # 保存结果汇总
    results_txt_path = OUTPUT_ROOT_DIR / "recognition_results.txt"
    print("最终识别结果汇总")
    
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write("File Name\tPlate Number\tConfidence\n")
        for filename, plate_num, conf in all_results:
            line = f"{filename:<30} | {plate_num:<15} | {conf:.4f}"
            print(line)
            f.write(f"{filename}\t{plate_num}\t{conf:.4f}\n")
            
    print(f"任务完成。共处理 {len(all_results)} 张图片。")
    print(f"详细结果已保存至: {OUTPUT_ROOT_DIR.absolute()}")

if __name__ == "__main__":
    main()
