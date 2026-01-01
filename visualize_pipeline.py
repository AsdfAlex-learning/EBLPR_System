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
FLOWCHART_ROOT_DIR = Path("flowchart")

def create_pipeline_summary(save_path, source_img, plate_crop, plate_binary, plate_seg, final_overlay):
    """
    生成步骤汇总网格图。
    """
    # 设定网格布局
    # Row 1: Source (Large)
    # Row 2: Final Result (Large)
    # Row 3: Plate Crop | Binary | Segmentation (Three small ones)
    
    target_width = 1200
    
    # Resize helper
    def resize_width(img, w):
        h = int(img.shape[0] * (w / img.shape[1]))
        return cv2.resize(img, (w, h))
        
    def resize_height(img, h):
        w = int(img.shape[1] * (h / img.shape[0]))
        return cv2.resize(img, (w, h))

    # 1. 准备 Row 1 & 2
    img_source_disp = resize_width(source_img, target_width)
    img_final_disp = resize_width(final_overlay, target_width)
    
    # 2. 准备 Row 3 (三个小图并排)
    # 统一高度
    target_row3_h = 200
    
    # 确保是彩色图
    def to_bgr(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    p_crop = to_bgr(plate_crop)
    p_bin = to_bgr(plate_binary)
    p_seg = to_bgr(plate_seg)
    
    p_crop_r = resize_height(p_crop, target_row3_h)
    p_bin_r = resize_height(p_bin, target_row3_h)
    p_seg_r = resize_height(p_seg, target_row3_h)
    
    # 拼接 Row 3
    # 如果宽度不够 target_width，用白色填充；如果超过，缩放
    row3_content_w = p_crop_r.shape[1] + p_bin_r.shape[1] + p_seg_r.shape[1]
    
    row3 = np.ones((target_row3_h, row3_content_w, 3), dtype=np.uint8) * 255
    current_x = 0
    row3[:, current_x:current_x+p_crop_r.shape[1]] = p_crop_r
    current_x += p_crop_r.shape[1]
    row3[:, current_x:current_x+p_bin_r.shape[1]] = p_bin_r
    current_x += p_bin_r.shape[1]
    row3[:, current_x:current_x+p_seg_r.shape[1]] = p_seg_r
    
    # 调整 Row 3 到 target_width
    if row3_content_w != target_width:
        row3 = cv2.resize(row3, (target_width, int(target_row3_h * (target_width/row3_content_w))))

    # 3. 垂直拼接
    pad = 20
    total_h = img_source_disp.shape[0] + pad + row3.shape[0] + pad + img_final_disp.shape[0]
    canvas = np.ones((total_h, target_width, 3), dtype=np.uint8) * 255
    
    curr_y = 0
    canvas[curr_y:curr_y+img_source_disp.shape[0], :] = img_source_disp
    
    curr_y += img_source_disp.shape[0] + pad
    canvas[curr_y:curr_y+row3.shape[0], :] = row3
    
    curr_y += row3.shape[0] + pad
    canvas[curr_y:curr_y+img_final_disp.shape[0], :] = img_final_disp
    
    # 添加标签
    cv2.putText(canvas, "00. Source Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    y_row3 = img_source_disp.shape[0] + pad - 10
    cv2.putText(canvas, "10. Crop / 11. Binary / 12. Segmentation", (10, y_row3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    y_final = img_source_disp.shape[0] + pad + row3.shape[0] + pad - 10
    cv2.putText(canvas, "99. Final Result Overlay", (10, y_final), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imwrite(str(save_path), canvas)

def draw_overlay_result(image, bbox, text):
    """
    绘制最终结果叠加图。
    """
    x, y, w, h = bbox
    img_vis = image.copy()
    cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    # 使用 PIL 绘制中文
    img_pil = Image.fromarray(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 字体加载
    font_path = "C:/Windows/Fonts/simhei.ttf"
    if not os.path.exists(font_path):
        font_path = "C:/Windows/Fonts/msyh.ttc"
    
    try:
        font_size = max(30, int(h * 0.8))
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
        
    # 绘制文字背景
    text_str = f"{text}"
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text_str, font=font)
        tw, th = right - left, bottom - top
    except AttributeError:
        tw, th = draw.textsize(text_str, font=font)
        
    text_x = x
    text_y = y - th - 5 if y - th - 5 > 0 else y + h + 5
    
    draw.rectangle((text_x, text_y, text_x + tw + 10, text_y + th + 5), fill=(0, 255, 0))
    draw.text((text_x + 5, text_y), text_str, font=font, fill=(0, 0, 0))
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def process_pipeline_single(img_path, recognizer):
    print(f"[INFO] Pipeline processing: {img_path.name}")
    
    # 0. 准备输出目录
    flowchart_dir = FLOWCHART_ROOT_DIR / img_path.stem
    if flowchart_dir.exists():
        shutil.rmtree(flowchart_dir)
    flowchart_dir.mkdir(parents=True)
    
    chars_dir = flowchart_dir / "chars"
    chars_dir.mkdir()
    
    # 1. 读取原图
    img_np = np.fromfile(str(img_path), dtype=np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if image is None:
        print(f"[ERROR] 无法读取: {img_path}")
        return

    # 保存 00_source_image.jpg
    cv2.imwrite(str(flowchart_dir / "00_source_image.jpg"), image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 定位
    plate_res = locate_plate_region(gray, debug=False, recognizer=recognizer)
    if not plate_res:
        print(f"[WARN] 未找到车牌: {img_path.name}")
        return

    # 倾斜矫正
    if plate_res.get('rotation_angle', 0.0) != 0.0:
        angle = plate_res['rotation_angle']
        print(f"[INFO] 倾斜矫正: {angle:.1f}°")
        image = rotate_image(image, angle)
        # 重新定位一次以获取矫正后的精确坐标
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plate_res = locate_plate_region(gray, debug=False) 
        
    bbox = plate_res['bbox']
    plate_crop = plate_res['image']
    
    # 保存 10_plate_crop_input.png
    cv2.imwrite(str(flowchart_dir / "10_plate_crop_input.png"), plate_crop)
    
    # 3. 识别 (使用临时目录捕获中间文件)
    temp_debug = flowchart_dir / "temp_debug"
    temp_debug.mkdir()
    
    text, conf = recognizer.recognize(
        plate_image=plate_crop,
        debug_output_dir=temp_debug,
        return_confidence=True
    )
    
    print(f"[INFO] 识别结果: {text} ({conf:.2f})")
    
    # 4. 移动/重命名中间文件
    # 11_plate_binary_processed.png
    src_processed = temp_debug / "plate_processed.png"
    if src_processed.exists():
        shutil.copy(src_processed, flowchart_dir / "11_plate_binary_processed.png")
        
    # 12_char_segmentation_result.png
    src_vis = temp_debug / "plate_recog_debug_vis.png"
    if src_vis.exists():
        shutil.copy(src_vis, flowchart_dir / "12_char_segmentation_result.png")
        
    # chars/
    for cut_file in temp_debug.glob("cut_*.png"):
        shutil.copy(cut_file, chars_dir / cut_file.name)
        
    # 读取用于生成的图片
    img_binary = cv2.imread(str(flowchart_dir / "11_plate_binary_processed.png"))
    img_seg = cv2.imread(str(flowchart_dir / "12_char_segmentation_result.png"))
    
    # 清理临时目录
    try:
        shutil.rmtree(temp_debug)
    except:
        pass
        
    # 5. 生成 99_final_result_overlay.jpg
    final_overlay = draw_overlay_result(image, bbox, text)
    cv2.imwrite(str(flowchart_dir / "99_final_result_overlay.jpg"), final_overlay)
    
    # 6. 生成 99_pipeline_summary_grid.jpg
    create_pipeline_summary(
        flowchart_dir / "99_pipeline_summary_grid.jpg",
        image,
        plate_crop,
        img_binary if img_binary is not None else plate_crop, # Fallback
        img_seg if img_seg is not None else plate_crop,       # Fallback
        final_overlay
    )
    
    print(f"[INFO] 已生成可视化流程: {flowchart_dir}")

def main():
    if not INPUT_ROOT_DIR.exists():
        print(f"[ERROR] 输入目录不存在: {INPUT_ROOT_DIR}")
        return
        
    if FLOWCHART_ROOT_DIR.exists():
        shutil.rmtree(FLOWCHART_ROOT_DIR)
    FLOWCHART_ROOT_DIR.mkdir()
    
    recognizer = CharacterRecognizer(template_dir="char_templates")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    count = 0
    for img_path in INPUT_ROOT_DIR.rglob("*"):
        if img_path.suffix.lower() in image_extensions:
            process_pipeline_single(img_path, recognizer)
            count += 1
            
    print(f"\n[INFO] 处理完成，共生成 {count} 组可视化流程数据。")
    print(f"[INFO] 输出目录: {FLOWCHART_ROOT_DIR.absolute()}")

if __name__ == "__main__":
    main()
