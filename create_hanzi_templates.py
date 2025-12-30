import cv2
import numpy as np
from pathlib import Path

def create_hanzi_templates():
    # 1. 设置路径
    input_path = Path(r"d:\Digital_Image_Processing\FinalProject\plate_recog\char_template_complete\hanzi.png")
    output_dir = Path(r"d:\Digital_Image_Processing\FinalProject\plate_recog\char_templates")
    
    if not input_path.exists():
        print(f"Error: 找不到输入图片 {input_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 读取图片
    img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: 无法读取图片")
        return

    # 3. 预处理 (二值化，确保黑底白字)
    # 车牌上的字符通常是：蓝底白字 -> 二值化后字是白，底是黑。
    # 我们的目标是统一为：黑底(0)，白字(255)
    
    # Otsu 二值化
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 智能反转检测：检查四个角，如果是白色(255)，说明是白底黑字，需要反转为黑底白字
    h, w = binary.shape
    corners = [binary[0,0], binary[0, w-1], binary[h-1, 0], binary[h-1, w-1]]
    if np.mean(corners) > 127:
        print("检测到白底，进行反转为黑底白字...")
        binary = cv2.bitwise_not(binary)
    else:
        print("检测到黑底，无需反转...")

    # 4. 分割 (4等份)
    # 广州佛山 -> 广, 州, 佛, 山
    char_names = ["guang", "zhou", "fo", "shan"]
    step = w // 4
    
    target_size = (40, 40) # 1:1 比例，大小约 40x40

    print(f"图片尺寸: {w}x{h}, 每个字符宽度约: {step}")

    for i, name in enumerate(char_names):
        # 切割
        x_start = i * step
        x_end = (i + 1) * step if i < 3 else w # 最后一个取到底
        
        char_img = binary[:, x_start:x_end]
        
        # 去除多余边框 (Find Contours to crop tight box)
        # 这一步可选，为了保证字符居中且大小合适
        coords = cv2.findNonZero(char_img)
        if coords is not None:
            x, y, cw, ch = cv2.boundingRect(coords)
            # 稍微留一点边距? 或者直接切
            char_crop = char_img[y:y+ch, x:x+cw]
            
            # Resize 到 1:1 (40x40)
            # 为了保持形状不畸变，可以先 pad 到方形，再 resize
            # 但用户要求 "按比例分割，并按照约1：1的width与height的去识别"
            # 如果原字就是方形的，直接 resize 即可。如果是长条的，强制 1:1 会变形。
            # 这里我们尝试保持比例 Pad 到方形
            ch_h, ch_w = char_crop.shape
            if ch_h > ch_w:
                # 高度大，补宽度
                pad_total = ch_h - ch_w
                pad_l = pad_total // 2
                pad_r = pad_total - pad_l
                char_square = cv2.copyMakeBorder(char_crop, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
            else:
                # 宽度大，补高度
                pad_total = ch_w - ch_h
                pad_t = pad_total // 2
                pad_b = pad_total - pad_t
                char_square = cv2.copyMakeBorder(char_crop, pad_t, pad_b, 0, 0, cv2.BORDER_CONSTANT, value=0)
                
            char_resized = cv2.resize(char_square, target_size, interpolation=cv2.INTER_AREA)
        else:
            print(f"Warning: 字符 {name} 是空的？")
            char_resized = cv2.resize(char_img, target_size)

        # 保存
        out_file = output_dir / f"{name}.png"
        cv2.imwrite(str(out_file), char_resized)
        print(f"Saved {name} to {out_file}")

if __name__ == "__main__":
    create_hanzi_templates()
