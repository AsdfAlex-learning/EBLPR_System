import cv2
import numpy as np
from pathlib import Path

def batch_convert_templates_to_black_bg():
    template_dir = Path("char_templates")
    if not template_dir.exists():
        print(f"Directory not found: {template_dir}")
        return

    print(f"[*] Processing templates in {template_dir}...")
    
    # 获取所有 png 文件
    files = list(template_dir.glob("*.png"))
    
    count = 0
    for file_path in files:
        if file_path.name == "template_preview.png":
            continue
            
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # 简单二值化 (Otsu)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 智能检测：如果边缘是白色的，说明是白底黑字，需要反转
        h, w = binary.shape
        # 采样边缘像素
        border_pixels = []
        border_pixels.extend(binary[0, :])       # Top
        border_pixels.extend(binary[h-1, :])     # Bottom
        border_pixels.extend(binary[:, 0])       # Left
        border_pixels.extend(binary[:, w-1])     # Right
        
        avg_border = np.mean(border_pixels)
        
        # 如果边缘平均值 > 127，说明背景是白色的，需要反转为黑底白字
        if avg_border > 127:
            # print(f"  -> Converting {file_path.name} from White-BG to Black-BG")
            binary = cv2.bitwise_not(binary)
            cv2.imwrite(str(file_path), binary)
            count += 1
        else:
            # 已经是黑底白字，或者无法确定，但也确保保存为二值化版本
            # print(f"  -> {file_path.name} is already Black-BG")
            cv2.imwrite(str(file_path), binary)
            
    print(f"[*] Batch conversion completed. Converted/Processed {count} files.")

if __name__ == "__main__":
    batch_convert_templates_to_black_bg()
