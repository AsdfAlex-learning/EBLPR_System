"""
模板裁剪脚本 (Template Trimming Script)

功能：
    遍历 `char_templates` 目录下的所有字符模板图片。
    对每一张模板进行以下处理：
    1. 读取灰度图并二值化 (黑底白字)。
    2. 查找最大连通域 (即字符本身)。
    3. 获取该连通域的最小外接矩形 (Bounding Rect)。
    4. 根据外接矩形裁剪图片，去除多余边框。
    5. 统一缩放到标准尺寸 (40x60) 以匹配 character_recognizer.py 的要求。
    6. 保存覆盖原文件 (或保存到新目录)。

使用方法：
    直接运行此脚本。
"""

import cv2
import numpy as np
from pathlib import Path
import shutil

TEMPLATE_DIR = Path("char_templates")
BACKUP_DIR = Path("char_templates_backup")
TARGET_SIZE = (40, 60) # 宽 40, 高 60

def trim_templates():
    if not TEMPLATE_DIR.exists():
        print(f"[Error] 模板目录不存在: {TEMPLATE_DIR}")
        return

    # 1. 备份原模板
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    shutil.copytree(TEMPLATE_DIR, BACKUP_DIR)
    print(f"[*] 已备份原模板至: {BACKUP_DIR}")

    files = list(TEMPLATE_DIR.glob("*.png"))
    print(f"[*] 找到 {len(files)} 个模板文件，开始处理...")

    count = 0
    for file_path in files:
        if file_path.name == "template_preview.png":
            continue
            
        # 读取
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # 二值化 (确保黑底白字)
        # 假设模板已经是比较干净的，但为了保险起见，做一次阈值处理
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # 检查是否需要反转 (如果是白底黑字)
        h, w = binary.shape
        border_mean = (np.mean(binary[0, :]) + np.mean(binary[h-1, :]) + 
                       np.mean(binary[:, 0]) + np.mean(binary[:, w-1])) / 4.0
        if border_mean > 127:
            binary = cv2.bitwise_not(binary)
            
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"  [Warn] {file_path.name}: 未找到轮廓，跳过")
            continue
            
        # 找到最大轮廓 (假设是字符)
        max_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_cnt)
        
        # 裁剪
        cropped = binary[y:y+h, x:x+w]
        
        # 缩放到标准尺寸 (40x60)
        # 注意：这里直接 resize 会改变宽高比，但对于模板匹配来说，
        # 如果输入字符也是被 resize 到 40x60 的，那么模板也应该是 40x60。
        # 之前的 character_recognizer.py 中输入字符是被强制 resize 到 (40, 60) 的。
        # 所以这里我们也强制 resize。
        resized = cv2.resize(cropped, TARGET_SIZE)
        
        # 保存覆盖
        cv2.imwrite(str(file_path), resized)
        count += 1
        
    print(f"[Done] 处理完成！共裁剪并重置 {count} 个模板。")

if __name__ == "__main__":
    trim_templates()
