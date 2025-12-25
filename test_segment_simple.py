"""
简单的字符分割测试
"""
import cv2
import numpy as np
import sys
from pathlib import Path

# 添加backend目录到Python路径
backend_dir = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_dir))

from backend.character_recognizer import segment_characters

def test_simple():
    """简单测试字符分割函数"""
    
    # 创建一个简单的测试图像
    test_image = np.ones((100, 500), dtype=np.uint8) * 255  # 白色背景
    
    # 添加一些黑色矩形模拟字符
    for i in range(7):
        x = 50 + i * 60
        y = 20
        w = 40
        h = 60
        test_image[y:y+h, x:x+w] = 0  # 黑色字符
    
    # 创建车牌区域字典
    plate_region = {
        'image': test_image,
        'bbox': (0, 0, test_image.shape[1], test_image.shape[0])
    }
    
    print("🔍 开始简单测试...")
    print(f"测试图像尺寸: {test_image.shape}")
    
    # 调用字符分割函数
    characters = segment_characters(plate_region, debug=True)
    
    print(f"🔍 字符分割完成，返回 {len(characters) if characters else 0} 个字符")
    
    if characters:
        print(f"✅ 成功分割出 {len(characters)} 个字符")
        
        # 显示每个字符的信息
        for i, char_info in enumerate(characters):
            char_img = char_info['image']
            position = char_info['position']
            print(f"  字符{i}: 位置{position}, 尺寸{char_img.shape}")
    else:
        print("❌ 字符分割失败，未分割出任何字符")

if __name__ == "__main__":
    test_simple()