"""
电动车牌字符分割专属函数测试
"""
import cv2
import numpy as np
import sys
from pathlib import Path

# 添加backend目录到Python路径
backend_dir = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_dir))

from backend.character_recognizer import segment_characters

def test_electric_plate_segment():
    """测试电动车牌字符分割函数"""
    
    print("🔍 开始测试电动车牌字符分割专属函数...")
    
    # 创建一个模拟电动车牌图像（基于标准化模板）
    # 电动车牌尺寸：183×362像素
    plate_width = 183
    plate_height = 362
    
    # 创建白色背景的车牌图像
    plate_image = np.ones((plate_height, plate_width), dtype=np.uint8) * 255
    
    # 电动车牌字符模板坐标
    char_templates = [
        # 汉字区域（"广""州"）
        {'name': '汉字1（广）', 'x': 60, 'y': 46, 'w': 45, 'h': 60},
        {'name': '汉字2（州）', 'x': 99, 'y': 46, 'w': 45, 'h': 60},
        # 字母数字区域（P53283）
        {'name': '字母1（P）', 'x': 35, 'y': 172, 'w': 45, 'h': 60},
        {'name': '数字1（5）', 'x': 80, 'y': 172, 'w': 45, 'h': 60},
        {'name': '数字2（3）', 'x': 125, 'y': 172, 'w': 45, 'h': 60},
        {'name': '数字3（2）', 'x': 170, 'y': 172, 'w': 45, 'h': 60},
        {'name': '数字4（8）', 'x': 215, 'y': 172, 'w': 45, 'h': 60},
        {'name': '数字5（3）', 'x': 260, 'y': 172, 'w': 45, 'h': 60}
    ]
    
    # 在车牌图像上添加模拟字符（黑色矩形）
    for template in char_templates:
        x, y, w, h = template['x'], template['y'], template['w'], template['h']
        plate_image[y:y+h, x:x+w] = 0  # 黑色字符
    
    # 创建车牌区域字典
    plate_region = {
        'image': plate_image,
        'bbox': (0, 0, plate_width, plate_height)
    }
    
    print(f"✅ 创建模拟电动车牌图像，尺寸: {plate_height}x{plate_width}")
    
    # 测试字符分割函数（启用调试模式）
    print("\n🔍 调用电动车牌字符分割专属函数...")
    characters = segment_characters(plate_region, debug=True)
    
    print(f"\n✅ 字符分割完成，返回 {len(characters)} 个字符")
    
    # 验证分割结果
    if len(characters) == 8:
        print("🎉 成功分割出8个字符（符合电动车牌标准）")
        
        # 显示每个字符的信息
        for i, char_info in enumerate(characters):
            char_img = char_info['image']
            position = char_info['position']
            print(f"  字符{i+1}: 位置{position}, 尺寸{char_img.shape}")
    else:
        print(f"❌ 字符数量异常，期望8个，实际{len(characters)}个")
    
    # 测试边界情况：稍微偏移的车牌图像
    print("\n🔍 测试边界情况：偏移车牌图像...")
    
    # 创建一个偏移的车牌图像
    offset_plate = np.roll(plate_image, 20, axis=1)  # 水平偏移20像素
    offset_plate[:, :20] = 255  # 填充空白区域
    
    offset_region = {
        'image': offset_plate,
        'bbox': (0, 0, plate_width, plate_height)
    }
    
    offset_characters = segment_characters(offset_region, debug=False)
    print(f"偏移车牌分割结果: {len(offset_characters)} 个字符")
    
    # 测试不同尺寸的车牌图像
    print("\n🔍 测试不同尺寸的车牌图像...")
    
    # 放大车牌图像
    scaled_plate = cv2.resize(plate_image, (366, 724))  # 放大2倍
    scaled_region = {
        'image': scaled_plate,
        'bbox': (0, 0, 366, 724)
    }
    
    scaled_characters = segment_characters(scaled_region, debug=False)
    print(f"放大车牌分割结果: {len(scaled_characters)} 个字符")
    
    print("\n🎉 电动车牌字符分割专属函数测试完成！")

if __name__ == "__main__":
    test_electric_plate_segment()