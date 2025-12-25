"""
测试字符模板相对位置逻辑
验证汉字与字母数字模板的间距和中心点对齐要求
"""
import sys
from typing import List, Dict, Any

# 复制character_recognizer.py中的calculate_char_templates函数定义
def calculate_char_templates(plate_width: int) -> List[Dict[str, Any]]:
    """
    动态计算字符模板坐标，满足以下要求：
    1. 字母数字模板水平间距：图像宽度的3%-8%
    2. 汉字模板水平间距：字母数字间距的3倍
    3. 汉字中心点与字母数字中心点垂直对齐
    """
    # 字母数字区域：6个字符，水平间距为图像宽度的3%-8%
    char_spacing = int(plate_width * 0.05)  # 取中间值5%
    char_spacing = max(int(plate_width * 0.03), min(char_spacing, int(plate_width * 0.08)))
    
    # 汉字区域：2个字符，水平间距为字母数字间距的3倍
    chinese_spacing = char_spacing * 3
    
    # 计算字母数字区域的起始位置（确保居中）
    total_alphanum_width = 6 * 40 + 5 * char_spacing  # 6个字符 + 5个间距
    alphanum_start_x = (plate_width - total_alphanum_width) // 2
    
    # 计算汉字区域的起始位置（确保汉字中心点与字母数字中心点垂直对齐）
    total_chinese_width = 2 * 50 + chinese_spacing  # 2个汉字 + 1个间距
    chinese_start_x = (plate_width - total_chinese_width) // 2
    
    # 所有字符垂直对齐（y坐标相同）
    char_y = 150
    
    templates = []
    
    # 汉字区域模板
    templates.append({'name': '汉字1', 'x': chinese_start_x, 'y': char_y, 'w': 50, 'h': 70})
    templates.append({'name': '汉字2', 'x': chinese_start_x + 50 + chinese_spacing, 'y': char_y, 'w': 50, 'h': 70})
    
    # 字母数字区域模板
    for i in range(6):
        x_pos = alphanum_start_x + i * (40 + char_spacing)
        templates.append({'name': f'字符{i+3}', 'x': x_pos, 'y': char_y, 'w': 40, 'h': 60})
    
    return templates

def test_template_relative_position():
    """测试模板相对位置逻辑"""
    
    print("🔍 开始测试字符模板相对位置逻辑...")
    print("=" * 60)
    
    # 测试不同尺寸的车牌宽度
    test_widths = [300, 400, 500, 600]
    
    for width in test_widths:
        print(f"\n📏 测试车牌宽度: {width} 像素")
        print("-" * 40)
        
        # 计算字符模板
        templates = calculate_char_templates(width)
        
        # 提取汉字和字母数字模板
        chinese_templates = [t for t in templates if '汉字' in t['name']]
        alphanum_templates = [t for t in templates if '字符' in t['name']]
        
        # 计算字母数字间距
        alphanum_positions = [t['x'] for t in alphanum_templates]
        alphanum_spacings = [alphanum_positions[i+1] - alphanum_positions[i] - 40 for i in range(5)]
        avg_alphanum_spacing = sum(alphanum_spacings) / len(alphanum_spacings)
        
        # 计算汉字间距
        chinese_positions = [t['x'] for t in chinese_templates]
        chinese_spacing = chinese_positions[1] - chinese_positions[0] - 50
        
        # 计算中心点对齐
        chinese_center_x = (chinese_positions[0] + chinese_positions[1] + 50) / 2
        alphanum_center_x = (alphanum_positions[0] + alphanum_positions[-1] + 40) / 2
        center_alignment = abs(chinese_center_x - alphanum_center_x)
        
        # 计算相对比例
        spacing_ratio = avg_alphanum_spacing / width  # 字母数字间距占图像宽度的比例
        chinese_ratio = chinese_spacing / avg_alphanum_spacing  # 汉字间距与字母数字间距的比例
        
        # 输出详细结果
        print(f"📊 字母数字间距: {avg_alphanum_spacing:.1f} 像素")
        print(f"   → 占图像宽度: {spacing_ratio*100:.1f}%")
        print(f"📊 汉字间距: {chinese_spacing:.1f} 像素")
        print(f"   → 是字母数字间距的 {chinese_ratio:.1f} 倍")
        print(f"🎯 中心点对齐误差: {center_alignment:.1f} 像素")
        
        # 验证间距要求
        print("\n✅ 验证结果:")
        
        # 验证字母数字间距 (3%-8%)
        if 0.03 <= spacing_ratio <= 0.08:
            print(f"   ✓ 字母数字间距符合要求 (3%-8%)")
        else:
            print(f"   ✗ 字母数字间距不符合要求: {spacing_ratio*100:.1f}%")
        
        # 验证汉字间距 (约3倍)
        if 2.5 <= chinese_ratio <= 3.5:
            print(f"   ✓ 汉字间距符合要求 (约3倍字母数字间距)")
        else:
            print(f"   ✗ 汉字间距不符合要求: {chinese_ratio:.1f}倍")
        
        # 验证中心点对齐
        if center_alignment <= 5:
            print(f"   ✓ 中心点垂直对齐良好")
        else:
            print(f"   ✗ 中心点对齐误差较大: {center_alignment:.1f}像素")
        
        # 显示模板坐标详情
        print(f"\n📋 模板坐标详情:")
        print(f"   汉字1: x={chinese_templates[0]['x']}, y={chinese_templates[0]['y']}")
        print(f"   汉字2: x={chinese_templates[1]['x']}, y={chinese_templates[1]['y']}")
        print(f"   字母数字: x范围[{alphanum_positions[0]}-{alphanum_positions[-1]}]")
        print(f"   汉字中心点: {chinese_center_x:.1f}")
        print(f"   字母数字中心点: {alphanum_center_x:.1f}")
    
    print("\n" + "=" * 60)
    print("🎉 字符模板相对位置逻辑测试完成！")

def visualize_template_layout():
    """可视化模板布局"""
    
    print("\n🎨 可视化模板布局 (以400像素宽度为例)...")
    
    width = 400
    templates = calculate_char_templates(width)
    
    # 创建简单的ASCII可视化
    print(" " * 20 + "汉字区域")
    print(" " * 20 + "↓")
    
    # 绘制水平标尺
    scale_line = ""
    for i in range(0, width, 50):
        scale_line += f"{i:3d}"
    print(scale_line)
    
    # 绘制字符位置
    position_line = ['.'] * width
    
    for template in templates:
        x = template['x']
        w = template['w']
        
        # 标记字符位置
        for i in range(x, min(x + w, width)):
            if i < len(position_line):
                if '汉字' in template['name']:
                    position_line[i] = '中'
                else:
                    position_line[i] = '字'
    
    print(''.join(position_line))
    
    # 显示间距信息
    chinese_templates = [t for t in templates if '汉字' in t['name']]
    alphanum_templates = [t for t in templates if '字符' in t['name']]
    
    chinese_positions = [t['x'] for t in chinese_templates]
    alphanum_positions = [t['x'] for t in alphanum_templates]
    
    chinese_spacing = chinese_positions[1] - chinese_positions[0] - 50
    alphanum_spacing = alphanum_positions[1] - alphanum_positions[0] - 40
    
    print(f"\n📏 实际间距:")
    print(f"   汉字间距: {chinese_spacing} 像素")
    print(f"   字母数字间距: {alphanum_spacing} 像素")
    print(f"   比例关系: {chinese_spacing/alphanum_spacing:.2f}倍")

if __name__ == "__main__":
    test_template_relative_position()
    visualize_template_layout()