"""
字符分割功能测试脚本
使用01_Test_locate_plate.py的输出结果进行测试
"""
import cv2
import numpy as np
import sys
from pathlib import Path

# 添加backend目录到Python路径
backend_dir = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_dir))

from backend.character_recognizer import segment_characters, load_char_templates
from backend.image_utils import preprocess_image

def test_segment_characters():
    """测试字符分割功能"""
    
    # 测试结果目录
    test_results_dir = Path('test_results')
    segment_results_dir = Path('test_results/test_segment')
    segment_results_dir.mkdir(exist_ok=True)
    
    # 检查测试目录是否存在
    if not test_results_dir.exists():
        print(f"❌ 错误: 测试结果目录 {test_results_dir} 不存在")
        print("请先运行01_Test_locate_plate.py生成测试结果")
        return
    
    # 获取所有车牌区域图像
    plate_region_files = list(test_results_dir.glob('*_plate_region.png'))
    
    if not plate_region_files:
        print(f"❌ 错误: 在 {test_results_dir} 中未找到车牌区域图像")
        print("请先运行01_Test_locate_plate.py生成测试结果")
        return
    
    print(f"✅ 找到 {len(plate_region_files)} 个车牌区域图像")
    print("=" * 60)
    
    # 测试结果统计
    total_tests = 0
    successful_segmentations = 0
    
    for plate_file in plate_region_files:
        total_tests += 1
        print(f"\n🔍 测试图像 {total_tests}: {plate_file.name}")
        
        # 读取车牌区域图像
        plate_image = cv2.imread(str(plate_file), cv2.IMREAD_GRAYSCALE)
        if plate_image is None:
            print(f"   ❌ 无法读取图像 {plate_file}")
            continue
        
        print(f"   车牌区域尺寸: {plate_image.shape}")
        print(f"   像素范围: {plate_image.min()}-{plate_image.max()}")
        
        # 创建车牌区域字典（模拟plate_detector的输出格式）
        plate_region = {
            'image': plate_image,
            'bbox': (0, 0, plate_image.shape[1], plate_image.shape[0])
        }
        
        try:
            # 分割字符（启用调试模式）
            print(f"   🔍 开始字符分割...")
            print(f"   🔍 调用 segment_characters 函数...")
            
            # 检查函数是否存在
            if 'segment_characters' in globals():
                print(f"   🔍 segment_characters 函数存在")
            else:
                print(f"   ❌ segment_characters 函数不存在")
                continue
                
            # 添加详细的调试信息
            print(f"   🔍 输入plate_region类型: {type(plate_region)}")
            print(f"   🔍 输入plate_region内容: {list(plate_region.keys())}")
            print(f"   🔍 图像尺寸: {plate_region['image'].shape}")
            
            characters = segment_characters(plate_region, debug=True)
            print(f"   🔍 字符分割完成，返回 {len(characters) if characters else 0} 个字符")
            
            if characters:
                successful_segmentations += 1
                print(f"   ✅ 成功分割出 {len(characters)} 个字符")
                
                # 显示每个字符的信息
                for i, char_info in enumerate(characters):
                    char_img = char_info['image']
                    position = char_info['position']
                    print(f"     字符{i}: 位置{position}, 尺寸{char_img.shape}")
                
                # 保存分割结果
                save_segmentation_results(plate_file, plate_image, characters, segment_results_dir)
                
            else:
                print("   ❌ 字符分割失败，未分割出任何字符")
                
        except Exception as e:
            print(f"   ❌ 字符分割失败: {e}")
        
        print("-" * 40)
    
    # 测试总结
    print("\n" + "=" * 60)
    print("📊 测试总结:")
    print(f"   总测试图像数: {total_tests}")
    print(f"   成功分割数: {successful_segmentations}")
    print(f"   成功率: {successful_segmentations/total_tests*100:.1f}%")
    
    if successful_segmentations > 0:
        print(f"\n✅ 详细结果已保存到 {segment_results_dir} 目录:")
        print("   - 字符分割可视化图像 (*_segmentation.png)")
        print("   - 单个字符图像 (*_char_*.png)")
        print("   - 字符位置信息 (*_positions.txt)")
    
    print("\n🎉 字符分割测试完成！")

def save_segmentation_results(plate_file, plate_image, characters, output_dir):
    """保存字符分割结果"""
    
    # 生成文件名前缀
    prefix = plate_file.stem.replace('_plate_region', '')
    
    # 1. 创建字符分割可视化图像
    vis_image = create_segmentation_visualization(plate_image, characters)
    vis_path = output_dir / f"{prefix}_segmentation.png"
    cv2.imwrite(str(vis_path), vis_image)
    print(f"   📁 分割可视化保存到: {vis_path}")
    
    # 2. 保存每个字符的单独图像
    char_dir = output_dir / f"{prefix}_characters"
    char_dir.mkdir(exist_ok=True)
    
    for i, char_info in enumerate(characters):
        char_img = char_info['image']
        char_path = char_dir / f"char_{i}.png"
        cv2.imwrite(str(char_path), char_img)
    
    print(f"   📁 单个字符保存到: {char_dir}")
    
    # 3. 保存字符位置信息
    pos_path = output_dir / f"{prefix}_positions.txt"
    with open(pos_path, 'w', encoding='utf-8') as f:
        f.write(f"车牌图像: {plate_file.name}\n")
        f.write(f"分割字符数: {len(characters)}\n\n")
        
        for i, char_info in enumerate(characters):
            position = char_info['position']
            f.write(f"字符{i}: x={position[0]}, y={position[1]}, w={position[2]}, h={position[3]}\n")
    
    print(f"   📁 位置信息保存到: {pos_path}")

def create_segmentation_visualization(plate_image, characters):
    """创建字符分割可视化图像"""
    
    # 转换为彩色图像用于可视化
    if len(plate_image.shape) == 2:
        vis_image = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = plate_image.copy()
    
    # 绘制字符边界框
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (255, 255, 255)]
    
    for i, char_info in enumerate(characters):
        x, y, w, h = char_info['position']
        color = colors[i % len(colors)]
        
        # 绘制边界框
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        
        # 添加字符编号
        cv2.putText(vis_image, str(i), (x + 5, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return vis_image

def analyze_segmentation_quality():
    """分析字符分割质量"""
    
    segment_results_dir = Path('test_results/test_segment')
    
    if not segment_results_dir.exists():
        print("❌ 错误: test_segment 目录不存在，请先运行测试")
        return
    
    # 获取所有分割结果文件
    segmentation_files = list(segment_results_dir.glob('*_segmentation.png'))
    
    if not segmentation_files:
        print("未找到分割结果图像")
        return
    
    print(f"\n📊 分析 {len(segmentation_files)} 个分割结果:")
    print("=" * 60)
    
    for seg_file in segmentation_files:
        # 读取位置信息文件
        pos_file = segment_results_dir / f"{seg_file.stem.replace('_segmentation', '')}_positions.txt"
        
        if pos_file.exists():
            with open(pos_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"\n{seg_file.name}:")
                print(content)
                
                # 分析字符分布
                lines = content.strip().split('\n')
                if len(lines) > 2:
                    char_count = int(lines[1].split(': ')[1])
                    print(f"  字符数量: {char_count}")
                    
                    # 检查字符分布是否均匀
                    positions = []
                    for line in lines[3:]:
                        if line.startswith('字符'):
                            parts = line.split(': ')[1].split(', ')
                            x = int(parts[0].split('=')[1])
                            w = int(parts[2].split('=')[1])
                            positions.append((x, w))
                    
                    if len(positions) >= 2:
                        # 计算字符间距
                        spacings = []
                        for i in range(1, len(positions)):
                            prev_end = positions[i-1][0] + positions[i-1][1]
                            curr_start = positions[i][0]
                            spacing = curr_start - prev_end
                            spacings.append(spacing)
                        
                        if spacings:
                            avg_spacing = np.mean(spacings)
                            std_spacing = np.std(spacings)
                            print(f"  平均字符间距: {avg_spacing:.1f} 像素")
                            print(f"  间距标准差: {std_spacing:.1f} 像素")
                            
                            if std_spacing < 5:
                                print("  ✓ 字符分布均匀")
                            else:
                                print("  ⚠ 字符分布不均匀")

if __name__ == "__main__":
    print("🔍 字符分割功能测试")
    print("=" * 60)
    
    # 运行字符分割测试
    test_segment_characters()
    
    # 分析分割质量
    analyze_segmentation_quality()
    
    print("\n🎉 测试完成！")