"""
车牌定位功能测试脚本
测试车牌定位算法在正常车牌图像上的表现
"""
import cv2
import numpy as np
import os
import sys
import shutil
from pathlib import Path

# 添加backend目录到Python路径
# backend_dir = Path(__file__).parent / 'backend'
# sys.path.insert(0, str(backend_dir))

# 直接导入需要的模块，避免相对导入问题
import plate_detector as plate_detector

def test_locate_plate():
    """测试车牌定位功能"""
    
    # 测试图像目录 (修改为 input_images 根目录，包含所有子文件夹)
    test_dir = Path('input_images')
    
    # 结果保存目录
    result_dir = Path('test_results')
    if result_dir.exists():
        try:
            shutil.rmtree(result_dir)
            print(f"已清除旧的测试结果: {result_dir}")
        except Exception as e:
            print(f"清除旧结果失败: {e}")
            
    result_dir.mkdir(exist_ok=True)
    
    # 检查测试目录是否存在
    if not test_dir.exists():
        print(f"错误: 测试目录 {test_dir} 不存在")
        return
    
    # 获取所有测试图像 (递归查找)
    # image_files = list(test_dir.rglob('*.png')) + list(test_dir.rglob('*.jpg'))
    # 优先测试 normal 图像
    image_files = list(test_dir.rglob('normal_plate_*.jpg'))
    if not image_files:
         image_files = list(test_dir.rglob('*.png')) + list(test_dir.rglob('*.jpg'))
    
    if not image_files:
        print(f"错误: 在 {test_dir} 中未找到测试图像")
        return
    
    print(f"找到 {len(image_files)} 张测试图像")
    print("=" * 60)
    
    # 测试结果统计
    total_tests = 0
    successful_detections = 0
    
    for image_path in image_files:
        total_tests += 1
        print(f"\n测试图像 {total_tests}: {image_path.name}")
        
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  错误: 无法读取图像 {image_path}")
            continue
        
        # 创建调试目录
        debug_dir = result_dir / f"{image_path.stem}_debug"
        debug_dir.mkdir(exist_ok=True)
        
        print(f"  原始图像尺寸: {image.shape}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 定位车牌区域
        try:
            # 传递 debug 参数
            plate_info = plate_detector.locate_plate_region(gray, debug=True, debug_dir=debug_dir)
            bbox = plate_info['bbox']
            plate_image = plate_info['image']
            
            print(f"  车牌区域: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
            
            # 计算并打印中心偏移度
            img_h, img_w = image.shape[:2]
            plate_cx = bbox[0] + bbox[2] / 2
            plate_cy = bbox[1] + bbox[3] / 2
            img_cx, img_cy = img_w / 2, img_h / 2
            offset_x = abs(plate_cx - img_cx)
            offset_y = abs(plate_cy - img_cy)
            offset_dist = np.sqrt(offset_x**2 + offset_y**2)
            diag_len = np.sqrt(img_w**2 + img_h**2)
            offset_ratio = offset_dist / (diag_len / 2)
            
            print(f"  中心偏移: {offset_dist:.1f} 像素 (偏离率 {offset_ratio:.1%})")
            if offset_ratio > 0.3:
                print("  [警告] 偏离图像中心过大！")
                
            print(f"  车牌区域尺寸: {plate_image.shape}")
            print(f"  车牌区域像素范围: {plate_image.min()}-{plate_image.max()}")
            
            # 检查车牌区域是否合理
            if plate_image.size > 0 and plate_image.shape[0] > 0 and plate_image.shape[1] > 0:
                successful_detections += 1
                print("  ✓ 车牌定位成功")
                
                # 在原图上绘制车牌边界框
                image_with_bbox = image.copy()
                x, y, w, h = bbox
                cv2.rectangle(image_with_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 保存结果
                result_prefix = image_path.stem
                
                # 保存带边界框的原图
                bbox_path = result_dir / f"{result_prefix}_with_bbox.png"
                cv2.imwrite(str(bbox_path), image_with_bbox)
                print(f"  带边界框图像保存到: {bbox_path}")
                
                # 保存车牌区域
                plate_path = result_dir / f"{result_prefix}_plate_region.png"
                cv2.imwrite(str(plate_path), plate_image)
                print(f"  车牌区域保存到: {plate_path}")
                
            else:
                print("  ✗ 车牌区域无效")
                
        except Exception as e:
            print(f"  ✗ 车牌定位失败: {e}")
        
        print("-" * 40)
    
    # 输出测试总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"总测试图像数: {total_tests}")
    print(f"成功定位数: {successful_detections}")
    print(f"成功率: {successful_detections/total_tests*100:.1f}%")
    
    if successful_detections > 0:
        print("\n详细结果已保存到 test_results 目录:")
        print("- 带边界框的原图 (*_with_bbox.png)")
        print("- 车牌区域图像 (*_plate_region.png)")

def analyze_plate_regions():
    """分析已检测到的车牌区域"""
    
    result_dir = Path('test_results')
    
    if not result_dir.exists():
        print("错误: test_results 目录不存在，请先运行测试")
        return
    
    # 获取所有车牌区域图像
    plate_files = list(result_dir.glob('*_plate_region.png'))
    
    if not plate_files:
        print("未找到车牌区域图像")
        return
    
    print(f"\n分析 {len(plate_files)} 个车牌区域:")
    print("=" * 60)
    
    for plate_path in plate_files:
        plate_image = cv2.imread(str(plate_path), cv2.IMREAD_GRAYSCALE)
        
        if plate_image is not None:
            print(f"\n{plate_path.name}:")
            print(f"  尺寸: {plate_image.shape}")
            print(f"  像素范围: {plate_image.min()}-{plate_image.max()}")
            print(f"  平均亮度: {plate_image.mean():.1f}")
            print(f"  对比度: {plate_image.std():.1f}")
            
            # 计算非零像素比例
            non_zero_ratio = np.sum(plate_image > 0) / plate_image.size
            print(f"  非零像素比例: {non_zero_ratio:.3f}")
            
            # 评估图像质量
            if plate_image.shape[0] >= 30 and plate_image.shape[1] >= 80:
                print("  ✓ 尺寸合理")
            else:
                print("  ⚠ 尺寸可能偏小")
                
            if plate_image.std() > 30:
                print("  ✓ 对比度良好")
            else:
                print("  ⚠ 对比度可能不足")

if __name__ == "__main__":
    print("车牌定位功能测试")
    print("=" * 60)
    
    # 运行车牌定位测试
    test_locate_plate()
    
    # 分析检测结果
    analyze_plate_regions()
    
    print("\n测试完成！")