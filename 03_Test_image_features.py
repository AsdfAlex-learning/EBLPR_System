"""
图像特征提取测试脚本
使用多种手段提取车牌图像特征，包括：
- 水平/垂直投影
- 二值化处理
- 边缘检测
- 轮廓分析
- 直方图分析
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def extract_image_features(image_path, output_dir):
    """提取图像特征并生成可视化结果"""
    
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    print(f"✅ 处理图像: {Path(image_path).name}")
    print(f"   图像尺寸: {image.shape}")
    print(f"   像素范围: {image.min()}-{image.max()}")
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 生成文件名前缀
    prefix = Path(image_path).stem
    
    # 1. 水平投影分析
    horizontal_projection = analyze_horizontal_projection(image, prefix, output_dir)
    
    # 2. 垂直投影分析
    vertical_projection = analyze_vertical_projection(image, prefix, output_dir)
    
    # 3. 二值化处理
    binary_images = analyze_binarization(image, prefix, output_dir)
    
    # 4. 边缘检测
    edge_images = analyze_edge_detection(image, prefix, output_dir)
    
    # 5. 轮廓分析
    contour_analysis = analyze_contours(image, prefix, output_dir)
    
    # 6. 直方图分析
    histogram_analysis = analyze_histogram(image, prefix, output_dir)
    
    # 7. 生成综合特征图
    create_comprehensive_feature_plot(
        image, horizontal_projection, vertical_projection, 
        binary_images, edge_images, prefix, output_dir
    )
    
    return {
        'horizontal_projection': horizontal_projection,
        'vertical_projection': vertical_projection,
        'binary_images': binary_images,
        'edge_images': edge_images,
        'contour_analysis': contour_analysis,
        'histogram_analysis': histogram_analysis
    }

def analyze_horizontal_projection(image, prefix, output_dir):
    """分析水平投影特征"""
    
    print(f"   📊 分析水平投影...")
    
    # 计算水平投影（每行的像素和）
    horizontal_proj = np.sum(image, axis=1)
    
    # 归一化投影值
    horizontal_proj_norm = horizontal_proj / horizontal_proj.max() * 255
    
    # 创建水平投影图
    proj_height = 200
    proj_width = image.shape[1]
    
    # 创建投影图像
    proj_image = np.zeros((proj_height, proj_width), dtype=np.uint8)
    
    for i, value in enumerate(horizontal_proj_norm):
        if i < proj_height:
            proj_image[i, :int(value * proj_width / 255)] = 255
    
    # 保存水平投影图
    proj_path = output_dir / f"{prefix}_horizontal_projection.png"
    cv2.imwrite(str(proj_path), proj_image)
    
    # 创建水平投影可视化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 原始图像
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'原始图像 - {prefix}')
    ax1.axis('off')
    
    # 水平投影图
    ax2.plot(horizontal_proj)
    ax2.set_title('水平投影（每行像素和）')
    ax2.set_xlabel('行索引')
    ax2.set_ylabel('像素和')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(str(output_dir / f"{prefix}_horizontal_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'projection': horizontal_proj,
        'image': proj_image,
        'max_value': horizontal_proj.max(),
        'min_value': horizontal_proj.min(),
        'mean_value': horizontal_proj.mean()
    }

def analyze_vertical_projection(image, prefix, output_dir):
    """分析垂直投影特征"""
    
    print(f"   📊 分析垂直投影...")
    
    # 计算垂直投影（每列的像素和）
    vertical_proj = np.sum(image, axis=0)
    
    # 归一化投影值
    vertical_proj_norm = vertical_proj / vertical_proj.max() * 255
    
    # 创建垂直投影图
    proj_height = 200
    proj_width = image.shape[0]
    
    # 创建投影图像
    proj_image = np.zeros((proj_width, proj_height), dtype=np.uint8)
    
    for i, value in enumerate(vertical_proj_norm):
        if i < proj_width:
            proj_image[i, :int(value * proj_height / 255)] = 255
    
    # 转置图像以正确显示
    proj_image = cv2.transpose(proj_image)
    
    # 保存垂直投影图
    proj_path = output_dir / f"{prefix}_vertical_projection.png"
    cv2.imwrite(str(proj_path), proj_image)
    
    # 创建垂直投影可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 原始图像
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'原始图像 - {prefix}')
    ax1.axis('off')
    
    # 垂直投影图
    ax2.plot(vertical_proj)
    ax2.set_title('垂直投影（每列像素和）')
    ax2.set_xlabel('列索引')
    ax2.set_ylabel('像素和')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(str(output_dir / f"{prefix}_vertical_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'projection': vertical_proj,
        'image': proj_image,
        'max_value': vertical_proj.max(),
        'min_value': vertical_proj.min(),
        'mean_value': vertical_proj.mean()
    }

def analyze_binarization(image, prefix, output_dir):
    """分析二值化处理"""
    
    print(f"   📊 分析二值化处理...")
    
    # 使用不同阈值进行二值化
    thresholds = [128, 150, 100, 200]
    binary_results = {}
    
    for i, threshold in enumerate(thresholds):
        # 二值化处理
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        
        # 保存二值化图像
        binary_path = output_dir / f"{prefix}_binary_{threshold}.png"
        cv2.imwrite(str(binary_path), binary)
        
        binary_results[f'threshold_{threshold}'] = {
            'image': binary,
            'white_pixels': np.sum(binary == 255),
            'black_pixels': np.sum(binary == 0),
            'white_ratio': np.sum(binary == 255) / binary.size
        }
    
    # 自适应阈值二值化
    adaptive_binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    adaptive_path = output_dir / f"{prefix}_binary_adaptive.png"
    cv2.imwrite(str(adaptive_path), adaptive_binary)
    
    binary_results['adaptive'] = {
        'image': adaptive_binary,
        'white_pixels': np.sum(adaptive_binary == 255),
        'black_pixels': np.sum(adaptive_binary == 0),
        'white_ratio': np.sum(adaptive_binary == 255) / adaptive_binary.size
    }
    
    # 创建二值化对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 原始图像
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 不同阈值二值化
    for i, (key, result) in enumerate(list(binary_results.items())[:4]):
        axes[i+1].imshow(result['image'], cmap='gray')
        white_pixels = result['white_pixels']
        axes[i+1].set_title(f'{key} (白像素: {white_pixels})')
        axes[i+1].axis('off')
    
    # 自适应二值化
    axes[5].imshow(binary_results['adaptive']['image'], cmap='gray')
    adaptive_white = binary_results['adaptive']['white_pixels']
    axes[5].set_title(f'自适应 (白像素: {adaptive_white})')
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(str(output_dir / f"{prefix}_binary_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    return binary_results

def analyze_edge_detection(image, prefix, output_dir):
    """分析边缘检测"""
    
    print(f"   📊 分析边缘检测...")
    
    edge_results = {}
    
    # Canny边缘检测（不同阈值）
    canny_low = cv2.Canny(image, 50, 150)
    canny_medium = cv2.Canny(image, 100, 200)
    canny_high = cv2.Canny(image, 150, 250)
    
    # 保存边缘检测结果
    cv2.imwrite(str(output_dir / f"{prefix}_canny_low.png"), canny_low)
    cv2.imwrite(str(output_dir / f"{prefix}_canny_medium.png"), canny_medium)
    cv2.imwrite(str(output_dir / f"{prefix}_canny_high.png"), canny_high)
    
    edge_results['canny_low'] = {
        'image': canny_low,
        'edge_pixels': np.sum(canny_low == 255)
    }
    edge_results['canny_medium'] = {
        'image': canny_medium,
        'edge_pixels': np.sum(canny_medium == 255)
    }
    edge_results['canny_high'] = {
        'image': canny_high,
        'edge_pixels': np.sum(canny_high == 255)
    }
    
    # Sobel边缘检测
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = np.uint8(sobel_combined)
    
    cv2.imwrite(str(output_dir / f"{prefix}_sobel.png"), sobel_combined)
    
    edge_results['sobel'] = {
        'image': sobel_combined,
        'edge_pixels': np.sum(sobel_combined > 50)  # 使用阈值统计边缘像素
    }
    
    # 创建边缘检测对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 原始图像
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # Canny边缘检测
    axes[1].imshow(canny_low, cmap='gray')
    canny_low_pixels = edge_results['canny_low']['edge_pixels']
    axes[1].set_title(f'Canny低阈值 (边缘: {canny_low_pixels})')
    axes[1].axis('off')
    
    axes[2].imshow(canny_medium, cmap='gray')
    canny_medium_pixels = edge_results['canny_medium']['edge_pixels']
    axes[2].set_title(f'Canny中阈值 (边缘: {canny_medium_pixels})')
    axes[2].axis('off')
    
    axes[3].imshow(canny_high, cmap='gray')
    canny_high_pixels = edge_results['canny_high']['edge_pixels']
    axes[3].set_title(f'Canny高阈值 (边缘: {canny_high_pixels})')
    axes[3].axis('off')
    
    # Sobel边缘检测
    axes[4].imshow(sobel_combined, cmap='gray')
    sobel_pixels = edge_results['sobel']['edge_pixels']
    axes[4].set_title(f'Sobel (边缘: {sobel_pixels})')
    axes[4].axis('off')
    
    # 组合边缘检测
    combined_edges = cv2.bitwise_or(canny_medium, sobel_combined)
    axes[5].imshow(combined_edges, cmap='gray')
    axes[5].set_title('Canny+Sobel组合')
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(str(output_dir / f"{prefix}_edge_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    return edge_results

def analyze_contours(image, prefix, output_dir):
    """分析轮廓特征"""
    
    print(f"   📊 分析轮廓特征...")
    
    # 二值化处理用于轮廓检测
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建轮廓可视化图像
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    # 保存轮廓图像
    contour_path = output_dir / f"{prefix}_contours.png"
    cv2.imwrite(str(contour_path), contour_image)
    
    # 分析轮廓特征
    contour_features = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area > 100:  # 只考虑较大的轮廓
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            contour_features.append({
                'index': i,
                'area': area,
                'perimeter': perimeter,
                'bounding_box': (x, y, w, h),
                'aspect_ratio': aspect_ratio
            })
    
    # 保存轮廓特征信息
    contour_info_path = output_dir / f"{prefix}_contour_info.txt"
    with open(contour_info_path, 'w', encoding='utf-8') as f:
        f.write(f"图像: {prefix}\n")
        f.write(f"检测到轮廓数: {len(contours)}\n")
        f.write(f"有效轮廓数(面积>100): {len(contour_features)}\n\n")
        
        for feature in contour_features:
            f.write(f"轮廓{feature['index']}: ")
            f.write(f"面积={feature['area']:.1f}, ")
            f.write(f"周长={feature['perimeter']:.1f}, ")
            f.write(f"宽高比={feature['aspect_ratio']:.2f}, ")
            f.write(f"边界框={feature['bounding_box']}\n")
    
    return {
        'contours': contours,
        'hierarchy': hierarchy,
        'features': contour_features,
        'image': contour_image
    }

def analyze_histogram(image, prefix, output_dir):
    """分析直方图特征"""
    
    print(f"   📊 分析直方图特征...")
    
    # 计算直方图
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # 创建直方图可视化
    plt.figure(figsize=(10, 6))
    plt.plot(hist)
    plt.title(f'灰度直方图 - {prefix}')
    plt.xlabel('像素值')
    plt.ylabel('频率')
    plt.grid(True)
    
    # 添加统计信息
    mean_val = np.mean(image)
    std_val = np.std(image)
    median_val = np.median(image)
    
    plt.axvline(mean_val, color='r', linestyle='--', label=f'均值: {mean_val:.1f}')
    plt.axvline(median_val, color='g', linestyle='--', label=f'中位数: {median_val:.1f}')
    plt.legend()
    
    plt.savefig(str(output_dir / f"{prefix}_histogram.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存统计信息
    stats_path = output_dir / f"{prefix}_histogram_stats.txt"
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"图像: {prefix}\n")
        f.write(f"像素统计信息:\n")
        f.write(f"  最小值: {image.min()}\n")
        f.write(f"  最大值: {image.max()}\n")
        f.write(f"  均值: {mean_val:.2f}\n")
        f.write(f"  标准差: {std_val:.2f}\n")
        f.write(f"  中位数: {median_val:.2f}\n")
        f.write(f"  像素总数: {image.size}\n")
    
    return {
        'histogram': hist,
        'mean': mean_val,
        'std': std_val,
        'median': median_val,
        'min': image.min(),
        'max': image.max()
    }

def create_comprehensive_feature_plot(image, horizontal_proj, vertical_proj, 
                                     binary_images, edge_images, prefix, output_dir):
    """创建综合特征图"""
    
    print(f"   📊 创建综合特征图...")
    
    # 创建大图
    fig = plt.figure(figsize=(20, 15))
    
    # 布局设置
    gs = fig.add_gridspec(4, 4)
    
    # 1. 原始图像
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title('原始图像')
    ax1.axis('off')
    
    # 2. 水平投影
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(horizontal_proj['projection'])
    ax2.set_title('水平投影')
    ax2.grid(True)
    
    # 3. 垂直投影
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(vertical_proj['projection'])
    ax3.set_title('垂直投影')
    ax3.grid(True)
    
    # 4. 最佳二值化结果
    ax4 = fig.add_subplot(gs[0, 3])
    best_binary = binary_images['threshold_128']['image']
    ax4.imshow(best_binary, cmap='gray')
    ax4.set_title('二值化(阈值128)')
    ax4.axis('off')
    
    # 5. 自适应二值化
    ax5 = fig.add_subplot(gs[1, 0])
    adaptive_binary = binary_images['adaptive']['image']
    ax5.imshow(adaptive_binary, cmap='gray')
    ax5.set_title('自适应二值化')
    ax5.axis('off')
    
    # 6. Canny边缘检测
    ax6 = fig.add_subplot(gs[1, 1])
    canny = edge_images['canny_medium']['image']
    ax6.imshow(canny, cmap='gray')
    ax6.set_title('Canny边缘检测')
    ax6.axis('off')
    
    # 7. Sobel边缘检测
    ax7 = fig.add_subplot(gs[1, 2])
    sobel = edge_images['sobel']['image']
    ax7.imshow(sobel, cmap='gray')
    ax7.set_title('Sobel边缘检测')
    ax7.axis('off')
    
    # 8. 直方图
    ax8 = fig.add_subplot(gs[1, 3])
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    ax8.plot(hist)
    ax8.set_title('灰度直方图')
    ax8.grid(True)
    
    # 9. 水平投影图像
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.imshow(horizontal_proj['image'], cmap='gray')
    ax9.set_title('水平投影图')
    ax9.axis('off')
    
    # 10. 垂直投影图像
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.imshow(vertical_proj['image'], cmap='gray')
    ax10.set_title('垂直投影图')
    ax10.axis('off')
    
    # 11-16. 其他二值化结果
    binary_keys = list(binary_images.keys())[1:4]  # 跳过第一个
    for i, key in enumerate(binary_keys):
        ax = fig.add_subplot(gs[2, 2+i])
        ax.imshow(binary_images[key]['image'], cmap='gray')
        ax.set_title(f'二值化({key})')
        ax.axis('off')
    
    plt.suptitle(f'图像特征综合分析 - {prefix}', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(str(output_dir / f"{prefix}_comprehensive_features.png"), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    
    print("🚀 开始图像特征提取分析")
    print("=" * 60)
    
    # 测试结果目录
    test_results_dir = Path('test_results')
    check_results_dir = Path('test_results/check')
    
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
    
    # 处理每个图像
    all_features = {}
    
    for plate_file in plate_region_files:
        print(f"\n🔍 处理图像: {plate_file.name}")
        
        try:
            features = extract_image_features(str(plate_file), check_results_dir)
            all_features[plate_file.stem] = features
            print(f"   ✅ 特征提取完成")
        except Exception as e:
            print(f"   ❌ 特征提取失败: {e}")
    
    # 生成总结报告
    generate_summary_report(all_features, check_results_dir)
    
    print("\n" + "=" * 60)
    print("🎉 图像特征提取分析完成！")
    print(f"📁 详细结果已保存到: {check_results_dir}")
    print("\n生成的文件包括:")
    print("   - 水平/垂直投影图和投影曲线")
    print("   - 多种二值化处理结果")
    print("   - 边缘检测结果")
    print("   - 轮廓分析结果")
    print("   - 直方图分析")
    print("   - 综合特征图")

def generate_summary_report(all_features, output_dir):
    """生成总结报告"""
    
    print(f"\n📊 生成总结报告...")
    
    report_path = output_dir / "feature_analysis_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("图像特征分析总结报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"分析图像数量: {len(all_features)}\n\n")
        
        for image_name, features in all_features.items():
            f.write(f"图像: {image_name}\n")
            f.write("-" * 30 + "\n")
            
            # 水平投影统计
            if 'horizontal_projection' in features:
                hp = features['horizontal_projection']
                f.write(f"水平投影: 最大值={hp['max_value']:.0f}, ")
                f.write(f"最小值={hp['min_value']:.0f}, ")
                f.write(f"均值={hp['mean_value']:.0f}\n")
            
            # 垂直投影统计
            if 'vertical_projection' in features:
                vp = features['vertical_projection']
                f.write(f"垂直投影: 最大值={vp['max_value']:.0f}, ")
                f.write(f"最小值={vp['min_value']:.0f}, ")
                f.write(f"均值={vp['mean_value']:.0f}\n")
            
            # 二值化统计
            if 'binary_images' in features:
                binary = features['binary_images']
                for key, result in binary.items():
                    f.write(f"{key}: 白像素={result['white_pixels']}, ")
                    f.write(f"白像素比例={result['white_ratio']:.3f}\n")
            
            # 边缘检测统计
            if 'edge_images' in features:
                edges = features['edge_images']
                for key, result in edges.items():
                    f.write(f"{key}: 边缘像素={result['edge_pixels']}\n")
            
            f.write("\n")
    
    print(f"   📁 总结报告保存到: {report_path}")

if __name__ == "__main__":
    main()