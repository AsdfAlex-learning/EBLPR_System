"""
图像二值化处理与重心定位测试脚本
功能：
- 图像二值化处理
- 定位黑色连续像素与白色连续像素的重心位置
- 使用2:3比例的矩形框标注黑色连续像素区域
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 字符模板定义（百分比坐标）
CHAR_TEMPLATES = [
    # 汉字区域 
    {'name': '汉字1', 'x': 37.88, 'y': 3.49, 'w': 20.06, 'h': 41.85}, 
    {'name': '汉字2', 'x': 55.28, 'y': 3.49, 'w': 20.06, 'h': 41.85}, 
    # 字母数字区域 8
    {'name': '字符1', 'x': 13.94, 'y': 45.91, 'w': 20.76, 'h': 47.94}, 
    {'name': '字符2', 'x': 24.39, 'y': 45.91, 'w': 20.76, 'h': 47.94}, 
    {'name': '字符3', 'x': 34.85, 'y': 45.91, 'w': 20.76, 'h': 47.94}, 
    {'name': '字符4', 'x': 45.31, 'y': 45.91, 'w': 20.76, 'h': 47.94}, 
    {'name': '字符5', 'x': 55.76, 'y': 45.91, 'w': 20.76, 'h': 47.94}, 
    {'name': '字符6', 'x': 66.22, 'y': 45.91, 'w': 20.76, 'h': 47.94} 
]

def extract_image_features(image_path, output_dir):
    """提取图像特征：二值化处理、重心定位、矩形框标注"""
    
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
    
    # 只进行二值化处理和重心定位
    binary_analysis = analyze_binarization_and_centroid(image, prefix, output_dir)
    
    return {
        'binary_analysis': binary_analysis
    }

def analyze_binarization_and_centroid(image, prefix, output_dir):
    """
    分析二值化图像，定位黑色和白色连续像素的重心位置，并标注矩形框
    
    新增双重降噪处理流程：
    1. 先进行连通区域分析去除黑色椒盐噪声
    2. 再进行形态学开运算去除黑色椒盐噪声
    3. 基于降噪后的二值化图像进行后续处理
    
    Args:
        image: 输入灰度图像
        prefix: 文件名前缀（用于生成输出文件名）
        output_dir: 输出目录（保存分析结果的文件夹路径）
    
    Returns:
        dict: 包含二值化分析结果
    """
    
    print(f"   📊 分析二值化与重心定位...")
    
    # 1. 二值化处理
    # cv2.threshold参数说明：
    # - image: 输入灰度图像
    # - 85: 阈值（小于85变为0/黑色，大于等于85变为255/白色）
    # - 255: 最大值（二值化后的白色像素值）
    # - cv2.THRESH_BINARY: 二值化类型（大于阈值设为最大值，否则设为0）
    _, binary = cv2.threshold(image, 85, 255, cv2.THRESH_BINARY)
    
    # 保存原始二值化图像
    binary_path = output_dir / f"{prefix}_binary.png"
    cv2.imwrite(str(binary_path), binary)
    
    # 2. 双重降噪处理
    print(f"   🧹 执行双重降噪处理...")
    
    # 2.1 基于连通区域分析去除黑色椒盐噪声
    # min_area=10: 面积小于10像素的黑色区域将被视为噪声并去除
    binary_cca = remove_noise_by_connected_component_analysis(binary, min_area=10)
    
    # 保存连通区域分析降噪后的图像
    binary_cca_path = output_dir / f"{prefix}_binary_cca.png"
    cv2.imwrite(str(binary_cca_path), binary_cca)
    
    # 2.2 基于形态学开运算去除黑色椒盐噪声
    # kernel_size=3: 使用3×3的正方形结构元素
    binary_denoised = remove_noise_by_morphological_opening(binary_cca, kernel_size=3)
    
    # 保存最终降噪后的图像
    binary_denoised_path = output_dir / f"{prefix}_binary_denoised.png"
    cv2.imwrite(str(binary_denoised_path), binary_denoised)
    
    # 3. 定位黑色连续像素区域（基于降噪后的图像）
    # 黑色连续像素的判定方法：黑色连续像素不少于300个的黑色像素区域
    # min_pixels=2900: 最小像素数阈值，过滤掉像素数小于2900的区域
    # min_width=180, min_height=270: 最小尺寸阈值，过滤掉小于180×270像素的区域
    black_regions = find_continuous_black_regions(binary_denoised, min_pixels=2900, min_width=40, min_height=60)
    
    # 4. 计算黑色区域重心位置
    black_centroids = calculate_centroids(black_regions)
    
    # 5. 新增：靠边筛选 - 过滤掉太靠近图像边缘的重心点
    # 参数说明：
    # - margin_ratio: 边缘比例阈值（0.0-1.0），越小表示越靠近边缘
    # - margin_pixels: 边缘像素阈值（绝对像素值），双重保障
    margin_ratio = 0.05  # 距离边缘5%以内的区域视为靠边
    margin_pixels = 20   # 距离边缘20像素以内的区域视为靠边
    
    edge_filtered_regions, edge_filtered_centroids = filter_edge_regions(
        black_regions, black_centroids, image.shape[1], image.shape[0], 
        margin_ratio=margin_ratio, margin_pixels=margin_pixels
    )
    
    # 6. 基于字符模板相对位置进行筛选（带保底机制）
    # 当且仅当模板的每一个框内都有黑色像素重心的时候才保留
    filtered_regions, filtered_centroids, spacing_factor = filter_by_template_matching_with_fallback(
        edge_filtered_regions, edge_filtered_centroids, image.shape[1], image.shape[0]
    )
    
    # 6. 创建标注图像（基于降噪后的二值化图像）
    annotated_image = create_annotated_image(
        image, binary_denoised, filtered_regions, filtered_centroids, spacing_factor,
        edge_filtered_regions, edge_filtered_centroids
    )
    
    # 保存标注图像
    annotated_path = output_dir / f"{prefix}_annotated.png"
    cv2.imwrite(str(annotated_path), annotated_image)
    
    # 7. 创建详细分析图（显示双重降噪过程）
    create_detailed_analysis_plot(
        image, 
        binary,           # 原始二值化图像
        binary_cca,        # 连通区域分析降噪后图像
        binary_denoised,   # 形态学开运算降噪后图像
        annotated_image,   # 基于降噪后图像的标注
        filtered_regions,  # 基于模板匹配筛选后的黑色区域
        prefix, 
        output_dir
    )
    
    return {
        'binary_image': binary,                    # 原始二值化图像
        'binary_cca': binary_cca,                  # 连通区域分析降噪后的图像
        'binary_denoised': binary_denoised,        # 最终降噪后的图像
        'black_regions': black_regions,            # 尺寸筛选后的黑色区域
        'black_centroids': black_centroids,        # 尺寸筛选后的重心坐标
        'filtered_regions': filtered_regions,      # 模板匹配筛选后的黑色区域
        'filtered_centroids': filtered_centroids,  # 模板匹配筛选后的重心坐标
        'annotated_image': annotated_image
    }

def find_continuous_black_regions(binary_image, min_pixels=300, min_width=180, min_height=270):
    """
    查找黑色连续像素区域，并进行尺寸筛选
    
    Args:
        binary_image: 二值化图像（0=黑色，255=白色）
        min_pixels: 最小像素数阈值，默认300
        min_width: 最小宽度阈值，默认180像素
        min_height: 最小高度阈值，默认270像素
    
    Returns:
        List[dict]: 每个区域的信息，包括边界框和像素坐标
    """
    # 反转图像：黑色像素变为白色（255），白色变为黑色（0）
    inverted = cv2.bitwise_not(binary_image)
    
    # 查找轮廓（黑色区域）
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    filtered_regions = []
    
    print(f"    📏 尺寸筛选参数: 最小面积={min_pixels}px, 最小宽度={min_width}px, 最小高度={min_height}px")
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        region_info = {
            'index': i,
            'area': area,
            'bounding_box': (x, y, w, h),
            'contour': contour,
            'width': w,
            'height': h
        }
        
        regions.append(region_info)
        
        # 三重筛选条件：面积、宽度、高度
        if area >= min_pixels and w >= min_width and h >= min_height:
            filtered_regions.append(region_info)
            print(f"    ✅ 保留区域 {i+1}: 面积={area:.0f}px, 尺寸={w}x{h}px")
        else:
            # 记录筛选原因
            reasons = []
            if area < min_pixels:
                reasons.append(f"面积不足({area:.0f}<{min_pixels})")
            if w < min_width:
                reasons.append(f"宽度不足({w}<{min_width})")
            if h < min_height:
                reasons.append(f"高度不足({h}<{min_height})")
            print(f"    ❌ 筛除区域 {i+1}: {', '.join(reasons)}")
    
    print(f"    📊 筛选结果: 原始{len(regions)}个区域 → 保留{len(filtered_regions)}个区域")
    
    return filtered_regions



def calculate_centroids(regions):
    """
    计算每个区域的重心位置

    Args:
        regions: 区域列表

    Returns:
        List[tuple]: 每个区域的重心坐标 (x, y)
    """
    centroids = []
    
    for region in regions:
        contour = region['contour']
        
        # 计算轮廓的矩
        M = cv2.moments(contour)
        
        # 计算重心坐标
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))
        else:
            # 如果m00为0，使用边界框中心作为重心
            x, y, w, h = region['bounding_box']
            cx = x + w // 2
            cy = y + h // 2
            centroids.append((cx, cy))
    
    return centroids


def filter_edge_regions(regions, centroids, image_width, image_height, margin_ratio=0.05, margin_pixels=20):
    """
    筛选掉太靠近图像边缘的黑色像素核心区域
    
    Args:
        regions: 黑色区域列表
        centroids: 对应的重心坐标列表
        image_width: 图像宽度
        image_height: 图像高度
        margin_ratio: 边缘比例阈值（0.0-1.0），越小表示越靠近边缘
        margin_pixels: 边缘像素阈值（绝对像素值），双重保障
        
    Returns:
        tuple: (filtered_regions, filtered_centroids) - 通过靠边筛选后的结果
    """
    if not regions or not centroids:
        return [], []
    
    print(f"    🎯 靠边筛选: {len(regions)}个区域")
    print(f"    筛选参数: 边缘比例={margin_ratio}, 边缘像素={margin_pixels}")
    
    # 计算边缘阈值
    left_margin = max(int(image_width * margin_ratio), margin_pixels)
    right_margin = image_width - left_margin
    top_margin = max(int(image_height * margin_ratio), margin_pixels)
    bottom_margin = image_height - top_margin
    
    print(f"    边缘范围: 左/右[{left_margin}-{right_margin}], 上/下[{top_margin}-{bottom_margin}]")
    
    filtered_regions = []
    filtered_centroids = []
    edge_regions = []
    
    for region, centroid in zip(regions, centroids):
        cx, cy = centroid
        
        # 检查重心是否在边缘区域内
        is_edge = (cx < left_margin or cx > right_margin or 
                   cy < top_margin or cy > bottom_margin)
        
        if is_edge:
            edge_regions.append(region)
            # 记录筛选原因
            reasons = []
            if cx < left_margin:
                reasons.append(f"左边缘({cx}<{left_margin})")
            elif cx > right_margin:
                reasons.append(f"右边缘({cx}>{right_margin})")
            if cy < top_margin:
                reasons.append(f"上边缘({cy}<{top_margin})")
            elif cy > bottom_margin:
                reasons.append(f"下边缘({cy}>{bottom_margin})")
            print(f"    ❌ 筛除靠边区域: {', '.join(reasons)}")
        else:
            filtered_regions.append(region)
            filtered_centroids.append(centroid)
            print(f"    ✅ 保留区域: 重心({cx},{cy})在安全区域内")
    
    print(f"    📊 靠边筛选结果: 原始{len(regions)}个区域 → 保留{len(filtered_regions)}个区域")
    
    # 返回筛选结果和边缘区域信息（用于调试）
    return filtered_regions, filtered_centroids


def adjust_template_spacing(templates, spacing_factor):
    """
    调整字符模板间距，保持六个字符模板的中心点不变
    
    Args:
        templates: 原始模板列表
        spacing_factor: 间距调整因子（1.0为原始间距）
        
    Returns:
        list: 调整间距后的模板列表
    """
    adjusted_templates = []
    
    # 处理汉字区域（保持不变）
    for template in templates[:2]:  # 前2个是汉字区域
        adjusted_templates.append(template.copy())
    
    # 处理6个字符区域
    char_templates = templates[2:8]
    
    # 计算原始字符模板的中心点坐标
    char_centers = []
    for template in char_templates:
        center_x = template['x'] + template['w'] / 2
        center_y = template['y'] + template['h'] / 2
        char_centers.append((center_x, center_y))
    
    # 计算原始字符模板之间的平均间距
    if len(char_centers) > 1:
        total_spacing = 0
        for i in range(len(char_centers) - 1):
            spacing = char_centers[i+1][0] - char_centers[i][0]
            total_spacing += spacing
        avg_spacing = total_spacing / (len(char_centers) - 1)
    else:
        avg_spacing = 0
    
    # 计算新的间距
    new_spacing = avg_spacing * spacing_factor
    
    # 重新计算字符模板位置
    for i, template in enumerate(char_templates):
        adjusted_template = template.copy()
        
        # 计算新的x坐标：保持中心点不变，调整位置
        if i == 0:
            # 第一个字符：向左移动
            adjusted_template['x'] = char_centers[i][0] - template['w'] / 2 - (new_spacing - avg_spacing) * 1.5
        elif i == 5:
            # 最后一个字符：向右移动
            adjusted_template['x'] = char_centers[i][0] - template['w'] / 2 + (new_spacing - avg_spacing) * 1.5
        else:
            # 中间字符：根据位置调整
            offset = (new_spacing - avg_spacing) * (i - 2.5)
            adjusted_template['x'] = char_centers[i][0] - template['w'] / 2 + offset
        
        adjusted_templates.append(adjusted_template)
    
    return adjusted_templates

def filter_by_template_matching_with_fallback(regions, centroids, image_width, image_height):
    """
    基于字符模板相对位置筛选黑色像素重心，带保底机制
    
    Args:
        regions: 黑色区域列表
        centroids: 对应的重心坐标列表
        image_width: 图像宽度
        image_height: 图像高度
        
    Returns:
        tuple: (filtered_regions, filtered_centroids, spacing_factor) - 通过模板匹配筛选后的结果和使用的间距因子
    """
    if not regions or not centroids:
        return [], [], 1.0
    
    print(f"    🎯 模板匹配筛选: {len(regions)}个区域")
    
    # 尝试不同的间距因子
    spacing_factors = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  # 逐渐增大间距
    
    for spacing_factor in spacing_factors:
        # 调整模板间距
        adjusted_templates = adjust_template_spacing(CHAR_TEMPLATES, spacing_factor)
        
        # 将百分比坐标转换为实际像素坐标
        template_boxes = []
        for template in adjusted_templates:
            # 转换百分比坐标为像素坐标
            x_px = int(template['x'] / 100 * image_width)
            y_px = int(template['y'] / 100 * image_height)
            w_px = int(template['w'] / 100 * image_width)
            h_px = int(template['h'] / 100 * image_height)
            
            template_box = {
                'name': template['name'],
                'x': x_px,
                'y': y_px,
                'w': w_px,
                'h': h_px,
                'center_x': x_px + w_px // 2,
                'center_y': y_px + h_px // 2
            }
            template_boxes.append(template_box)
        
        # 筛选逻辑：当且仅当模板的每一个框内都有黑色像素重心的时候才保留
        # 检查每个模板框内是否至少有一个重心点
        template_matches = {}
        for template in template_boxes:
            template_matches[template['name']] = False
            
            # 检查该模板框内是否有重心点
            for centroid in centroids:
                cx, cy = centroid
                if (template['x'] <= cx <= template['x'] + template['w'] and
                    template['y'] <= cy <= template['y'] + template['h']):
                    template_matches[template['name']] = True
                    break
        
        # 检查是否所有模板框内都有重心点
        all_templates_matched = all(template_matches.values())
        
        if all_templates_matched:
            # 排除不在模板框内的重心点
            filtered_regions = []
            filtered_centroids = []
            
            for region, centroid in zip(regions, centroids):
                cx, cy = centroid
                
                # 检查重心点是否在任意模板框内
                in_any_template = False
                for template in template_boxes:
                    if (template['x'] <= cx <= template['x'] + template['w'] and
                        template['y'] <= cy <= template['y'] + template['h']):
                        in_any_template = True
                        break
                
                if in_any_template:
                    filtered_regions.append(region)
                    filtered_centroids.append(centroid)
            
            print(f"    ✅ 模板匹配成功: 保留{len(filtered_regions)}个区域 (间距因子: {spacing_factor})")
            return filtered_regions, filtered_centroids, spacing_factor
        
    # 如果所有间距因子都尝试失败，返回空结果
    print(f"    ❌ 模板匹配失败: 保留0个区域")
    return [], [], 1.0  # 返回默认间距因子

def remove_noise_by_connected_component_analysis(binary_image, min_area=10):
    """
    基于连通区域分析去除黑色椒盐噪声
    
    核心逻辑：
    1. 对二值化车牌图像进行连通区域标记，识别图像中所有独立的连通区域
    2. 统计每个连通区域的像素面积，设定合理的面积阈值（如小于10个像素）
    3. 将面积小于阈值的黑色连通区域（即黑色椒盐噪声点）全部置为白色
    4. 保留面积符合阈值的车牌字符连通区域，精准剔除小面积的黑色噪声点且不破坏字符结构
    
    Args:
        binary_image: 二值化图像（0=黑色，255=白色）
        min_area: 最小面积阈值，小于此面积的黑色区域将被视为噪声并去除
    
    Returns:
        np.ndarray: 去除黑色椒盐噪声后的二值化图像
    """
    # 创建图像副本，避免修改原始图像
    denoised_image = binary_image.copy()
    
    # 反转图像：黑色像素变为白色（255），白色变为黑色（0）
    # 这样黑色噪声点就变成了白色区域，便于连通区域分析
    inverted = cv2.bitwise_not(binary_image)
    
    # 连通区域标记
    # cv2.connectedComponentsWithStats参数说明：
    # - inverted: 输入图像（8位单通道）
    # - connectivity: 连通性（4或8连通）
    # - ltype: 输出标签图像的数据类型
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8, ltype=cv2.CV_32S)
    
    # 遍历所有连通区域（跳过背景区域，索引0）
    for label in range(1, num_labels):
        # 获取当前区域的统计信息
        area = stats[label, cv2.CC_STAT_AREA]  # 区域面积
        
        # 如果区域面积小于阈值，则认为是噪声
        if area < min_area:
            # 找到该区域的所有像素位置
            mask = (labels == label).astype(np.uint8) * 255
            
            # 将噪声区域置为白色（在原始二值化图像中，白色=255）
            # 由于我们处理的是黑色噪声点，需要将其变为白色
            denoised_image[mask > 0] = 255
    
    print(f"    🔍 连通区域分析降噪：检测到{num_labels-1}个连通区域，去除面积小于{min_area}像素的噪声")
    return denoised_image

def remove_noise_by_morphological_opening(binary_image, kernel_size=3):
    """
    基于形态学开运算去除黑色椒盐噪声
    
    核心逻辑：
    1. 首先将二值化图像处理为黑底白字（噪声为黑色小点、字符为白色）
    2. 选择3×3的正方形结构元素
    3. 先对图像执行腐蚀操作，"消除"面积较小的黑色噪声点
    4. 再执行膨胀操作，恢复车牌字符的原始轮廓和尺寸
    5. 最终得到去除黑色椒盐噪声且字符边缘完整的图像
    
    Args:
        binary_image: 二值化图像（0=黑色，255=白色）
        kernel_size: 结构元素大小（默认3×3）
    
    Returns:
        np.ndarray: 形态学开运算降噪后的二值化图像
    """
    # 反转图像：使黑色噪声点变为白色区域，便于形态学操作
    # 在反转后的图像中：黑色噪声点→白色小点，白色字符→黑色区域
    inverted = cv2.bitwise_not(binary_image)
    
    # 创建3×3的正方形结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # 执行开运算（先腐蚀后膨胀）
    # 腐蚀操作：消除小的白色噪声点（即原始图像中的黑色噪声点）
    eroded = cv2.erode(inverted, kernel, iterations=1)
    
    # 膨胀操作：恢复字符的原始尺寸和轮廓
    opened = cv2.dilate(eroded, kernel, iterations=1)
    
    # 将处理后的图像反转回原始的二值化格式
    denoised_image = cv2.bitwise_not(opened)
    
    print(f"    🔍 形态学开运算降噪：使用{kernel_size}×{kernel_size}结构元素完成开运算")
    return denoised_image

def create_annotated_image(original_image, binary_image, black_regions, black_centroids, spacing_factor=1.0, edge_filtered_regions=None, edge_filtered_centroids=None):
    """
    创建标注图像，显示黑色区域重心位置、矩形框和字符模板框（支持间距调整）
    
    重要说明：此函数基于降噪后的二值化图像进行标注
    - binary_image参数应为经过双重降噪处理后的二值化图像
    - 标注内容包括：黑色区域的重心点、2:3比例的矩形框和字符模板框
    
    Args:
        original_image: 原始灰度图像（仅用于参考，实际标注基于binary_image）
        binary_image: 降噪后的二值化图像（基于此图像进行标注）
        black_regions: 黑色区域列表（基于降噪图像检测）
        black_centroids: 黑色区域重心列表（基于降噪图像计算）
        spacing_factor: 模板间距调整因子
        edge_filtered_regions: 靠边筛选后的区域列表（可选）
        edge_filtered_centroids: 靠边筛选后的重心列表（可选）
    
    Returns:
        np.ndarray: 标注后的彩色图像
    """
    # 基于降噪后的二值化图像创建彩色图像用于标注
    if len(binary_image.shape) == 2:
        annotated = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    else:
        annotated = binary_image.copy()
    
    # 获取图像尺寸
    image_height, image_width = binary_image.shape[:2]
    
    # 添加筛选参数说明（便于精调）
    param_text = f"筛选参数: 面积≥2900px, 尺寸≥180×270px"
    cv2.putText(annotated, param_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 2)
    
    # 添加区域统计信息
    stats_text = f"检测结果: {len(black_regions)}个区域通过筛选"
    cv2.putText(annotated, stats_text, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 2)
    
    # 添加模板匹配信息
    template_text = f"模板匹配: 8个字符模板框（蓝色边框）"
    cv2.putText(annotated, template_text, (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 0), 2)
    
    # 添加间距因子信息
    spacing_text = f"间距因子: {spacing_factor}"
    cv2.putText(annotated, spacing_text, (10, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 0), 2)
    
    # 添加靠边筛选信息（如果提供了靠边筛选结果）
    if edge_filtered_regions is not None and edge_filtered_centroids is not None:
        edge_stats_text = f"靠边筛选: {len(edge_filtered_regions)}个区域通过"
        cv2.putText(annotated, edge_stats_text, (10, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 200), 2)
        
        # 绘制靠边筛选范围（紫色虚线框）
        margin_ratio = 0.05  # 5%边缘范围
        margin_pixels = 20   # 20像素边缘范围
        
        # 计算边缘范围（双重阈值）
        margin_x_ratio = int(image_width * margin_ratio)
        margin_y_ratio = int(image_height * margin_ratio)
        
        # 绘制相对边缘范围（紫色虚线）
        cv2.rectangle(annotated, 
                     (margin_x_ratio, margin_y_ratio), 
                     (image_width - margin_x_ratio, image_height - margin_y_ratio), 
                     (200, 0, 200), 1, cv2.LINE_AA)  # 紫色虚线框
        
        # 绘制绝对边缘范围（紫色虚线）
        cv2.rectangle(annotated, 
                     (margin_pixels, margin_pixels), 
                     (image_width - margin_pixels, image_height - margin_pixels), 
                     (200, 0, 200), 1, cv2.LINE_AA)  # 紫色虚线框
        
        # 添加边缘范围说明
        edge_text1 = f"边缘范围: 相对{int(margin_ratio*100)}% ({margin_x_ratio}x{margin_y_ratio}px)"
        cv2.putText(annotated, edge_text1, (10, 165), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 1)
        
        edge_text2 = f"绝对{max(margin_pixels, margin_pixels)}px"
        cv2.putText(annotated, edge_text2, (10, 185), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 1)
    
    # 绘制字符模板框（基于图像尺寸）
    image_height, image_width = binary_image.shape[:2]
    
    # 调整模板间距
    adjusted_templates = adjust_template_spacing(CHAR_TEMPLATES, spacing_factor)
    
    for template in adjusted_templates:
        # 转换百分比坐标为像素坐标
        x_px = int(template['x'] / 100 * image_width)
        y_px = int(template['y'] / 100 * image_height)
        w_px = int(template['w'] / 100 * image_width)
        h_px = int(template['h'] / 100 * image_height)
        
        # 绘制模板框（蓝色边框）
        cv2.rectangle(annotated, 
                     (x_px, y_px), 
                     (x_px + w_px, y_px + h_px), 
                     (200, 100, 0), 2)  # 蓝色边框
        
        # 添加模板名称标签
        cv2.putText(annotated, template['name'], (x_px, y_px - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 100, 0), 1)
    
    # 1. 标注黑色区域重心和矩形框
    for i, (region, centroid) in enumerate(zip(black_regions, black_centroids)):
        x, y, w, h = region['bounding_box']
        cx, cy = centroid
        area = region['area']
        
        # 矩形框生成逻辑说明：
        # 目标：生成2:3比例的矩形框，几何中心与黑色像素重心重合
        
        # 计算矩形框宽度：基于区域的最小尺寸进行缩放
        # min(w, h): 取区域边界框的较小尺寸作为基准
        # 2.5: 缩放因子，控制矩形框相对于区域大小的比例（可调整参数）
        rect_width = int(min(w, h) * 2)  # 适当放大
        
        # 计算矩形框高度：保持2:3比例
        # 3/2: 高度与宽度的比例（2:3比例，即宽度:高度 = 2:3）
        rect_height = int(rect_width * 3 / 2)  # 2:3比例
        
        # 计算矩形框的左上角坐标：使几何中心与重心重合
        # cx - rect_width//2: 矩形框左上角x坐标 = 重心x坐标 - 矩形框宽度的一半
        # cy - rect_height//2: 矩形框左上角y坐标 = 重心y坐标 - 矩形框高度的一半
        rect_x = cx - rect_width // 2
        rect_y = cy - rect_height // 2
        
        # 绘制矩形框（红色）
        # cv2.rectangle参数说明：
        # - annotated: 目标图像
        # - (rect_x, rect_y): 矩形框左上角坐标
        # - (rect_x + rect_width, rect_y + rect_height): 矩形框右下角坐标
        # - (0, 0, 255): BGR颜色值（红色）
        # - 2: 线条粗细（像素）
        cv2.rectangle(annotated, 
                     (rect_x, rect_y), 
                     (rect_x + rect_width, rect_y + rect_height), 
                     (0, 0, 255), 2)  # 红色边框
        
        # 绘制重心点（红色）
        # cv2.circle参数说明：
        # - annotated: 目标图像
        # - (cx, cy): 圆心坐标（重心位置）
        # - 5: 圆的半径（像素）
        # - (0, 0, 255): BGR颜色值（红色）
        # - -1: 填充圆（正值表示线条粗细，负值表示填充）
        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)  # 红色实心圆
        
        # 添加标签和尺寸信息
        # cv2.putText参数说明：
        # - annotated: 目标图像
        # - f"Black{i+1}": 标签文本
        # - (cx+10, cy-10): 文本位置（重心右下方）
        # - cv2.FONT_HERSHEY_SIMPLEX: 字体类型
        # - 0.5: 字体大小
        # - (0, 0, 255): BGR颜色值（红色）
        # - 1: 线条粗细
        cv2.putText(annotated, f"Black{i+1}", (cx+10, cy-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 添加矩形框尺寸信息（像素格式）
        # 在矩形框上方显示宽度和高度信息
        size_text = f"{rect_width} x {rect_height} px"
        cv2.putText(annotated, size_text, (rect_x, rect_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 添加区域原始尺寸信息（便于精调）
        original_size_text = f"原始: {w}x{h}px ({area:.0f}px)"
        cv2.putText(annotated, original_size_text, (rect_x, rect_y - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 1)
        
        # 添加缩放因子信息
        scale_factor = 2.0  # 当前缩放因子
        scale_text = f"缩放: {scale_factor}×"
        cv2.putText(annotated, scale_text, (rect_x, rect_y - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 0, 150), 1)
    
    return annotated

def create_detailed_analysis_plot(original_image, binary_image, binary_cca, binary_denoised, 
                                 annotated_image, black_regions, prefix, output_dir):
    """
    创建详细分析图，显示原始图像、二值化图像、双重降噪过程和标注图像
    
    Args:
        original_image: 原始图像
        binary_image: 原始二值化图像
        binary_cca: 连通区域分析降噪后图像
        binary_denoised: 形态学开运算降噪后图像
        annotated_image: 基于降噪后图像的标注图像
        black_regions: 黑色区域列表（基于降噪后图像）
        prefix: 文件名前缀
        output_dir: 输出目录
    """
    # 创建6个子图的布局（2行3列）
    fig, axes = plt.subplots(2, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    # 1. 原始图像
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 2. 原始二值化图像
    axes[1].imshow(binary_image, cmap='gray')
    axes[1].set_title('原始二值化图像')
    axes[1].axis('off')
    
    # 3. 连通区域分析降噪后图像
    axes[2].imshow(binary_cca, cmap='gray')
    axes[2].set_title('连通区域分析降噪后')
    axes[2].axis('off')
    
    # 4. 形态学开运算降噪后图像
    axes[3].imshow(binary_denoised, cmap='gray')
    axes[3].set_title('形态学开运算降噪后')
    axes[3].axis('off')
    
    # 5. 标注图像
    axes[4].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    axes[4].set_title('重心与矩形框标注')
    axes[4].axis('off')
    
    # 6. 统计信息
    axes[5].axis('off')
    
    # 添加统计信息文本
    stats_text = f"双重降噪处理流程统计信息:\n\n"
    stats_text += f"双重降噪处理流程:\n"
    stats_text += f"1. 连通区域分析降噪\n"
    stats_text += f"   - 面积阈值: <10像素\n"
    stats_text += f"   - 去除小面积黑色噪声点\n\n"
    stats_text += f"2. 形态学开运算降噪\n"
    stats_text += f"   - 结构元素: 3×3正方形\n"
    stats_text += f"   - 先腐蚀后膨胀操作\n\n"
    stats_text += f"黑色区域检测结果:\n"
    stats_text += f"黑色区域数量: {len(black_regions)}\n\n"
    
    if black_regions:
        stats_text += f"最大黑色区域面积: {max(r['area'] for r in black_regions):.0f}\n\n"
    
    stats_text += f"黑色区域筛选条件:\n"
    stats_text += f"- 面积阈值: ≥2900像素\n"
    stats_text += f"- 最小宽度: ≥180像素\n"
    stats_text += f"- 最小高度: ≥270像素\n\n"
    
    stats_text += f"矩形框关键参数:\n"
    stats_text += f"- 矩形框比例: 2:3 (宽度:高度)\n"
    stats_text += f"- 缩放因子: 2.0×区域最小尺寸\n"
    stats_text += f"- 几何中心: 与黑色像素重心重合\n"
    stats_text += f"- 边框颜色: 红色 (BGR: 0,0,255)\n"
    stats_text += f"- 边框粗细: 2像素\n\n"
    
    stats_text += f"标注说明:\n"
    stats_text += f"- 红色矩形框: 黑色区域 (2:3比例)\n"
    stats_text += f"- 红色点: 黑色区域重心\n"
    stats_text += f"- 尺寸标注: 矩形框上方显示宽度×高度像素\n"
    stats_text += f"- 原始尺寸: 绿色文本显示区域原始尺寸和面积\n"
    stats_text += f"- 缩放因子: 紫色文本显示当前缩放倍数\n"
    stats_text += f"- 标签编号: Black1, Black2, ...\n\n"
    
    stats_text += f"字符模板匹配筛选（带保底机制）:\n"
    stats_text += f"- 模板数量: 8个字符区域\n"
    stats_text += f"- 汉字区域: 2个 (汉字1, 汉字2)\n"
    stats_text += f"- 字母数字区域: 6个 (字符1-6)\n"
    stats_text += f"- 筛选条件: 所有模板框内都有重心点\n"
    stats_text += f"- 排除条件: 不在模板框内的重心点\n"
    stats_text += f"- 模板框颜色: 蓝色 (BGR: 200,100,0)\n"
    stats_text += f"- 保底机制: 间距因子1.0-1.5逐步调整\n"
    stats_text += f"- 调整方式: 保持中心点不变，左右平移\n\n"
    
    stats_text += f"精调参数说明:\n"
    stats_text += f"- 筛选参数: 面积≥2900px, 尺寸≥180×270px\n"
    stats_text += f"- 缩放因子: 2.0×区域最小尺寸\n"
    stats_text += f"- 矩形比例: 固定2:3 (宽度:高度)\n\n"
    
    stats_text += f"注意: 所有标注基于双重降噪后的二值化图像"
    
    axes[5].text(0.1, 0.9, stats_text, transform=axes[5].transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(f'图像二值化与重心定位分析（双重降噪） - {prefix}', fontsize=16)
    plt.tight_layout()
    plt.savefig(str(output_dir / f"{prefix}_detailed_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()

def recognize_characters_from_regions(filtered_regions, filtered_centroids, binary_denoised, image_width, image_height):
    """
    对筛选后的黑色像素框进行字符识别
    
    Args:
        filtered_regions: 模板匹配筛选后的黑色区域列表
        filtered_centroids: 模板匹配筛选后的重心坐标列表
        binary_denoised: 降噪后的二值化图像
        image_width: 图像宽度
        image_height: 图像高度
    
    Returns:
        dict: 字符识别结果，包含每个字符模板对应的识别字符和相似度
    """
    
    # 加载字符模板
    templates_dir = Path("char_templates")
    if not templates_dir.exists():
        print("❌ 字符模板目录不存在！")
        return {}
    
    # 加载所有模板图像
    templates = {}
    template_files = list(templates_dir.glob("*.png"))
    
    # 排除预览图
    template_files = [f for f in template_files if f.name != "template_preview.png"]
    
    for template_file in template_files:
        template_name = template_file.stem
        template_img = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
        if template_img is not None:
            templates[template_name] = template_img
    
    # 字符识别结果
    recognition_results = {}
    
    # 调整模板间距
    adjusted_templates = adjust_template_spacing(CHAR_TEMPLATES, 1.0)
    
    for template in adjusted_templates:
        # 转换百分比坐标为像素坐标
        x_px = int(template['x'] / 100 * image_width)
        y_px = int(template['y'] / 100 * image_height)
        w_px = int(template['w'] / 100 * image_width)
        h_px = int(template['h'] / 100 * image_height)
        
        # 查找在模板框内的黑色像素框
        template_centroids = []
        template_regions = []
        
        for region, centroid in zip(filtered_regions, filtered_centroids):
            cx, cy = centroid
            # 检查重心是否在模板框内
            if (x_px <= cx <= x_px + w_px and 
                y_px <= cy <= y_px + h_px):
                template_centroids.append(centroid)
                template_regions.append(region)
        
        # 如果模板框内有多个黑色像素框，选择相似度最高的
        if template_regions:
            best_match = None
            best_similarity = -1
            best_char = None
            
            # 定义汉字模板列表
            chinese_templates = ['guangdong', 'zhou', 'foshan', 'shan', 'guang', 'fo', 'shan']
            
            for region in template_regions:
                x, y, w, h = region['bounding_box']
                
                # 提取字符区域图像
                char_region = binary_denoised[y:y+h, x:x+w]
                
                # 调整字符区域尺寸为20x40（与模板尺寸一致）
                char_resized = cv2.resize(char_region, (40, 60))
                
                # 与模板进行匹配
                for template_name, template_img in templates.items():
                    # 如果是汉字1或汉字2的框，只匹配汉字模板
                    if template['name'] in ['汉字1', '汉字2']:
                        if template_name not in chinese_templates:
                            continue
                    
                    # 计算模板匹配相似度
                    similarity = cv2.matchTemplate(char_resized, template_img, cv2.TM_CCOEFF_NORMED)
                    max_similarity = np.max(similarity)
                    
                    if max_similarity > best_similarity:
                        best_similarity = max_similarity
                        best_match = region
                        best_char = template_name
            
            # 保存最佳匹配结果
            if best_char:
                # 处理中文映射
                char_map = {
                    'guangdong': '广',
                    'zhou': '州', 
                    'foshan': '佛',
                    'shan': '山'
                }
                
                final_char = char_map.get(best_char, best_char)
                recognition_results[template['name']] = {
                    'character': final_char,
                    'similarity': best_similarity,
                    'region': best_match
                }
                
                # 如果是汉字1或汉字2且匹配到汉字模板，显示特殊标记
                if template['name'] in ['汉字1', '汉字2'] and best_char in chinese_templates:
                    print(f"   ✅ {template['name']}: '{final_char}' ({best_similarity:.3f}) [汉字]")
                else:
                    print(f"   ✅ {template['name']}: '{final_char}' ({best_similarity:.3f})")
            else:
                # 如果是汉字1或汉字2但没有匹配到汉字模板
                if template['name'] in ['汉字1', '汉字2']:
                    print(f"   ❌ {template['name']}: 未找到汉字匹配")
                else:
                    print(f"   ❌ {template['name']}: 未识别")
        else:
            print(f"   ❌ {template['name']}: 无字符")
    
    return recognition_results

def save_recognition_results(recognition_results, output_dir, prefix):
    """
    保存字符识别结果到txt文件
    
    Args:
        recognition_results: 字符识别结果
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    
    # 按模板顺序排列字符
    template_order = ['汉字1', '汉字2', '字符1', '字符2', '字符3', '字符4', '字符5', '字符6']
    
    # 创建结果副本用于更正
    corrected_results = recognition_results.copy()
    correction_info = ""
    
    # 检查汉字1和汉字2的识别结果
    if '汉字1' in recognition_results and '汉字2' in recognition_results:
        char1 = recognition_results['汉字1']['character']
        char2 = recognition_results['汉字2']['character']
        
        # 如果汉字1和汉字2的结果包含"fo"或"shan"，则更正汉字1为"佛"，汉字2为"山"
        if char1 in ['fo', 'shan'] or char2 in ['fo', 'shan']:
            corrected_results['汉字1'] = {
                'character': '佛',
                'similarity': recognition_results['汉字1']['similarity'],
                'region': recognition_results['汉字1']['region']
            }
            corrected_results['汉字2'] = {
                'character': '山',
                'similarity': recognition_results['汉字2']['similarity'],
                'region': recognition_results['汉字2']['region']
            }
            correction_info = f"（原结果: {char1}{char2}，根据汉字识别结果自动更正为佛山）"
        # 如果汉字1和汉字2的结果包含"guang"或"zhou"，则更正汉字1为"广"，汉字2为"州"
        elif char1 in ['guang', 'zhou'] or char2 in ['guang', 'zhou']:
            corrected_results['汉字1'] = {
                'character': '广',
                'similarity': recognition_results['汉字1']['similarity'],
                'region': recognition_results['汉字1']['region']
            }
            corrected_results['汉字2'] = {
                'character': '州',
                'similarity': recognition_results['汉字2']['similarity'],
                'region': recognition_results['汉字2']['region']
            }
            correction_info = f"（原结果: {char1}{char2}，根据汉字识别结果自动更正为广州）"
    
    # 构建更正后的车牌号码
    corrected_plate_number = ""
    for template_name in template_order:
        if template_name in corrected_results:
            corrected_plate_number += corrected_results[template_name]['character']
        else:
            corrected_plate_number += "?"
    
    # 保存到txt文件
    result_file = output_dir / f"{prefix}_recognition_result.txt"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"车牌号码识别结果: {corrected_plate_number}")
        if correction_info:
            f.write(f" {correction_info}")
        f.write(f"\n\n")
        f.write("详细识别信息:\n")
        f.write("=" * 50 + "\n")
        
        for template_name in template_order:
            if template_name in corrected_results:
                result = corrected_results[template_name]
                f.write(f"{template_name}: {result['character']} (相似度: {result['similarity']:.3f})\n")
            else:
                f.write(f"{template_name}: 未识别\n")
        
        # 添加更正说明
        if correction_info:
            f.write(f"\n更正说明:\n")
            f.write(f"- 汉字1原识别为: {recognition_results['汉字1']['character'] if '汉字1' in recognition_results else '?'}\n")
            f.write(f"- 汉字2原识别为: {recognition_results['汉字2']['character'] if '汉字2' in recognition_results else '?'}\n")
            f.write(f"- 根据识别结果自动更正为: {corrected_results['汉字1']['character']}{corrected_results['汉字2']['character']}\n")
    
    print(f"   ✅ 识别结果保存: {result_file}")
    print(f"   📋 车牌号码: {corrected_plate_number}")
    if correction_info:
        print(f"   🔄 自动更正: {correction_info}")
    
    return corrected_plate_number

def main():
    """主函数"""
    
    # 设置路径（与原本的03test路径一样）
    input_dir = Path("test_results")
    output_dir = Path("test_results\check")
    
    # 确保输出目录存在
    output_dir.mkdir(exist_ok=True)
    
    # 获取输入图像
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if not image_files:
        print("❌ 未找到输入图像！")
        return
    
    print(f"处理 {len(image_files)} 个图像文件")
    
    for image_file in image_files:
        print(f"\n处理: {image_file.name}")
        
        # 读取图像
        image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"   无法读取图像")
            continue
        
        # 创建图像特定的输出目录
        image_output_dir = output_dir / image_file.stem
        image_output_dir.mkdir(exist_ok=True)
        
        # 分析二值化与重心定位
        binary_analysis = analyze_binarization_and_centroid(image, image_file.stem, image_output_dir)
        
        # 输出分析结果
        black_count = len(binary_analysis['black_regions'])
        
        print(f"   二值化完成，检测到 {black_count} 个黑色区域")
        
        # 进行字符识别
        if binary_analysis['filtered_regions']:
            print(f"   开始字符识别...")
            
            recognition_results = recognize_characters_from_regions(
                binary_analysis['filtered_regions'],
                binary_analysis['filtered_centroids'],
                binary_analysis['binary_denoised'],
                image.shape[1],
                image.shape[0]
            )
            
            # 保存识别结果
            plate_number = save_recognition_results(recognition_results, image_output_dir, image_file.stem)
            
        else:
            print(f"   未找到字符区域，跳过识别")
        
        print(f"   结果保存到: {image_output_dir}")
        
    
    print(f"\n所有图像处理完成！")
    print(f"   处理了 {len(image_files)} 个图像")
    print(f"   结果保存在: {output_dir}")
    print("   - 详细分析图")
    print("   - 字符识别结果txt文件")
    print("   - 车牌号码识别结果")



if __name__ == "__main__":
    main()