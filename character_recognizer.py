"""
字符识别模块 (重构版)
基于 03_Test_image_features.py 的车牌字符定位与识别逻辑
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any

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

class CharacterRecognizer:
    def __init__(self, template_dir: str = "char_templates"):
        self.template_dir = Path(template_dir)
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, np.ndarray]:
        """加载字符模板"""
        templates = {}
        if not self.template_dir.exists():
            print(f"❌ 字符模板目录不存在: {self.template_dir}")
            return templates
            
        template_files = list(self.template_dir.glob("*.png"))
        template_files = [f for f in template_files if f.name != "template_preview.png"]
        
        for template_file in template_files:
            template_name = template_file.stem
            template_img = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
            if template_img is not None:
                templates[template_name] = template_img
                
        return templates

    def recognize(self, plate_image: np.ndarray, debug_output_dir: Path = None) -> str:
        """
        识别车牌图像中的字符
        Args:
            plate_image: 车牌灰度图像
            debug_output_dir: 调试输出目录，如果提供则保存中间结果
            
        Returns:
            str: 识别出的车牌号码
        """
        if plate_image is None:
            return ""
            
        # 1. 预处理与特征提取
        # 二值化处理
        _, binary = cv2.threshold(plate_image, 85, 255, cv2.THRESH_BINARY)
        
        # 保存中间结果
        if debug_output_dir:
            debug_output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_output_dir / "binary.png"), binary)
            
        # 双重降噪
        # 1.1 连通区域分析降噪
        binary_cca = self._remove_noise_by_connected_component_analysis(binary, min_area=10)
        
        # 1.2 形态学开运算降噪
        binary_denoised = self._remove_noise_by_morphological_opening(binary_cca, kernel_size=3)
        
        if debug_output_dir:
            cv2.imwrite(str(debug_output_dir / "binary_denoised.png"), binary_denoised)
            
        # 2. 定位黑色连续像素区域
        black_regions = self._find_continuous_black_regions(binary_denoised, min_pixels=2900, min_width=40, min_height=60)
        black_centroids = self._calculate_centroids(black_regions)
        
        # 3. 靠边筛选
        edge_filtered_regions, edge_filtered_centroids = self._filter_edge_regions(
            black_regions, black_centroids, plate_image.shape[1], plate_image.shape[0], 
            margin_ratio=0.05, margin_pixels=20
        )
        
        # 4. 基于字符模板相对位置进行筛选
        filtered_regions, filtered_centroids, _ = self._filter_by_template_matching_with_fallback(
            edge_filtered_regions, edge_filtered_centroids, plate_image.shape[1], plate_image.shape[0]
        )
        
        if not filtered_regions:
            print("❌ 未找到有效的字符区域")
            return ""
            
        # 5. 字符识别
        recognition_results = self._recognize_characters_from_regions(
            filtered_regions, filtered_centroids, binary_denoised, 
            plate_image.shape[1], plate_image.shape[0]
        )
        
        # 6. 结果整合与修正
        plate_number = self._process_results(recognition_results)
        
        return plate_number

    def _remove_noise_by_connected_component_analysis(self, binary_image, min_area=10):
        denoised_image = binary_image.copy()
        inverted = cv2.bitwise_not(binary_image)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8, ltype=cv2.CV_32S)
        
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area:
                mask = (labels == label).astype(np.uint8) * 255
                denoised_image[mask > 0] = 255
        return denoised_image

    def _remove_noise_by_morphological_opening(self, binary_image, kernel_size=3):
        inverted = cv2.bitwise_not(binary_image)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        eroded = cv2.erode(inverted, kernel, iterations=1)
        opened = cv2.dilate(eroded, kernel, iterations=1)
        return cv2.bitwise_not(opened)

    def _find_continuous_black_regions(self, binary_image, min_pixels=300, min_width=180, min_height=270):
        inverted = cv2.bitwise_not(binary_image)
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_regions = []
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
            
            if area >= min_pixels and w >= min_width and h >= min_height:
                filtered_regions.append(region_info)
                
        return filtered_regions

    def _calculate_centroids(self, regions):
        centroids = []
        for region in regions:
            contour = region['contour']
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy))
            else:
                x, y, w, h = region['bounding_box']
                cx = x + w // 2
                cy = y + h // 2
                centroids.append((cx, cy))
        return centroids

    def _filter_edge_regions(self, regions, centroids, image_width, image_height, margin_ratio=0.05, margin_pixels=20):
        if not regions or not centroids:
            return [], []
            
        left_margin = max(int(image_width * margin_ratio), margin_pixels)
        right_margin = image_width - left_margin
        top_margin = max(int(image_height * margin_ratio), margin_pixels)
        bottom_margin = image_height - top_margin
        
        filtered_regions = []
        filtered_centroids = []
        
        for region, centroid in zip(regions, centroids):
            cx, cy = centroid
            is_edge = (cx < left_margin or cx > right_margin or 
                       cy < top_margin or cy > bottom_margin)
            
            if not is_edge:
                filtered_regions.append(region)
                filtered_centroids.append(centroid)
                
        return filtered_regions, filtered_centroids

    def _adjust_template_spacing(self, templates, spacing_factor):
        adjusted_templates = []
        for template in templates[:2]:
            adjusted_templates.append(template.copy())
            
        char_templates = templates[2:8]
        char_centers = []
        for template in char_templates:
            center_x = template['x'] + template['w'] / 2
            center_y = template['y'] + template['h'] / 2
            char_centers.append((center_x, center_y))
            
        if len(char_centers) > 1:
            total_spacing = 0
            for i in range(len(char_centers) - 1):
                spacing = char_centers[i+1][0] - char_centers[i][0]
                total_spacing += spacing
            avg_spacing = total_spacing / (len(char_centers) - 1)
        else:
            avg_spacing = 0
            
        new_spacing = avg_spacing * spacing_factor
        
        for i, template in enumerate(char_templates):
            adjusted_template = template.copy()
            if i == 0:
                adjusted_template['x'] = char_centers[i][0] - template['w'] / 2 - (new_spacing - avg_spacing) * 1.5
            elif i == 5:
                adjusted_template['x'] = char_centers[i][0] - template['w'] / 2 + (new_spacing - avg_spacing) * 1.5
            else:
                offset = (new_spacing - avg_spacing) * (i - 2.5)
                adjusted_template['x'] = char_centers[i][0] - template['w'] / 2 + offset
            adjusted_templates.append(adjusted_template)
            
        return adjusted_templates

    def _filter_by_template_matching_with_fallback(self, regions, centroids, image_width, image_height):
        if not regions or not centroids:
            return [], [], 1.0
            
        spacing_factors = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        
        for spacing_factor in spacing_factors:
            adjusted_templates = self._adjust_template_spacing(CHAR_TEMPLATES, spacing_factor)
            
            template_boxes = []
            for template in adjusted_templates:
                x_px = int(template['x'] / 100 * image_width)
                y_px = int(template['y'] / 100 * image_height)
                w_px = int(template['w'] / 100 * image_width)
                h_px = int(template['h'] / 100 * image_height)
                
                template_boxes.append({
                    'name': template['name'],
                    'x': x_px, 'y': y_px, 'w': w_px, 'h': h_px
                })
            
            template_matches = {}
            for template in template_boxes:
                template_matches[template['name']] = False
                for centroid in centroids:
                    cx, cy = centroid
                    if (template['x'] <= cx <= template['x'] + template['w'] and
                        template['y'] <= cy <= template['y'] + template['h']):
                        template_matches[template['name']] = True
                        break
            
            if all(template_matches.values()):
                filtered_regions = []
                filtered_centroids = []
                for region, centroid in zip(regions, centroids):
                    cx, cy = centroid
                    in_any_template = False
                    for template in template_boxes:
                        if (template['x'] <= cx <= template['x'] + template['w'] and
                            template['y'] <= cy <= template['y'] + template['h']):
                            in_any_template = True
                            break
                    if in_any_template:
                        filtered_regions.append(region)
                        filtered_centroids.append(centroid)
                return filtered_regions, filtered_centroids, spacing_factor
                
        return [], [], 1.0

    def _recognize_characters_from_regions(self, filtered_regions, filtered_centroids, binary_denoised, image_width, image_height):
        recognition_results = {}
        adjusted_templates = self._adjust_template_spacing(CHAR_TEMPLATES, 1.0)
        
        for template in adjusted_templates:
            x_px = int(template['x'] / 100 * image_width)
            y_px = int(template['y'] / 100 * image_height)
            w_px = int(template['w'] / 100 * image_width)
            h_px = int(template['h'] / 100 * image_height)
            
            template_regions = []
            for region, centroid in zip(filtered_regions, filtered_centroids):
                cx, cy = centroid
                if (x_px <= cx <= x_px + w_px and y_px <= cy <= y_px + h_px):
                    template_regions.append(region)
            
            if template_regions:
                best_match = None
                best_similarity = -1
                best_char = None
                chinese_templates = ['guangdong', 'zhou', 'foshan', 'shan', 'guang', 'fo', 'shan']
                
                for region in template_regions:
                    x, y, w, h = region['bounding_box']
                    char_region = binary_denoised[y:y+h, x:x+w]
                    char_resized = cv2.resize(char_region, (40, 60))
                    
                    for template_name, template_img in self.templates.items():
                        if template['name'] in ['汉字1', '汉字2']:
                            if template_name not in chinese_templates:
                                continue
                                
                        similarity = cv2.matchTemplate(char_resized, template_img, cv2.TM_CCOEFF_NORMED)
                        max_similarity = np.max(similarity)
                        
                        if max_similarity > best_similarity:
                            best_similarity = max_similarity
                            best_match = region
                            best_char = template_name
                            
                if best_char:
                    char_map = {'guangdong': '广', 'zhou': '州', 'foshan': '佛', 'shan': '山'}
                    final_char = char_map.get(best_char, best_char)
                    recognition_results[template['name']] = {
                        'character': final_char,
                        'similarity': best_similarity,
                        'region': best_match
                    }
                    
        return recognition_results

    def _process_results(self, recognition_results):
        template_order = ['汉字1', '汉字2', '字符1', '字符2', '字符3', '字符4', '字符5', '字符6']
        corrected_results = recognition_results.copy()
        
        # 汉字修正逻辑
        if '汉字1' in recognition_results and '汉字2' in recognition_results:
            char1 = recognition_results['汉字1']['character']
            char2 = recognition_results['汉字2']['character']
            
            if char1 in ['fo', 'shan'] or char2 in ['fo', 'shan']:
                corrected_results['汉字1']['character'] = '佛'
                corrected_results['汉字2']['character'] = '山'
            elif char1 in ['guang', 'zhou'] or char2 in ['guang', 'zhou']:
                corrected_results['汉字1']['character'] = '广'
                corrected_results['汉字2']['character'] = '州'
        
        plate_number = ""
        for template_name in template_order:
            if template_name in corrected_results:
                plate_number += corrected_results[template_name]['character']
            else:
                plate_number += "?"
                
        return plate_number
