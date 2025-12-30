
import cv2
import numpy as np
import os
import glob
from pathlib import Path
from typing import Dict, Tuple, List
import shutil

class ZoningRecognizer:
    def __init__(self, template_dir: str):
        self.templates: Dict[str, np.ndarray] = {}
        self.template_features: Dict[str, np.ndarray] = {}
        self.template_dir = template_dir
        self.grid_rows = 6  # 网格行数
        self.grid_cols = 4  # 网格列数
        self.load_templates()

    def load_templates(self):
        """加载模板并计算特征"""
        template_files = glob.glob(os.path.join(self.template_dir, "*.png"))
        if not template_files:
            print(f"[Warning] No templates found in {self.template_dir}")
            return

        print(f"[*] Loading {len(template_files)} templates for Zoning...")
        for fpath in template_files:
            fname = Path(fpath).stem
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # 统一调整为固定大小 (比如 60x40)
            img_resized = cv2.resize(img, (40, 60))
            
            # 确保存储的是黑底白字 (Zoning是统计白色像素密度)
            if np.mean(img_resized) > 127:
                 img_resized = cv2.bitwise_not(img_resized)
                 
            self.templates[fname] = img_resized
            self.template_features[fname] = self._compute_zoning_features(img_resized)

    def _compute_zoning_features(self, img: np.ndarray) -> np.ndarray:
        """
        计算 Zoning 特征
        将图像划分为 grid_rows x grid_cols 的网格
        计算每个网格内白色像素的比例 (0.0 - 1.0)
        返回一个一维特征向量
        """
        h, w = img.shape
        dy = h // self.grid_rows
        dx = w // self.grid_cols
        
        features = []
        
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                # 提取当前网格的 ROI
                y_start, y_end = r * dy, (r + 1) * dy
                x_start, x_end = c * dx, (c + 1) * dx
                
                roi = img[y_start:y_end, x_start:x_end]
                
                # 计算白色像素占比 (密度)
                # roi > 127 会得到一个布尔矩阵，mean() 即为 True 的比例
                density = np.mean(roi > 127)
                features.append(density)
                
        return np.array(features, dtype=np.float32)

    def recognize_char(self, char_img: np.ndarray, is_chinese: bool = False) -> Tuple[str, float]:
        """
        识别字符
        1. 计算 char_img 的 Zoning 特征
        2. 与所有模板特征计算欧氏距离
        3. 返回距离最近的字符 (距离越小越好，这里转换为相似度)
        """
        # 预处理：Resize
        target_size = (40, 60)
        if char_img.shape != target_size:
            char_img = cv2.resize(char_img, target_size)
            
        # 预处理：确保黑底白字
        # 简单判断：如果边缘平均值很高，可能是白底，需要反转
        if np.mean(char_img) > 127: # 简单粗暴判断，实际建议用边缘检测
             pass # 假设传入已经是黑底白字了，或者由外部控制
             
        # 计算特征
        input_features = self._compute_zoning_features(char_img)
        
        best_dist = float('inf')
        best_char = "?"
        
        chinese_keys = ['guangdong', 'zhou', 'foshan', 'shan', 'guang', 'fo']
        
        for name, tmpl_features in self.template_features.items():
            # 过滤逻辑
            is_tmpl_chinese = any(k in name for k in chinese_keys)
            if is_chinese and not is_tmpl_chinese: continue
            if not is_chinese and is_tmpl_chinese: continue
            
            # 计算欧氏距离
            dist = np.linalg.norm(input_features - tmpl_features)
            
            if dist < best_dist:
                best_dist = dist
                best_char = name
                
        # 将距离转换为置信度 (距离越小，置信度越高)
        # 经验公式：confidence = 1 / (1 + dist)
        confidence = 1.0 / (1.0 + best_dist)
        
        # 映射回真实字符
        char_map = {
            'guangdong': '广', 'zhou': '州', 'foshan': '佛', 'shan': '山',
            'guang': '广', 'fo': '佛'
        }
        final_char = char_map.get(best_char, best_char)
        
        return final_char, confidence

# ==========================================
# 复用 test_segment 的逻辑来获取字符图像
# ==========================================
import test_segment # 导入现有的分割逻辑

def run_zoning_test():
    print("=== 开始 Zoning (网格特征) 识别测试 ===")
    
    # 1. 初始化识别器
    recognizer = ZoningRecognizer(template_dir="char_templates")
    
    # 2. 读取图像并分割 (借用 test_segment 的功能)
    img_path = r"d:\Digital_Image_Processing\FinalProject\plate_recog\input_images\normal_plate\normal_plate_1.jpg"
    print(f"[*] 读取图像: {img_path}")
    
    original_img = cv2.imread(img_path)
    if original_img is None:
        print("Error: 无法读取图像")
        return

    # 调用 test_segment 的处理流程
    # 注意：我们需要修改 test_segment 让他能返回中间结果，或者我们直接复制关键代码
    # 为了方便，这里直接调用 test_segment.process_plate_image (如果它没有这个函数，我们就手动写一下流程)
    
    # 这里的流程参考 test_segment.py 的 main
    # locate_plate_region 返回的是字典 {'bbox': (x,y,w,h), 'image': plate_img}
    plate_result = test_segment.locate_plate_region(cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY), debug=False)
    
    if not plate_result:
        print("未找到车牌")
        return
        
    rect = plate_result['bbox']
    x, y, w, h = rect
    plate_img = original_img[y:y+h, x:x+w]
    
    # 预处理 (使用 test_segment.preprocess_plate，注意它需要灰度图输入)
    # 先转灰度
    plate_img_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # 由于 test_segment.py 中没有封装 preprocess_plate 函数，这里直接复制其预处理逻辑
    # 3.1 尺寸归一化 (固定高度为 140px，宽度按比例缩放)
    target_h = 140
    h_raw, w_raw = plate_img_gray.shape
    scale = target_h / h_raw
    target_w = int(w_raw * scale)
    
    plate_img_gray = cv2.resize(plate_img_gray, (target_w, target_h))
    
    # 3.2 图像增强 (高斯模糊 + 二值化 + 形态学)
    # 高斯模糊
    blurred = cv2.GaussianBlur(plate_img_gray, (3, 3), 0)
    
    # 二值化 (使用固定阈值 140，与 character_recognizer 保持一致)
    _, plate_binary = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)
    
    # [新增] 智能背景反转：确保是黑底白字
    h_bin, w_bin = plate_binary.shape
    border_mean = (np.mean(plate_binary[0, :]) + np.mean(plate_binary[h_bin-1, :]) + 
                   np.mean(plate_binary[:, 0]) + np.mean(plate_binary[:, w_bin-1])) / 4.0
                   
    if border_mean > 127:
        # print("    -> 检测到白底，自动反转为黑底白字...")
        plate_binary = cv2.bitwise_not(plate_binary)
    
    # 形态学操作 (开运算去除噪点)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed_plate = cv2.morphologyEx(plate_binary, cv2.MORPH_OPEN, kernel)
    
    # 分割参数
    CHAR_CONFIGS = [
        {'name': 'top_1',    'x_pct': 0.170, 'y_pct': 0.28, 'w_pct': 0.10, 'h_pct': 0.35},
        {'name': 'top_2',    'x_pct': 0.285, 'y_pct': 0.28, 'w_pct': 0.10, 'h_pct': 0.35},
        {'name': 'bottom_1', 'x_pct': 0.040, 'y_pct': 0.60, 'w_pct': 0.10, 'h_pct': 0.35},
        {'name': 'bottom_2', 'x_pct': 0.155, 'y_pct': 0.60, 'w_pct': 0.10, 'h_pct': 0.35},
        {'name': 'bottom_3', 'x_pct': 0.330, 'y_pct': 0.60, 'w_pct': 0.10, 'h_pct': 0.35},
        {'name': 'bottom_4', 'x_pct': 0.445, 'y_pct': 0.60, 'w_pct': 0.10, 'h_pct': 0.35},
        {'name': 'bottom_5', 'x_pct': 0.560, 'y_pct': 0.60, 'w_pct': 0.10, 'h_pct': 0.35},
        {'name': 'bottom_6', 'x_pct': 0.675, 'y_pct': 0.60, 'w_pct': 0.10, 'h_pct': 0.35},
    ]
    
    ph, pw = processed_plate.shape
    
    print("\n[识别结果]")
    print(f"{'位置':<15} | {'字符':<5} | {'置信度':<10}")
    print("-" * 40)
    
    results = []
    
    for config in CHAR_CONFIGS:
        cx = int(config['x_pct'] * pw)
        cy = int(config['y_pct'] * ph)
        cw = int(config['w_pct'] * pw)
        ch = int(config['h_pct'] * ph)
        
        char_roi = processed_plate[cy:cy+ch, cx:cx+cw]
        
        # 识别
        is_chinese = 'top' in config['name']
        char_result, conf = recognizer.recognize_char(char_roi, is_chinese=is_chinese)
        
        print(f"{config['name']:<15} | {char_result:<5} | {conf:.4f}")
        results.append(char_result)
        
    print("-" * 40)
    print(f"最终结果: {''.join(results)}")

if __name__ == "__main__":
    run_zoning_test()
