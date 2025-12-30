"""
字符识别模块 (重构版)
基于 03_Test_image_features.py 的车牌字符定位与识别逻辑
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

# ==========================================
# [默认参数设置] - 可在外部修改
# ==========================================

# 字符框宽高比设置
# 下层字符 (数字/字母) 比例：1.55 (用户指定)
RATIO_BOTTOM = 1.55
# 上层字符 (汉字/字母) 比例：0.95 (默认方形略扁)
RATIO_TOP = 0.95

# 字符分割配置 (双层车牌 8 字符模板)
CHAR_CONFIGS = [
    # --- 第一行 (Top Row) ---
    # 字符 1 (汉字，如 "苏")
    {'name': '01_top_1', 'center_x': 0.37, 'center_y': 0.25, 'w': 0.18},
    # 字符 2 (字母，如 "E")
    {'name': '02_top_2', 'center_x': 0.62, 'center_y': 0.25, 'w': 0.18},
    
    # --- 第二行 (Bottom Row) ---
    # 字符 3
    {'name': '03_bottom_1', 'center_x': 0.1, 'center_y': 0.70, 'w': 0.15},
    # 字符 4
    {'name': '04_bottom_2', 'center_x': 0.26, 'center_y': 0.70, 'w': 0.15},
    # 字符 5
    {'name': '05_bottom_3', 'center_x': 0.42, 'center_y': 0.70, 'w': 0.15},
    # 字符 6
    {'name': '06_bottom_4', 'center_x': 0.58, 'center_y': 0.70, 'w': 0.15},
    # 字符 7
    {'name': '07_bottom_5', 'center_x': 0.74, 'center_y': 0.70, 'w': 0.15},
    # 字符 8
    {'name': '08_bottom_6', 'center_x': 0.902, 'center_y': 0.70, 'w': 0.15},
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
                # 确保模板大小也是 40x60，或者在匹配时统一
                templates[template_name] = template_img
                
        return templates

    def recognize(self, plate_image: np.ndarray, debug_output_dir: Optional[Path] = None, 
                  configs: List[Dict] = None, ratio_bottom: float = None, ratio_top: float = None) -> str:
        """
        识别车牌图像中的字符
        Args:
            plate_image: 车牌灰度图像
            debug_output_dir: 调试输出目录
            configs: 可选，覆盖默认的 CHAR_CONFIGS
            ratio_bottom: 可选，覆盖默认的 RATIO_BOTTOM
            ratio_top: 可选，覆盖默认的 RATIO_TOP
            
        Returns:
            str: 识别出的车牌号码
        """
        if plate_image is None:
            return ""

        # 使用传入的配置或默认配置
        current_configs = configs if configs is not None else CHAR_CONFIGS
        current_ratio_bottom = ratio_bottom if ratio_bottom is not None else RATIO_BOTTOM
        current_ratio_top = ratio_top if ratio_top is not None else RATIO_TOP
        
        # --- 图像预处理增强 ---
        # 0. 尺寸归一化 (关键步骤：与调试脚本保持一致)
        # 固定高度为 140px，宽度按比例缩放
        target_h = 140
        h_raw, w_raw = plate_image.shape
        scale = target_h / h_raw
        target_w = int(w_raw * scale)
        
        plate_image_resized = cv2.resize(plate_image, (target_w, target_h))
        
        # 1. 高斯模糊去噪
        blurred = cv2.GaussianBlur(plate_image_resized, (3, 3), 0)
        
        # 2. 二值化
        # 用户说明：Otsu 会忽略第二个参数(140)，如果需要手动调整阈值，请移除 cv2.THRESH_OTSU
        # _, binary_plate = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 使用 THRESH_BINARY_INV (反转) 来让白底黑字变为黑底白字？
        # 不，通常车牌是蓝底白字 -> 二值化后字是白，底是黑。
        # 如果是白底黑字的车牌，THRESH_BINARY 会得到白底黑字。
        # 我们这里统一使用 THRESH_BINARY，然后智能反转背景。
        _, binary_plate = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)
        
        # [新增] 智能背景反转：确保是黑底白字
        # 检查边缘像素，如果是白色，说明背景是白，需要反转
        h_bin, w_bin = binary_plate.shape
        border_mean = (np.mean(binary_plate[0, :]) + np.mean(binary_plate[h_bin-1, :]) + 
                       np.mean(binary_plate[:, 0]) + np.mean(binary_plate[:, w_bin-1])) / 4.0
                       
        if border_mean > 127:
            # print("  [Debug] Detected White Background, Inverting to Black Background...")
            binary_plate = cv2.bitwise_not(binary_plate)
        
        # 3. 形态学操作 (开运算去除细小噪点)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed_plate = cv2.morphologyEx(binary_plate, cv2.MORPH_OPEN, kernel)
        
        # 使用处理后的图像进行分割和识别
        # 注意：这里我们使用 processed_plate 替代原始的 plate_image
        plate_h, plate_w = processed_plate.shape
        
        recognition_results = {}
        
        if debug_output_dir:
            debug_output_dir.mkdir(parents=True, exist_ok=True)
            # 保存原图
            cv2.imwrite(str(debug_output_dir / "plate_input.png"), plate_image)
            cv2.imwrite(str(debug_output_dir / "plate_processed.png"), processed_plate)

        # 遍历每个配置的字符框进行分割和识别
        for cfg in current_configs:
            name = cfg['name']
            cx_pct = cfg['center_x']
            cy_pct = cfg['center_y']
            w_pct = cfg['w']
            
            # 确定使用哪个宽高比
            if 'top' in name:
                ratio = current_ratio_top
            else:
                ratio = current_ratio_bottom
            
            # 计算像素坐标
            box_w = int(plate_w * w_pct)
            box_h = int(box_w * ratio)
            
            center_x = int(plate_w * cx_pct)
            center_y = int(plate_h * cy_pct)
            
            x1 = center_x - box_w // 2
            y1 = center_y - box_h // 2
            
            # 边界检查
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(plate_w, x1 + box_w)
            y2 = min(plate_h, y1 + box_h)
            
            # 提取字符图像
            if x2 > x1 and y2 > y1:
                # 使用处理后的二值图像进行切割
                char_img = processed_plate[y1:y2, x1:x2]
                
                if debug_output_dir:
                    cv2.imwrite(str(debug_output_dir / f"cut_{name}.png"), char_img)
                
                # 识别单个字符
                char_code, confidence = self._recognize_single_char(char_img, name)
                recognition_results[name] = {
                    'char': char_code,
                    'conf': confidence,
                    'box': (x1, y1, x2, y2)
                }
            else:
                recognition_results[name] = {'char': "?", 'conf': 0.0, 'box': (0, 0, 0, 0)}

        # 结果修正 (佛山/广州逻辑)
        self._apply_plate_logic(recognition_results)
        
        # --- 调试输出增强 ---
        print("\n" + "="*30)
        print(" [详细识别结果]")
        print(f" {'位置':<15} | {'字符':<5} | {'置信度':<10}")
        print("-" * 35)
        
        # 准备调试绘图 (转为彩色以便绘制彩色文字)
        debug_vis_img = None
        if debug_output_dir:
            debug_vis_img = cv2.cvtColor(processed_plate, cv2.COLOR_GRAY2BGR)

        sorted_keys = sorted(recognition_results.keys())
        for key in sorted_keys:
            res = recognition_results[key]
            char_str = res['char']
            conf_val = res['conf']
            print(f" {key:<15} | {char_str:<5} | {conf_val:.4f}")
            
            # 绘制到调试图
            if debug_vis_img is not None and res['box'][2] > 0:
                x1, y1, x2, y2 = res['box']
                # 画框
                cv2.rectangle(debug_vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 写字 (字符 + 置信度)
                label = f"{char_str}:{conf_val:.2f}"
                # 为了防止文字出界，根据位置调整
                text_y = y1 - 10 if y1 > 20 else y2 + 20
                cv2.putText(debug_vis_img, label, (x1, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        print("="*30 + "\n")
        
        if debug_output_dir and debug_vis_img is not None:
            cv2.imwrite(str(debug_output_dir / "plate_recog_debug_vis.png"), debug_vis_img)

        # 整合结果
        plate_number = self._format_result(recognition_results)
        return plate_number

    def _apply_plate_logic(self, results: Dict[str, Dict]):
        """应用特殊的车牌逻辑修正结果"""
        # 获取前两个字符 (假设是 01_top_1 和 02_top_2)
        # results 结构: {'01_top_1': {'char': '广', 'conf': 0.8, ...}, ...}
        
        r1 = results.get('01_top_1')
        r2 = results.get('02_top_2')
        
        if not r1 or not r2:
            return

        c1 = r1['char']
        c2 = r2['char']
        
        # 逻辑：如果含有 佛 或 山 -> 佛山
        if c1 in ['佛', '山'] or c2 in ['佛', '山']:
            r1['char'] = '佛'
            r2['char'] = '山'
            # 可选：标记为逻辑修正
            r1['logic_fixed'] = True
            r2['logic_fixed'] = True
            
        # 逻辑：如果含有 广 或 州 -> 广州
        elif c1 in ['广', '州'] or c2 in ['广', '州']:
            r1['char'] = '广'
            r2['char'] = '州'
            r1['logic_fixed'] = True
            r2['logic_fixed'] = True

    def _recognize_single_char(self, char_img: np.ndarray, char_pos_name: str) -> Tuple[str, float]:
        """
        识别单个字符图像
        结合多种方法：
        1. 模板匹配 (Template Matching) - 适用于形状和位置比较固定的情况
        2. 特征匹配 (Hu Moments / SIFT) - 对位置、缩放、旋转不敏感
        """
        
        # --- 1. 预处理 ---
        # 调整到模板大小
        target_size = (40, 60)
        char_resized = cv2.resize(char_img, target_size)
        
        # --- 2. 准备候选模板 ---
        # 定义汉字模板关键字
        chinese_keys = ['guangdong', 'zhou', 'foshan', 'shan', 'guang', 'fo']
        
        # 确定当前框允许的字符类型
        allow_chinese = False
        allow_alphanum = False
        
        if 'top' in char_pos_name:
            allow_chinese = True
        else:
            allow_alphanum = True
            
        best_score = -1.0 # 综合评分 (越高越好)
        best_char = "?"
        
        # --- 3. 遍历模板进行匹配 ---
        for template_name, template_img in self.templates.items():
            # 过滤逻辑
            is_chinese_template = any(k in template_name for k in chinese_keys)
            if allow_chinese and not is_chinese_template: continue
            if allow_alphanum and is_chinese_template: continue
            
            # 确保模板也是 target_size
            if template_img.shape != target_size:
                template_img = cv2.resize(template_img, target_size)
            
            # === 方法 A: 优化的模板匹配 (允许微小平移) ===
            # 不是只匹配一次，而是在小范围内滑动匹配
            # 这里的 matchTemplate 本身就是滑动匹配，但前提是 char_resized 要比 template_img 大
            # 由于我们强制 resize 成了一样大，所以 matchTemplate 退化成了像素对比。
            # 改进：将 char_img resize 得稍微大一点点 (比如 44x64)，然后用 40x60 的模板去匹配
            # 这样允许 4px 的位移容错。
            
            search_h, search_w = 64, 44
            char_search = cv2.resize(char_img, (search_w, search_h))
            
            res = cv2.matchTemplate(char_search, template_img, cv2.TM_CCOEFF_NORMED)
            tm_score = float(np.max(res))
            
            # === 方法 B: 轮廓特征匹配 (Hu Moments) - 可选 ===
            # Hu 矩对平移、缩放、旋转不变。
            # cv2.matchShapes 返回值越小越相似 (0是完全相同)
            # hu_score_raw = cv2.matchShapes(char_resized, template_img, cv2.CONTOURS_MATCH_I1, 0)
            # hu_score = 1.0 / (1.0 + hu_score_raw) # 转换为 0-1 分数 (越高越好)
            
            # === 综合评分 ===
            # 目前先主要信赖允许位移的模板匹配
            final_score = tm_score 
            
            # 如果是汉字，由于笔画复杂，可以适当降低对位置的敏感度
            # if is_chinese_template:
            #     final_score = 0.7 * tm_score + 0.3 * hu_score
            
            if final_score > best_score:
                best_score = final_score
                best_char = template_name
        
        # 映射回真实字符
        char_map = {
            'guangdong': '广', 'zhou': '州', 'foshan': '佛', 'shan': '山',
            'guang': '广', 'fo': '佛'
        }
        final_char = char_map.get(best_char, best_char)
        
        return final_char, best_score

    def _format_result(self, results: Dict[str, Dict]) -> str:
        """将识别结果字典转换为字符串"""
        # 按照配置顺序排序
        sorted_keys = sorted(results.keys())
        
        final_str = ""
        for key in sorted_keys:
            final_str += results[key]['char']
            
        return final_str

