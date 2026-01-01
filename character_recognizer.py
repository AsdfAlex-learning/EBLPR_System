import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

RATIO_BOTTOM = 1.55
RATIO_TOP = 0.95

# 字符位置配置
CHAR_CONFIGS = [
    {'name': '01_top_1', 'center_x': 0.37, 'center_y': 0.25, 'w': 0.13},
    {'name': '02_top_2', 'center_x': 0.62, 'center_y': 0.25, 'w': 0.13},
    {'name': '03_bottom_1', 'center_x': 0.1, 'center_y': 0.70, 'w': 0.15},
    {'name': '04_bottom_2', 'center_x': 0.26, 'center_y': 0.70, 'w': 0.15},
    {'name': '05_bottom_3', 'center_x': 0.42, 'center_y': 0.70, 'w': 0.15},
    {'name': '06_bottom_4', 'center_x': 0.58, 'center_y': 0.70, 'w': 0.15},
    {'name': '07_bottom_5', 'center_x': 0.74, 'center_y': 0.70, 'w': 0.15},
    {'name': '08_bottom_6', 'center_x': 0.902, 'center_y': 0.70, 'w': 0.15},
]

class CharacterRecognizer:
    def __init__(self, template_dir: str = "char_templates"):
        self.template_dir = Path(template_dir)
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, np.ndarray]:
        templates = {}
        if not self.template_dir.exists():
            print(f"[ERROR] 模板目录不存在: {self.template_dir}")
            return templates
            
        template_files = list(self.template_dir.glob("*.png"))
        template_files = [f for f in template_files if f.name != "template_preview.png"]
        
        for template_file in template_files:
            template_name = template_file.stem
            template_img = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
            if template_img is not None:
                templates[template_name] = template_img
                
        return templates

    def recognize(self, plate_image: np.ndarray, debug_output_dir: Optional[Path] = None, 
                  configs: List[Dict] = None, ratio_bottom: float = None, ratio_top: float = None,
                  return_confidence: bool = False) -> Any:

        if plate_image is None:
            return ("", 0.0) if return_confidence else ""

        current_configs = configs if configs is not None else CHAR_CONFIGS
        current_ratio_bottom = ratio_bottom if ratio_bottom is not None else RATIO_BOTTOM
        current_ratio_top = ratio_top if ratio_top is not None else RATIO_TOP
        
        # 预处理 (归一化/去噪/二值化)
        target_h = 140
        h_raw, w_raw = plate_image.shape
        scale = target_h / h_raw
        target_w = int(w_raw * scale)
        
        plate_image_resized = cv2.resize(plate_image, (target_w, target_h))
        blurred = cv2.GaussianBlur(plate_image_resized, (3, 3), 0)
        _, binary_plate = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)
        
        # 背景色检测与反转
        h_bin, w_bin = binary_plate.shape
        border_mean = (np.mean(binary_plate[0, :]) + np.mean(binary_plate[h_bin-1, :]) + 
                       np.mean(binary_plate[:, 0]) + np.mean(binary_plate[:, w_bin-1])) / 4.0
                       
        if border_mean > 127:
            binary_plate = cv2.bitwise_not(binary_plate)
        
        binary_plate = cv2.medianBlur(binary_plate, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed_plate = cv2.morphologyEx(binary_plate, cv2.MORPH_OPEN, kernel)
        
        plate_h, plate_w = processed_plate.shape
        recognition_results = {}
        
        if debug_output_dir:
            debug_output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_output_dir / "plate_input.png"), plate_image)
            cv2.imwrite(str(debug_output_dir / "plate_processed.png"), processed_plate)

        # 字符提取与识别
        for cfg in current_configs:
            name = cfg['name']
            cx_pct = cfg['center_x']
            cy_pct = cfg['center_y']
            w_pct = cfg['w']
            
            ratio = current_ratio_top if 'top' in name else current_ratio_bottom
            
            box_w = int(plate_w * w_pct)
            box_h = int(box_w * ratio)
            center_x = int(plate_w * cx_pct)
            center_y = int(plate_h * cy_pct)
            
            x1 = max(0, center_x - box_w // 2)
            y1 = max(0, center_y - box_h // 2)
            x2 = min(plate_w, x1 + box_w)
            y2 = min(plate_h, y1 + box_h)
            
            real_w = x2 - x1
            real_h = y2 - y1
            
            if real_w > 0 and real_h > 0:
                char_roi_rough = processed_plate[y1:y2, x1:x2]
                contours, _ = cv2.findContours(char_roi_rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                best_rect = None
                max_score = -1
                roi_h, roi_w = char_roi_rough.shape
                roi_center_x, roi_center_y = roi_w / 2, roi_h / 2
                
                for cnt in contours:
                    cx, cy, cw, ch = cv2.boundingRect(cnt)
                    if cw * ch < 20: continue
                    area = cw * ch
                    cnt_center_x = cx + cw / 2
                    cnt_center_y = cy + ch / 2
                    dist = ((cnt_center_x - roi_center_x)**2 + (cnt_center_y - roi_center_y)**2)**0.5
                    score = area / (1 + 0.1 * dist)
                    if score > max_score:
                        max_score = score
                        best_rect = (cx, cy, cw, ch)
                
                if best_rect:
                    rx, ry, rw, rh = best_rect
                    x1 += rx
                    y1 += ry
                    x2 = x1 + rw
                    y2 = y1 + rh
                
                char_img = processed_plate[y1:y2, x1:x2]
                if debug_output_dir:
                    cv2.imwrite(str(debug_output_dir / f"cut_{name}.png"), char_img)
                
                char_code, confidence = self._recognize_single_char(char_img, name)
                recognition_results[name] = {
                    'char': char_code,
                    'conf': confidence,
                    'box': (x1, y1, x2, y2)
                }
            else:
                recognition_results[name] = {'char': "?", 'conf': 0.0, 'box': (0, 0, 0, 0)}

        # 后处理修正
        self._apply_plate_logic(recognition_results)
        
        debug_vis_img = None
        if debug_output_dir:
            debug_vis_img = cv2.cvtColor(processed_plate, cv2.COLOR_GRAY2BGR)

        sorted_keys = sorted(recognition_results.keys())
        for key in sorted_keys:
            res = recognition_results[key]
            char_str = res['char']
            conf_val = res['conf']
            
            if debug_vis_img is not None and res['box'][2] > 0:
                x1, y1, x2, y2 = res['box']
                cv2.rectangle(debug_vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{char_str}:{conf_val:.2f}"
                text_y = y1 - 10 if y1 > 20 else y2 + 20
                cv2.putText(debug_vis_img, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if debug_output_dir and debug_vis_img is not None:
            cv2.imwrite(str(debug_output_dir / "plate_recog_debug_vis.png"), debug_vis_img)

        plate_number = self._format_result(recognition_results)
        
        if return_confidence:
            total_conf = sum(r['conf'] for r in recognition_results.values())
            count = len(recognition_results)
            avg_conf = total_conf / count if count > 0 else 0.0
            if "?" in plate_number:
                avg_conf *= 0.5
            return plate_number, avg_conf
            
        return plate_number

    def _apply_plate_logic(self, results: Dict[str, Dict]):
        r1 = results.get('01_top_1')
        r2 = results.get('02_top_2')
        
        if not r1 or not r2:
            return

        c1 = r1['char']
        c2 = r2['char']
        
        # 修正中文字符识别结果
        if c1 in ['佛', '山'] or c2 in ['佛', '山']:
            print("触发城市前缀修正: 佛山")
            r1['char'] = '佛'
            r2['char'] = '山'
        elif c1 in ['广', '州'] or c2 in ['广', '州']:
            print("触发城市前缀修正: 广州")
            r1['char'] = '广'
            r2['char'] = '州'

    def _recognize_single_char(self, char_img: np.ndarray, char_pos_name: str) -> Tuple[str, float]:
        target_size = (40, 60)
        char_resized = cv2.resize(char_img, target_size)
        
        # 中文关键词
        chinese_keys = ['guangdong', 'zhou', 'foshan', 'shan', 'guang', 'fo']
        
        allow_chinese = False
        allow_alphanum = False
        
        if 'top' in char_pos_name:
            allow_chinese = True
        else:
            allow_alphanum = True
            
        best_score = -1.0
        best_char = "?"
        
        for template_name, template_img in self.templates.items():
            is_chinese_template = any(k in template_name for k in chinese_keys)
            if allow_chinese and not is_chinese_template: continue
            if allow_alphanum and is_chinese_template: continue
            
            if template_img.shape != target_size:
                template_img = cv2.resize(template_img, target_size)
            
            if 'top' in char_pos_name:
                search_h, search_w = 70, 50
            else:
                search_h, search_w = 64, 44
                
            char_search = cv2.resize(char_img, (search_w, search_h))
            res = cv2.matchTemplate(char_search, template_img, cv2.TM_CCOEFF_NORMED)
            final_score = float(np.max(res))
            
            if final_score > best_score:
                best_score = final_score
                best_char = template_name
        
        char_map = {
            'guangdong': '广', 'zhou': '州', 'foshan': '佛', 'shan': '山',
            'guang': '广', 'fo': '佛'
        }
        final_char = char_map.get(best_char, best_char)
        
        return final_char, best_score

    # 格式化识别结果
    def _format_result(self, results: Dict[str, Dict]) -> str:
        sorted_keys = sorted(results.keys())
        return "".join([results[key]['char'] for key in sorted_keys])
