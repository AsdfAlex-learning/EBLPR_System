import os

# Tesseract-OCR配置
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 输入输出路径配置
INPUT_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'input_images')
OUTPUT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'output_results')
PROCESSED_IMAGES_DIR = os.path.join(OUTPUT_RESULTS_DIR, 'processed_images')

# 确保输出目录存在
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)

# OCR配置
OCR_CONFIG = r'--psm 7 --oem 3'  # PSM 7适用于单行文本
