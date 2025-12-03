import os
import platform
import shutil

def find_tesseract_executable():
    """自动检测Tesseract-OCR可执行文件路径（跨平台）"""
    system = platform.system()
    
    # 首先尝试从环境变量或PATH中查找
    tesseract_cmd = shutil.which('tesseract')
    if tesseract_cmd:
        return tesseract_cmd
    
    # 如果PATH中找不到，尝试常见安装路径
    if system == 'Windows':
        common_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            os.path.expanduser(r'~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'),
        ]
    elif system == 'Darwin':  # macOS
        common_paths = [
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract',
            '/usr/bin/tesseract',
        ]
    else:  # Linux
        common_paths = [
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
        ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    # 如果都找不到，返回默认Windows路径（向后兼容）
    if system == 'Windows':
        return r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    else:
        return 'tesseract'  # 假设在PATH中

# Tesseract-OCR配置（自动检测）
TESSERACT_CMD = find_tesseract_executable()

# 输入输出路径配置
INPUT_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'input_images')
OUTPUT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'output_results')
PROCESSED_IMAGES_DIR = os.path.join(OUTPUT_RESULTS_DIR, 'processed_images')

# 确保输出目录存在
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)

# OCR配置
OCR_CONFIG = r'--psm 7 --oem 3'  # PSM 7适用于单行文本
