"""
配置文件：定义车牌识别后端的核心路径配置
- INPUT_IMAGES_DIR: 上传的原始图像存储目录
- PROCESSED_IMAGES_DIR: MATLAB处理后的车牌图像存储目录
- 补充通用配置（如日志级别），方便后续扩展
"""
import os
from pathlib import Path

# ==================== 基础路径配置 ====================
# 获取当前config.py文件所在的目录（适配不同运行环境）
BASE_DIR = Path(__file__).resolve().parent

# 原始上传图像存储目录（默认：backend/input_images）
INPUT_IMAGES_DIR = os.path.join(BASE_DIR, "input_images")

# MATLAB处理后的车牌图像存储目录（默认：backend/processed_images）
PROCESSED_IMAGES_DIR = os.path.join(BASE_DIR, "processed_images")

# ==================== 可选扩展配置（保留兼容性） ====================
# 如果你后续需要恢复OCR功能，可保留以下配置（当前main.py已移除，仅作备份）
TESSERACT_CMD = "tesseract"  # Tesseract OCR可执行文件路径（Windows需指定完整路径，如"C:/Program Files/Tesseract-OCR/tesseract.exe"）
OCR_CONFIG = "--psm 8 -l chi_sim+eng"  # OCR配置：psm8表示单字符识别，语言包包含中文+英文

# ==================== 日志配置（可选） ====================
LOG_LEVEL = "INFO"  # 日志级别：DEBUG/INFO/WARNING/ERROR
LOG_FILE = os.path.join(BASE_DIR, "logs", "lpr_backend.log")  # 日志文件存储路径

# ==================== 路径初始化（可选，可移到main.py） ====================
def init_directories():
    """初始化所需的目录（如果不存在则创建）"""
    # 创建输入/输出图像目录
    Path(INPUT_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    Path(PROCESSED_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    # 创建日志目录（如果启用日志文件）
    Path(os.path.dirname(LOG_FILE)).mkdir(parents=True, exist_ok=True)

# 可选：导入时自动初始化目录（也可在main.py中调用init_directories()）
# if __name__ != "__main__":
#     init_directories()