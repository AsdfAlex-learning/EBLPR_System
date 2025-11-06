import pytesseract
from PIL import Image
import os
from backend.config import TESSERACT_CMD

def test_tesseract_installation():
    # 设置Tesseract-OCR路径
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    
    try:
        # 测试Tesseract版本
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract-OCR 版本: {version}")
        print("Tesseract-OCR 安装验证成功！")
        
        # 输出Tesseract-OCR路径
        print(f"Tesseract-OCR 路径: {TESSERACT_CMD}")
        
        return True
    except Exception as e:
        print(f"Tesseract-OCR 安装验证失败：{str(e)}")
        print("\n可能的解决方案：")
        print("1. 确保已安装Tesseract-OCR")
        print("2. 检查config.py中的TESSERACT_CMD路径是否正确")
        print("3. 确保Tesseract-OCR已添加到系统环境变量PATH中")
        return False

if __name__ == "__main__":
    test_tesseract_installation()