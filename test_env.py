"""
环境测试脚本 - 检查项目所需的所有依赖和配置是否正确安装 - AI生成
"""
import sys
import os
from pathlib import Path

# 颜色输出支持（Windows，可选）
try:
    import colorama  # type: ignore
    colorama.init()
    GREEN = colorama.Fore.GREEN
    RED = colorama.Fore.RED
    YELLOW = colorama.Fore.YELLOW
    RESET = colorama.Fore.RESET
except ImportError:
    GREEN = RED = YELLOW = RESET = ""

def print_success(message):
    print(f"{GREEN}✓ {message}{RESET}")

def print_error(message):
    print(f"{RED}✗ {message}{RESET}")

def print_warning(message):
    print(f"{YELLOW}⚠ {message}{RESET}")

def print_info(message):
    print(f"  {message}")

def test_python_version():
    """测试Python版本"""
    print("\n" + "="*60)
    print("1. 检查Python版本")
    print("="*60)
    version = sys.version_info
    print_info(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 8:
        print_success("Python版本符合要求 (>= 3.8)")
        return True
    else:
        print_error("Python版本不符合要求，需要Python 3.8或更高版本")
        return False

def test_core_packages():
    """测试核心Python包"""
    print("\n" + "="*60)
    print("2. 检查核心Python包")
    print("="*60)
    
    packages = {
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'pytesseract': 'Pytesseract',
        'cv2': 'OpenCV (opencv-python)',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'aiofiles': 'Aiofiles'
    }
    
    results = {}
    for module_name, display_name in packages.items():
        try:
            if module_name == 'cv2':
                import cv2  # type: ignore
                version = cv2.__version__
            elif module_name == 'PIL':
                from PIL import Image  # type: ignore
                import PIL  # type: ignore
                version = PIL.__version__
            elif module_name == 'numpy':
                import numpy  # type: ignore
                version = numpy.__version__
            else:
                module = __import__(module_name)
                version = getattr(module, '__version__', '未知版本')
            
            print_success(f"{display_name}: {version}")
            results[module_name] = True
        except ImportError as e:
            print_error(f"{display_name}: 未安装 - {str(e)}")
            results[module_name] = False
    
    return all(results.values())

def test_tesseract_installation():
    """测试Tesseract-OCR安装"""
    print("\n" + "="*60)
    print("3. 检查Tesseract-OCR")
    print("="*60)
    
    try:
        import pytesseract  # type: ignore
        from backend.config import TESSERACT_CMD
        
        # 设置Tesseract-OCR路径
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        
        # 测试Tesseract版本
        version = pytesseract.get_tesseract_version()
        print_success(f"Tesseract-OCR版本: {version}")
        print_info(f"Tesseract-OCR路径: {TESSERACT_CMD}")
        
        # 检查路径是否存在
        if os.path.exists(TESSERACT_CMD):
            print_success("Tesseract-OCR可执行文件存在")
        else:
            print_warning(f"Tesseract-OCR路径可能不正确: {TESSERACT_CMD}")
            print_info("请检查backend/config.py中的TESSERACT_CMD配置")
        
        return True
    except ImportError:
        print_error("pytesseract未安装")
        return False
    except Exception as e:
        print_error(f"Tesseract-OCR验证失败: {str(e)}")
        print_info("\n可能的解决方案：")
        print_info("1. 确保已安装Tesseract-OCR")
        print_info("2. 检查config.py中的TESSERACT_CMD路径是否正确")
        print_info("3. 确保Tesseract-OCR已添加到系统环境变量PATH中")
        return False

def test_matlab_engine():
    """测试MATLAB Engine（可选）"""
    return None

def test_project_structure():
    """测试项目目录结构"""
    print("\n" + "="*60)
    print("5. 检查项目目录结构")
    print("="*60)
    
    base_dir = Path(__file__).parent
    required_dirs = [
        'backend',
        'data/input_images',
        'data/output_results',
        'opencv_core',
        'opencv_core/char_templates',
        'frontend'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            print_success(f"目录存在: {dir_path}")
        else:
            print_error(f"目录不存在: {dir_path}")
            all_exist = False
    
    # 检查关键文件
    required_files = [
        'backend/main.py',
        'backend/config.py',
        'backend/image_utils.py',
        'backend/plate_detector.py',
        'backend/character_recognizer.py',
        'backend/opencv_processor.py',
        'backend/__init__.py'
    ]
    
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print_success(f"文件存在: {file_path}")
        else:
            print_warning(f"文件不存在: {file_path}")
    
    return all_exist

def test_backend_config():
    """测试后端配置"""
    print("\n" + "="*60)
    print("6. 检查后端配置")
    print("="*60)
    
    try:
        from backend.config import (
            TESSERACT_CMD,
            INPUT_IMAGES_DIR,
            PROCESSED_IMAGES_DIR,
            OCR_CONFIG
        )
        
        print_success("配置文件加载成功")
        print_info(f"输入图像目录: {INPUT_IMAGES_DIR}")
        print_info(f"处理后图像目录: {PROCESSED_IMAGES_DIR}")
        print_info(f"OCR配置: {OCR_CONFIG}")
        
        # 检查目录是否可写
        try:
            os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)
            test_file = os.path.join(PROCESSED_IMAGES_DIR, '.test_write')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print_success("输出目录可写")
        except Exception as e:
            print_error(f"输出目录不可写: {str(e)}")
            return False
        
        return True
    except Exception as e:
        print_error(f"配置加载失败: {str(e)}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "="*60)
    print("7. 测试基本功能")
    print("="*60)
    
    # 测试OpenCV
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        # 创建一个测试图像
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        print_success("OpenCV基本功能正常")
    except Exception as e:
        print_error(f"OpenCV测试失败: {str(e)}")
        return False
    
    # 测试PIL
    try:
        from PIL import Image  # type: ignore
        test_img = Image.new('RGB', (100, 100), color='red')
        print_success("Pillow基本功能正常")
    except Exception as e:
        print_error(f"Pillow测试失败: {str(e)}")
        return False
    
    # 测试NumPy
    try:
        import numpy as np  # type: ignore
        arr = np.array([1, 2, 3])
        print_success("NumPy基本功能正常")
    except Exception as e:
        print_error(f"NumPy测试失败: {str(e)}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("环境测试脚本 - 车牌识别项目")
    print("="*60)
    
    results = {}
    
    # 运行所有测试
    results = {
        'python_version': test_python_version(),
        'core_packages': test_core_packages(),
        'tesseract': test_tesseract_installation(),
        'project_structure': test_project_structure()
    }
    results['backend_config'] = test_backend_config()
    results['basic_functionality'] = test_basic_functionality()

    # 打印测试结果摘要
    print("\n" + "="*70)
    print("环境测试结果摘要")
    print("="*70)

    all_passed = True
    required_tests = ['python_version', 'core_packages', 'tesseract', 
                     'project_structure', 'backend_config', 'basic_functionality']
    optional_tests = []
    
    all_passed = True
    for test_name in required_tests:
        if results.get(test_name):
            print_success(f"{test_name}: 通过")
        else:
            print_error(f"{test_name}: 失败")
            all_passed = False
    
    for test_name in optional_tests:
        result = results.get(test_name)
        if result is True:
            print_success(f"{test_name}: 已安装（可选）")
        elif result is None:
            print_warning(f"{test_name}: 未安装（可选）")
    
    print("\n" + "="*60)
    if all_passed:
        print_success("所有必需测试通过！环境配置正确。")
        return 0
    else:
        print_error("部分测试失败，请检查上述错误信息并修复。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)