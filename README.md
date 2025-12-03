# 电动自行车车牌识别系统

一个基于FastAPI和MATLAB的智能车牌识别系统，支持图像预处理、车牌定位和OCR文字识别功能。

## 📋 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [API文档](#api文档)
- [使用说明](#使用说明)
- [项目结构](#项目结构)
- [开发说明](#开发说明)
- [常见问题](#常见问题)
- [许可证](#许可证)

## 项目简介

本项目是一个完整的电动自行车车牌识别系统，采用前后端分离架构，提供Web界面和RESTful API接口。系统支持多种图像处理算法，能够处理正常、倾斜、干扰等多种场景下的车牌图像，并通过OCR技术提取车牌号码。

### 主要应用场景

- 电动自行车管理
- 停车场管理系统
- 交通监控系统
- 车牌识别研究

## 功能特性

### 🖼️ 图像处理
- **多场景支持**：支持正常、倾斜、干扰等多种车牌图像
- **图像预处理**：自动进行图像增强、去噪、二值化等处理
- **MATLAB集成**：可选使用MATLAB进行高级图像处理算法

### 🔍 OCR识别
- **高精度识别**：基于Tesseract-OCR引擎
- **单行文本优化**：针对车牌单行文本进行优化配置
- **多语言支持**：支持中英文车牌识别

### 🌐 Web界面
- **简洁易用**：直观的Web界面，支持拖拽上传
- **实时预览**：实时显示原始图像和处理结果
- **结果展示**：清晰展示识别结果

### 🔌 API接口
- **RESTful API**：标准化的API接口设计
- **文件上传**：支持多种图像格式上传
- **MATLAB桥接**：可选的MATLAB引擎控制接口

## 技术栈

### 后端
- **FastAPI**：现代、快速的Web框架
- **Uvicorn**：ASGI服务器
- **OpenCV**：图像处理库
- **Pillow (PIL)**：图像处理库
- **NumPy**：数值计算库
- **Pytesseract**：OCR识别库
- **MATLAB Engine**（可选）：MATLAB集成

### 前端
- **HTML5/CSS3**：现代Web标准
- **JavaScript**：原生JavaScript实现

### 工具
- **Tesseract-OCR**：OCR识别引擎
- **MATLAB**（可选）：高级图像处理

## 系统要求

- **Python**: 3.8 或更高版本
- **操作系统**: Windows 10/11、Linux、macOS
- **Tesseract-OCR**: 用于OCR识别（必需）
- **MATLAB**: R2019b 或更高版本（可选，用于高级图像处理）
- **包管理**: Conda 或 pip

> 📖 **详细安装说明请查看 [INSTALL.md](INSTALL.md)**

## 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd license_plate_recognition
```

### 2. 安装依赖

**使用Conda（推荐）：**
```bash
conda env create -f environment.yml
conda activate plate_recog
```

**或使用pip：**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python test_env.py
```

### 4. 启动服务

```bash
python -m backend.main
# 或
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. 访问系统

- **Web界面**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **交互式API**: http://localhost:8000/redoc

> 💡 **提示**：首次安装请查看 [INSTALL.md](INSTALL.md) 获取详细的分步安装指南。

## API文档

### 核心接口

#### 1. 图像处理接口

**POST** `/api/process-image`

上传图像并进行车牌识别处理。

**请求**:
- Content-Type: `multipart/form-data`
- Body: 图像文件（支持常见图像格式）

**响应**:
```json
{
  "result": "识别结果",
  "uploaded": "文件路径"
}
```

#### 2. MATLAB引擎状态

**GET** `/api/matlab/status`

查询MATLAB引擎安装和运行状态。

**响应**:
```json
{
  "engine_installed": true,
  "engine_running": false
}
```

#### 3. MATLAB引擎控制

- **POST** `/api/matlab/start` - 启动MATLAB引擎
- **POST** `/api/matlab/stop` - 停止MATLAB引擎
- **POST** `/api/matlab/call` - 调用MATLAB函数

### 完整API文档

启动服务后，访问 http://localhost:8000/docs 查看完整的交互式API文档。

## 使用说明

### Web界面使用

1. **启动后端服务**
   ```bash
   python -m backend.main
   ```

2. **打开浏览器**
   访问 http://localhost:8000

3. **上传图像**
   - 点击"选择图片"按钮
   - 选择要识别的车牌图像
   - 系统自动处理并显示结果

4. **查看结果**
   - 原始图片：显示上传的原始图像
   - 处理结果：显示处理后的图像
   - 识别结果：显示识别出的车牌号码

### API调用示例

**使用curl：**
```bash
curl -X POST "http://localhost:8000/api/process-image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

**使用Python：**
```python
import requests

url = "http://localhost:8000/api/process-image"
files = {"file": open("path/to/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## 项目结构

```
license_plate_recognition/
├── backend/                  # 后端代码
│   ├── __init__.py
│   ├── main.py              # FastAPI主应用
│   ├── config.py            # 配置文件（路径、OCR配置等）
│   └── matlab_client.py     # MATLAB客户端封装
├── frontend/                 # 前端代码
│   ├── index.html           # 主页面
│   ├── css/
│   │   └── style.css        # 样式文件
│   └── js/                  # JavaScript文件（待实现）
├── matlab_core/             # MATLAB核心代码
│   └── process_image.m     # 图像处理函数
├── data/                    # 数据目录
│   ├── input_images/        # 输入图像
│   │   ├── normal_plate/    # 正常车牌
│   │   ├── tilted_plate/    # 倾斜车牌
│   │   ├── interfered_plate/# 干扰车牌
│   │   └── optional_plate/  # 可选测试图像
│   └── output_results/      # 输出结果
│       └── processed_images/# 处理后的图像
├── docs/                    # 文档目录
├── test_env.py              # 环境测试脚本
├── activate_env.bat         # Windows环境激活脚本
├── activate_env.ps1         # PowerShell环境激活脚本
├── requirements.txt         # pip依赖列表
├── environment.yml          # conda环境配置
├── INSTALL.md              # 详细安装指南
└── README.md               # 项目说明（本文件）
```

## 开发说明

### 技术架构

- **后端架构**: FastAPI + Uvicorn
- **图像处理**: OpenCV + Pillow + NumPy
- **OCR识别**: Tesseract-OCR + Pytesseract
- **可选集成**: MATLAB Engine（用于高级图像处理算法）

### 配置说明

主要配置文件：`backend/config.py`

- **Tesseract路径**: 自动检测，支持跨平台
- **OCR配置**: `--psm 7 --oem 3`（单行文本优化）
- **输入输出路径**: 自动创建，相对路径配置

### 扩展开发

1. **添加新的图像处理算法**
   - 在 `matlab_core/process_image.m` 中实现
   - 或直接在Python中使用OpenCV实现

2. **自定义OCR配置**
   - 修改 `backend/config.py` 中的 `OCR_CONFIG`
   - 参考Tesseract文档调整PSM和OEM参数

3. **添加新的API端点**
   - 在 `backend/main.py` 中添加新的路由
   - 参考FastAPI文档

### 环境测试

运行 `test_env.py` 可以检查：
- ✅ Python版本
- ✅ 核心Python包安装情况
- ✅ Tesseract-OCR配置
- ✅ MATLAB Engine（可选）
- ✅ 项目目录结构
- ✅ 后端配置
- ✅ 基本功能测试

## 常见问题

### 1. Tesseract找不到

**问题**: 系统提示找不到Tesseract-OCR

**解决方案**:
- 确保已安装Tesseract-OCR
- 检查 `backend/config.py` 中的路径配置
- 将Tesseract添加到系统PATH环境变量
- 运行 `python test_env.py` 进行诊断

### 2. MATLAB Engine导入失败

**问题**: 导入matlab.engine失败

**解决方案**:
- MATLAB Engine是可选的，不影响基本功能
- 如需使用，请参考 [INSTALL.md](INSTALL.md) 安装MATLAB Engine
- 确保在正确的Python环境中安装

### 3. 依赖安装失败

**问题**: pip或conda安装依赖时出错

**解决方案**:
- 确保Python版本 >= 3.8
- 升级pip：`python -m pip install --upgrade pip`
- 使用conda环境可以避免依赖冲突
- 使用国内镜像源（如清华源）

### 4. 图像处理失败

**问题**: 上传图像后处理失败

**解决方案**:
- 检查图像格式是否支持（JPG、PNG等）
- 查看后端日志获取详细错误信息
- 确保MATLAB Engine已正确安装（如果使用MATLAB处理）

### 5. 识别准确率低

**问题**: OCR识别结果不准确

**解决方案**:
- 确保图像质量良好（清晰、对比度高）
- 调整OCR配置参数（`backend/config.py`）
- 对图像进行预处理（去噪、增强等）
- 考虑使用更高质量的图像

> 📖 **更多问题请查看 [INSTALL.md](INSTALL.md) 的故障排除章节**

## 许可证

详见 [LICENSE](LICENSE) 文件

## 贡献

欢迎提交Issue和Pull Request！

---

**项目维护者**: [您的名字]  
**最后更新**: 2024年
