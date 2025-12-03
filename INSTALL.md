# 安装指南

本文档提供详细的分步安装说明，适用于Windows、Linux和macOS操作系统。

> 📖 **提示**：如果您想了解项目的功能特性、技术栈和使用方法，请查看 [README.md](README.md)

## 📋 目录

- [安装前准备](#安装前准备)
- [Windows安装指南](#windows安装指南)
- [Linux安装指南](#linux安装指南)
- [macOS安装指南](#macos安装指南)
- [安装MATLAB Engine（可选）](#安装matlab-engine可选)
- [配置说明](#配置说明)
- [验证安装](#验证安装)
- [故障排除](#故障排除)
- [下一步](#下一步)

## 安装前准备

### 系统要求

在开始安装之前，请确保您的系统满足以下要求：

- **Python**: 3.8 或更高版本（推荐 3.8-3.11）
- **操作系统**: 
  - Windows 10/11
  - Linux (Ubuntu 18.04+, CentOS 7+, 或其他主流发行版)
  - macOS 10.14+
- **内存**: 至少 4GB RAM（推荐 8GB+）
- **磁盘空间**: 至少 2GB 可用空间

### 必需软件

1. **Python 3.8+**
2. **Tesseract-OCR**（用于OCR识别）
3. **Git**（用于克隆项目）

### 可选软件

1. **Conda/Anaconda**（推荐，用于环境管理）
2. **MATLAB R2019b+**（可选，用于高级图像处理）

## Windows安装指南

### 步骤1：安装Python

1. **下载Python**
   - 访问 https://www.python.org/downloads/
   - 下载Python 3.8或更高版本（推荐3.10或3.11）

2. **安装Python**
   - 运行下载的安装程序
   - ⚠️ **重要**：勾选"Add Python to PATH"选项
   - 选择"Install Now"或"Customize installation"
   - 如果选择自定义安装，确保勾选"pip"和"Add Python to environment variables"

3. **验证安装**
   ```bash
   python --version
   pip --version
   ```
   应该显示类似 `Python 3.10.x` 和 `pip 23.x.x` 的输出

### 步骤2：安装Conda（推荐）

Conda可以更好地管理Python环境和依赖，避免版本冲突。

#### 选项A：安装Anaconda（完整版）

1. **下载Anaconda**
   - 访问 https://www.anaconda.com/products/distribution
   - 下载Windows 64位安装程序

2. **安装Anaconda**
   - 运行安装程序
   - 选择"Just Me"（推荐）或"All Users"
   - 选择安装路径（默认即可）
   - ⚠️ **重要**：勾选"Add Anaconda to PATH"选项
   - 完成安装

#### 选项B：安装Miniconda（轻量版）

1. **下载Miniconda**
   - 访问 https://docs.conda.io/en/latest/miniconda.html
   - 下载Windows 64位安装程序

2. **安装Miniconda**
   - 运行安装程序
   - 按照提示完成安装
   - 勾选"Add Miniconda to PATH"

3. **验证安装**
   ```bash
   conda --version
   ```
   应该显示类似 `conda 23.x.x` 的输出

### 步骤3：安装Tesseract-OCR

1. **下载Tesseract**
   - 访问 https://github.com/UB-Mannheim/tesseract/wiki
   - 下载最新版本的Windows安装程序（.exe文件）
   - 推荐下载包含中文语言包的版本

2. **安装Tesseract**
   - 运行安装程序
   - 选择安装路径（推荐默认：`C:\Program Files\Tesseract-OCR`）
   - ⚠️ **重要**：在"Additional language data"中选择需要的语言包（至少选择English和Chinese）
   - 完成安装

3. **添加到PATH（如果未自动添加）**
   - 右键"此电脑" → "属性" → "高级系统设置" → "环境变量"
   - 在"系统变量"中找到"Path"，点击"编辑"
   - 添加Tesseract安装路径：`C:\Program Files\Tesseract-OCR`
   - 点击"确定"保存

4. **验证安装**
   ```bash
   tesseract --version
   ```
   应该显示Tesseract版本信息

### 步骤4：安装Git（如果未安装）

1. **下载Git**
   - 访问 https://git-scm.com/download/win
   - 下载Windows安装程序

2. **安装Git**
   - 运行安装程序
   - 使用默认设置即可
   - 完成安装

3. **验证安装**
   ```bash
   git --version
   ```

### 步骤5：克隆项目

1. **打开命令提示符或PowerShell**
   - 按 `Win + R`，输入 `cmd` 或 `powershell`

2. **克隆项目**
   ```bash
   git clone <repository-url>
   cd license_plate_recognition
   ```
   将 `<repository-url>` 替换为实际的仓库地址

### 步骤6：创建Python环境

#### 方法A：使用Conda（推荐）

1. **创建Conda环境**
   ```bash
   conda env create -f environment.yml
   ```
   这会自动安装所有必需的Python包，可能需要几分钟时间

2. **激活环境**
   ```bash
   conda activate plate_recog
   ```

3. **验证环境**
   ```bash
   conda info --envs
   ```
   应该看到 `plate_recog` 环境前面有 `*` 标记

#### 方法B：使用pip和虚拟环境

1. **创建虚拟环境**
   ```bash
   python -m venv venv
   ```

2. **激活虚拟环境**
   ```bash
   venv\Scripts\activate
   ```
   激活后，命令提示符前面会显示 `(venv)`

3. **升级pip**
   ```bash
   python -m pip install --upgrade pip
   ```

4. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

### 步骤7：运行环境测试

#### 使用自动脚本（推荐）

```bash
# 批处理脚本（自动激活环境并测试）
activate_env.bat

# 或使用PowerShell脚本
.\activate_env.ps1
```

#### 手动运行测试

```bash
# 确保环境已激活
conda activate plate_recog  # 或 venv\Scripts\activate

# 运行测试脚本
python test_env.py
```

如果所有测试通过（显示 ✓），说明安装成功！

## Linux安装指南

### 步骤1：安装Python和pip

#### Ubuntu/Debian系统

```bash
# 更新包列表
sudo apt-get update

# 安装Python 3和pip
sudo apt-get install python3 python3-pip python3-venv

# 验证安装
python3 --version
pip3 --version
```

#### CentOS/RHEL系统

```bash
# 安装Python 3和pip
sudo yum install python3 python3-pip

# 或使用dnf（较新版本）
sudo dnf install python3 python3-pip

# 验证安装
python3 --version
pip3 --version
```

#### Arch Linux

```bash
sudo pacman -S python python-pip
```

### 步骤2：安装Conda（可选但推荐）

1. **下载Miniconda**
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```

2. **安装Miniconda**
   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh
   ```
   - 按Enter阅读许可协议
   - 输入 `yes` 同意许可
   - 选择安装路径（默认即可）
   - 输入 `yes` 初始化conda

3. **重新加载shell配置**
   ```bash
   source ~/.bashrc
   # 或
   source ~/.zshrc
   ```

4. **验证安装**
   ```bash
   conda --version
   ```

### 步骤3：安装Tesseract-OCR

#### Ubuntu/Debian系统

```bash
# 更新包列表
sudo apt-get update

# 安装Tesseract-OCR
sudo apt-get install tesseract-ocr

# 安装中文语言包（可选）
sudo apt-get install tesseract-ocr-chi-sim tesseract-ocr-chi-tra

# 验证安装
tesseract --version
```

#### CentOS/RHEL系统

```bash
# 安装EPEL仓库（如果未安装）
sudo yum install epel-release

# 安装Tesseract-OCR
sudo yum install tesseract

# 验证安装
tesseract --version
```

#### Arch Linux

```bash
sudo pacman -S tesseract tesseract-data-chi-sim
```

### 步骤4：安装Git（如果未安装）

#### Ubuntu/Debian

```bash
sudo apt-get install git
```

#### CentOS/RHEL

```bash
sudo yum install git
```

### 步骤5：克隆项目

```bash
git clone <repository-url>
cd license_plate_recognition
```

### 步骤6：创建Python环境

#### 方法A：使用Conda（推荐）

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate plate_recog
```

#### 方法B：使用pip和虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级pip
python3 -m pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### 步骤7：运行环境测试

```bash
# 确保环境已激活
conda activate plate_recog  # 或 source venv/bin/activate

# 运行测试
python test_env.py
```

## macOS安装指南

### 步骤1：安装Homebrew（如果未安装）

Homebrew是macOS的包管理器，推荐使用。

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

安装完成后，按照提示将Homebrew添加到PATH。

### 步骤2：安装Python

```bash
# 使用Homebrew安装Python
brew install python3

# 验证安装
python3 --version
pip3 --version
```

### 步骤3：安装Conda（可选但推荐）

1. **下载Miniconda**
   ```bash
   curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
   ```
   
   对于Apple Silicon (M1/M2) Mac：
   ```bash
   curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
   ```

2. **安装Miniconda**
   ```bash
   bash Miniconda3-latest-MacOSX-x86_64.sh
   # 或
   bash Miniconda3-latest-MacOSX-arm64.sh
   ```

3. **重新加载shell配置**
   ```bash
   source ~/.zshrc  # 或 ~/.bash_profile
   ```

4. **验证安装**
   ```bash
   conda --version
   ```

### 步骤4：安装Tesseract-OCR

```bash
# 使用Homebrew安装
brew install tesseract

# 安装中文语言包（可选）
brew install tesseract-lang

# 验证安装
tesseract --version
```

### 步骤5：安装Git（如果未安装）

```bash
brew install git
```

### 步骤6：克隆项目

```bash
git clone <repository-url>
cd license_plate_recognition
```

### 步骤7：创建Python环境

#### 方法A：使用Conda（推荐）

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate plate_recog
```

#### 方法B：使用pip和虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级pip
python3 -m pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### 步骤8：运行环境测试

```bash
# 确保环境已激活
conda activate plate_recog  # 或 source venv/bin/activate

# 运行测试
python test_env.py
```

## 安装MATLAB Engine（可选）

MATLAB Engine用于在Python中调用MATLAB函数，这是**可选的**。如果不安装，项目的基本功能仍然可以正常使用。

### 前置要求

- 已安装MATLAB R2019b或更高版本
- 已创建并激活项目的Python环境

### Windows安装步骤

1. **打开命令提示符或PowerShell**
   - 确保已激活项目的Python环境

2. **切换到MATLAB Python引擎目录**
   ```bash
   cd "C:\Program Files\MATLAB\R2023a\extern\engines\python"
   ```
   ⚠️ **注意**：将 `R2023a` 替换为您的MATLAB版本号（如R2022a、R2023b等）

3. **安装MATLAB Engine**
   ```bash
   python setup.py install
   ```

4. **验证安装**
   ```bash
   python -c "import matlab.engine; print('MATLAB Engine安装成功')"
   ```

### Linux安装步骤

1. **切换到MATLAB Python引擎目录**
   ```bash
   cd /usr/local/MATLAB/R2023a/extern/engines/python
   ```
   或
   ```bash
   cd /opt/MATLAB/R2023a/extern/engines/python
   ```
   ⚠️ **注意**：路径可能因安装位置而异

2. **安装MATLAB Engine**
   ```bash
   python3 setup.py install
   ```

3. **验证安装**
   ```bash
   python3 -c "import matlab.engine; print('MATLAB Engine安装成功')"
   ```

### macOS安装步骤

1. **切换到MATLAB Python引擎目录**
   ```bash
   cd "/Applications/MATLAB_R2023a.app/extern/engines/python"
   ```
   ⚠️ **注意**：将 `R2023a` 替换为您的MATLAB版本号

2. **安装MATLAB Engine**
   ```bash
   python3 setup.py install
   ```

3. **验证安装**
   ```bash
   python3 -c "import matlab.engine; print('MATLAB Engine安装成功')"
   ```

### 常见问题

**问题：找不到MATLAB安装目录**

- Windows: 通常在 `C:\Program Files\MATLAB\`
- Linux: 通常在 `/usr/local/MATLAB/` 或 `/opt/MATLAB/`
- macOS: 通常在 `/Applications/MATLAB_R20XX.app/`

**问题：安装时提示权限错误**

- Windows: 以管理员身份运行命令提示符
- Linux/macOS: 使用 `sudo`（不推荐）或确保有写入权限

**问题：安装后仍无法导入**

- 确保在正确的Python环境中安装
- 重新激活环境后重试
- 运行 `python test_env.py` 检查MATLAB Engine状态

## 配置说明

### Tesseract路径配置

项目会自动检测Tesseract的安装路径。如果自动检测失败，可以手动配置：

1. **编辑配置文件**
   - 打开 `backend/config.py`

2. **修改TESSERACT_CMD**
   ```python
   # Windows示例
   TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   
   # Linux示例
   TESSERACT_CMD = '/usr/bin/tesseract'
   
   # macOS示例
   TESSERACT_CMD = '/usr/local/bin/tesseract'
   ```

3. **保存文件并重新运行测试**
   ```bash
   python test_env.py
   ```

### OCR配置

OCR参数可以在 `backend/config.py` 中修改：

```python
OCR_CONFIG = r'--psm 7 --oem 3'
```

- `--psm 7`: 假设图像是单个文本行
- `--oem 3`: 使用默认OCR引擎模式

更多配置选项请参考Tesseract文档。

### 输入输出路径配置

默认情况下，项目使用以下路径：

- **输入图像**: `data/input_images/`
- **输出结果**: `data/output_results/processed_images/`

这些路径在 `backend/config.py` 中自动配置，通常不需要修改。

## 验证安装

### 运行环境测试

安装完成后，运行环境测试脚本进行全面检查：

```bash
# 确保环境已激活
conda activate plate_recog  # 或 source venv/bin/activate

# 运行测试
python test_env.py
```

### 预期输出

如果安装成功，您应该看到类似以下输出：

```
============================================================
环境测试脚本 - 车牌识别项目
============================================================

============================================================
1. 检查Python版本
============================================================
  Python版本: 3.10.9
✓ Python版本符合要求 (>= 3.8)

============================================================
2. 检查核心Python包
============================================================
✓ FastAPI: 0.104.1
✓ Uvicorn: 0.24.0
✓ Pytesseract: 0.3.10
✓ OpenCV (opencv-python): 4.8.1
✓ NumPy: 1.24.3
✓ Pillow: 10.1.0
✓ Aiofiles: 23.2.1

============================================================
3. 检查Tesseract-OCR
============================================================
✓ Tesseract-OCR版本: 5.3.0
  Tesseract-OCR路径: C:\Program Files\Tesseract-OCR\tesseract.exe
✓ Tesseract-OCR可执行文件存在

============================================================
4. 检查MATLAB Engine（可选）
============================================================
⚠ MATLAB Engine未安装（这是可选的）

============================================================
5. 检查项目目录结构
============================================================
✓ 目录存在: backend
✓ 目录存在: data/input_images
✓ 目录存在: data/output_results
✓ 目录存在: matlab_core
✓ 目录存在: frontend
✓ 文件存在: backend/main.py
✓ 文件存在: backend/config.py
✓ 文件存在: backend/matlab_client.py
✓ 文件存在: matlab_core/process_image.m

============================================================
6. 检查后端配置
============================================================
✓ 配置文件加载成功
  输入图像目录: D:\...\data\input_images
  输出结果目录: D:\...\data\output_results
  处理后图像目录: D:\...\data\output_results\processed_images
  OCR配置: --psm 7 --oem 3
✓ 输出目录可写

============================================================
7. 测试基本功能
============================================================
✓ OpenCV基本功能正常
✓ Pillow基本功能正常
✓ NumPy基本功能正常

============================================================
测试结果汇总
============================================================
✓ python_version: 通过
✓ core_packages: 通过
✓ tesseract: 通过
⚠ matlab: 未安装（可选）
✓ project_structure: 通过
✓ backend_config: 通过
✓ basic_functionality: 通过

============================================================
✓ 所有必需测试通过！环境配置正确。
============================================================
```

### 快速验证命令

```bash
# 检查Python版本
python --version

# 检查关键包
python -c "import fastapi, cv2, numpy, PIL; print('核心包正常')"

# 检查Tesseract
tesseract --version

# 检查项目配置
python -c "from backend.config import TESSERACT_CMD; print(f'Tesseract路径: {TESSERACT_CMD}')"
```

## 故障排除

### 问题1：conda命令未找到

**症状**: 运行 `conda` 命令时提示"命令未找到"

**解决方案**:
1. **检查安装**
   - 确认Conda已正确安装
   - 重新打开终端窗口

2. **检查PATH**
   - Windows: 检查环境变量中是否包含Conda路径
   - Linux/macOS: 检查 `~/.bashrc` 或 `~/.zshrc` 中是否有conda初始化代码

3. **手动初始化**
   ```bash
   # Linux/macOS
   source ~/miniconda3/etc/profile.d/conda.sh
   # 或
   source ~/anaconda3/etc/profile.d/conda.sh
   ```

### 问题2：pip安装失败

**症状**: 运行 `pip install -r requirements.txt` 时出错

**解决方案**:

1. **升级pip**
   ```bash
   python -m pip install --upgrade pip
   ```

2. **使用国内镜像源（如果网络较慢）**
   ```bash
   # 清华镜像
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   
   # 阿里云镜像
   pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
   ```

3. **逐个安装包（定位问题包）**
   ```bash
   pip install fastapi
   pip install uvicorn[standard]
   # ... 逐个安装
   ```

4. **使用conda安装（更稳定）**
   ```bash
   conda install -c conda-forge fastapi uvicorn opencv numpy pillow
   ```

### 问题3：Tesseract未找到

**症状**: 测试脚本提示找不到Tesseract-OCR

**解决方案**:

1. **验证Tesseract安装**
   ```bash
   tesseract --version
   ```
   如果命令不存在，说明Tesseract未正确安装或未添加到PATH

2. **Windows手动配置**
   - 编辑 `backend/config.py`
   - 修改 `TESSERACT_CMD` 为实际安装路径
   - 或添加Tesseract到系统PATH环境变量

3. **Linux/macOS检查**
   ```bash
   # 查找Tesseract位置
   which tesseract
   # 或
   find /usr -name tesseract 2>/dev/null
   ```

4. **重新安装Tesseract**
   - 确保安装时选择了"添加到PATH"选项
   - 重新安装后重启终端

### 问题4：OpenCV安装失败

**症状**: 安装opencv-python时出错或导入失败

**解决方案**:

1. **使用conda安装（推荐）**
   ```bash
   conda install opencv -c conda-forge
   ```

2. **使用headless版本**
   ```bash
   pip install opencv-python-headless
   ```

3. **Linux系统依赖**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libopencv-dev python3-opencv
   ```

4. **检查系统架构**
   - 确保Python和OpenCV的架构匹配（32位/64位）

### 问题5：权限错误（Linux/macOS）

**症状**: 安装时提示权限被拒绝

**解决方案**:

1. **使用虚拟环境（推荐）**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **检查目录权限**
   ```bash
   ls -la venv/
   ```

3. **避免使用sudo**
   - 不要使用 `sudo pip install`，这会导致权限问题

### 问题6：环境创建失败

**症状**: `conda env create -f environment.yml` 失败

**解决方案**:

1. **更新conda**
   ```bash
   conda update conda
   ```

2. **检查YAML文件**
   ```bash
   cat environment.yml
   ```
   确保文件格式正确

3. **手动创建环境**
   ```bash
   conda create -n plate_recog python=3.8
   conda activate plate_recog
   pip install -r requirements.txt
   ```

4. **清理缓存**
   ```bash
   conda clean --all
   ```

### 问题7：MATLAB Engine安装失败

**症状**: 无法导入matlab.engine

**解决方案**:

1. **检查MATLAB版本**
   - 确保MATLAB版本 >= R2019b

2. **检查Python版本匹配**
   - MATLAB Engine需要与Python版本匹配
   - 查看MATLAB文档了解支持的Python版本

3. **在正确的环境中安装**
   ```bash
   # 确保环境已激活
   conda activate plate_recog
   # 然后安装
   cd "C:\Program Files\MATLAB\R2023a\extern\engines\python"
   python setup.py install
   ```

4. **检查MATLAB许可证**
   - 确保MATLAB许可证有效

### 问题8：依赖版本冲突

**症状**: 安装时提示版本冲突

**解决方案**:

1. **使用conda环境（推荐）**
   - Conda可以更好地解决依赖冲突

2. **检查requirements.txt**
   - 确保版本要求合理
   - 可以尝试放宽版本限制（如 `>=` 改为 `~=`）

3. **创建全新环境**
   ```bash
   conda create -n plate_recog_new python=3.10
   conda activate plate_recog_new
   pip install -r requirements.txt
   ```

### 问题9：网络连接问题

**症状**: 下载包时超时或连接失败

**解决方案**:

1. **使用国内镜像源**
   ```bash
   # pip镜像
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   
   # conda镜像
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
   conda config --set show_channel_urls yes
   ```

2. **配置代理（如果有）**
   ```bash
   pip install --proxy http://proxy.example.com:8080 -r requirements.txt
   ```

3. **增加超时时间**
   ```bash
   pip install --default-timeout=100 -r requirements.txt
   ```

### 问题10：测试脚本运行失败

**症状**: `python test_env.py` 报错

**解决方案**:

1. **检查环境是否激活**
   ```bash
   # 应该看到环境名称
   conda activate plate_recog
   # 或
   source venv/bin/activate
   ```

2. **检查Python路径**
   ```bash
   which python
   # 应该指向环境中的Python
   ```

3. **检查项目结构**
   ```bash
   ls -la
   # 确保在项目根目录
   ```

4. **查看详细错误信息**
   ```bash
   python test_env.py 2>&1 | tee test_output.log
   ```

## 下一步

安装和验证完成后，您可以：

1. **查看项目文档**
   - 阅读 [README.md](README.md) 了解项目功能和使用方法

2. **启动项目**
   ```bash
   # 确保环境已激活
   conda activate plate_recog  # 或 source venv/bin/activate
   
   # 启动后端服务
   python -m backend.main
   ```

3. **访问Web界面**
   - 打开浏览器访问 http://localhost:8000
   - 查看API文档：http://localhost:8000/docs

4. **开始使用**
   - 上传测试图像进行车牌识别
   - 查看识别结果

---

**需要帮助？**

如果遇到其他问题，请：
- 查看 [README.md](README.md) 的常见问题章节
- 检查项目的Issue列表
- 提交新的Issue描述您的问题

**祝您使用愉快！** 🎉
