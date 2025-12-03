@echo off
REM 激活conda环境并运行环境测试脚本

echo ========================================
echo 车牌识别项目 - 环境激活脚本
echo ========================================
echo.

REM 检查conda是否可用
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 未找到conda命令，请确保已安装Anaconda或Miniconda
    echo 并且已将conda添加到系统PATH环境变量中
    pause
    exit /b 1
)

echo [信息] 正在激活conda环境: plate_recog
call conda activate plate_recog

if %ERRORLEVEL% NEQ 0 (
    echo [错误] 无法激活conda环境 plate_recog
    echo [提示] 如果环境不存在，请运行以下命令创建：
    echo        conda env create -f environment.yml
    pause
    exit /b 1
)

echo [成功] conda环境已激活
echo.

REM 运行环境测试
echo [信息] 正在运行环境测试...
echo.
python test_tesseract.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo [成功] 环境测试通过！
    echo ========================================
) else (
    echo.
    echo ========================================
    echo [警告] 环境测试发现问题，请检查上述输出
    echo ========================================
)

echo.
echo 环境已激活，您现在可以运行项目了
echo 按任意键退出...
pause >nul

