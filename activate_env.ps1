# PowerShell脚本 - 激活conda环境并运行环境测试

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "车牌识别项目 - 环境激活脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查conda是否可用
$condaPath = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaPath) {
    Write-Host "[错误] 未找到conda命令，请确保已安装Anaconda或Miniconda" -ForegroundColor Red
    Write-Host "并且已将conda添加到系统PATH环境变量中" -ForegroundColor Red
    Read-Host "按Enter键退出"
    exit 1
}

# 初始化conda（如果需要）
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    $condaInit = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
    if (Test-Path $condaInit) {
        & $condaInit init powershell
    }
}

Write-Host "[信息] 正在激活conda环境: plate_recog" -ForegroundColor Yellow
conda activate plate_recog

if ($LASTEXITCODE -ne 0) {
    Write-Host "[错误] 无法激活conda环境 plate_recog" -ForegroundColor Red
    Write-Host "[提示] 如果环境不存在，请运行以下命令创建：" -ForegroundColor Yellow
    Write-Host "      conda env create -f environment.yml" -ForegroundColor Yellow
    Read-Host "按Enter键退出"
    exit 1
}

Write-Host "[成功] conda环境已激活" -ForegroundColor Green
Write-Host ""

# 运行环境测试
Write-Host "[信息] 正在运行环境测试..." -ForegroundColor Yellow
Write-Host ""
python test_tesseract.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "[成功] 环境测试通过！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
}
else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "[警告] 环境测试发现问题，请检查上述输出" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "环境已激活，您现在可以运行项目了" -ForegroundColor Cyan
Read-Host "按Enter键退出"

