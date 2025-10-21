@echo off
echo ========================================
echo 文本风格迁移与模仿器 - Web服务
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.7或更高版本
    pause
    exit /b 1
)

REM 运行启动脚本
echo 正在启动Web服务...
python run_web.py

pause
