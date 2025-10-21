@echo off
chcp 65001
echo ====================================
echo GitHub 推送脚本
echo ====================================
echo.

echo 正在检查Git状态...
git status
echo.

echo 是否继续推送到GitHub? (Y/N)
set /p confirm=
if /i not "%confirm%"=="Y" (
    echo 操作已取消
    pause
    exit
)

echo.
echo 正在推送到GitHub...
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ====================================
    echo 推送成功！
    echo 访问: https://github.com/selenium603/writing-style-imitator
    echo ====================================
) else (
    echo.
    echo ====================================
    echo 推送失败！
    echo.
    echo 可能的原因:
    echo 1. 网络连接问题 - 需要VPN或代理
    echo 2. 需要身份验证 - 配置Personal Access Token或SSH密钥
    echo 3. 远程仓库不存在 - 确认仓库已创建
    echo.
    echo 请查看 "GitHub推送指南.md" 了解详细解决方案
    echo ====================================
)

echo.
pause

