@echo off
chcp 65001
echo ====================================
echo Git 代理配置脚本
echo ====================================
echo.
echo 请选择操作:
echo 1. 设置HTTP/HTTPS代理 (适用于Clash、V2Ray等)
echo 2. 切换到SSH方式（推荐）
echo 3. 清除代理设置
echo 4. 查看当前配置
echo 5. 退出
echo.

set /p choice=请输入选项 (1-5): 

if "%choice%"=="1" goto set_proxy
if "%choice%"=="2" goto set_ssh
if "%choice%"=="3" goto clear_proxy
if "%choice%"=="4" goto show_config
if "%choice%"=="5" goto end

:set_proxy
echo.
echo 请输入代理端口号 (常见: 7890, 10809, 1080)
set /p port=端口号: 
echo.
echo 正在配置代理...
git config --global http.proxy http://127.0.0.1:%port%
git config --global https.proxy https://127.0.0.1:%port%
echo ✓ 代理已设置为: http://127.0.0.1:%port%
echo.
echo 现在可以运行 push_to_github.bat 推送代码
goto end

:set_ssh
echo.
echo 切换到SSH方式需要先配置SSH密钥
echo.
echo 步骤:
echo 1. 生成SSH密钥: ssh-keygen -t ed25519 -C "your_email@example.com"
echo 2. 复制公钥: type %USERPROFILE%\.ssh\id_ed25519.pub
echo 3. 在GitHub添加SSH密钥: https://github.com/settings/keys
echo.
set /p confirm=是否已完成SSH密钥配置? (Y/N): 
if /i not "%confirm%"=="Y" (
    echo 请先完成SSH密钥配置
    goto end
)
echo.
echo 正在切换到SSH...
git remote set-url origin git@github.com:selenium603/writing-style-imitator.git
echo ✓ 已切换到SSH方式
echo.
echo 现在可以运行: git push -u origin main
goto end

:clear_proxy
echo.
echo 正在清除代理设置...
git config --global --unset http.proxy
git config --global --unset https.proxy
echo ✓ 代理设置已清除
goto end

:show_config
echo.
echo 当前Git配置:
echo --------------------------------
echo HTTP代理:
git config --global http.proxy
echo HTTPS代理:
git config --global https.proxy
echo.
echo 远程仓库URL:
git remote -v
echo --------------------------------
goto end

:end
echo.
pause

