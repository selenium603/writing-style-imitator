#!/usr/bin/env python
"""
文本风格迁移与模仿器 - Web服务启动脚本
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_requirements():
    """检查必要的依赖是否已安装"""
    print("检查依赖...")
    required_packages = ['flask', 'flask_cors', 'torch', 'transformers']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        return False
    
    print("✓ 所有依赖已安装")
    return True

def check_models():
    """检查是否有已训练的模型"""
    models_dir = Path("models")
    if not models_dir.exists():
        models_dir.mkdir(exist_ok=True)
        print("\n提示: 还没有训练任何模型")
        print("请先使用以下命令训练模型:")
        print("python train.py [作者名] --epochs 3")
        return False
    
    model_count = len([d for d in models_dir.iterdir() if d.is_dir() and (d / "final_model").exists()])
    if model_count == 0:
        print("\n提示: 还没有训练完成的模型")
        print("请先使用以下命令训练模型:")
        print("python train.py [作者名] --epochs 3")
        return False
    
    print(f"✓ 找到 {model_count} 个已训练的模型")
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("文本风格迁移与模仿器 - Web服务")
    print("=" * 60)
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("错误: 需要Python 3.7或更高版本")
        sys.exit(1)
    
    # 检查依赖
    if not check_requirements():
        sys.exit(1)
    
    # 检查模型（不强制要求）
    check_models()
    
    print("\n正在启动Web服务...")
    print("-" * 60)
    
    try:
        # 启动Flask应用
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n服务已停止")
    except subprocess.CalledProcessError as e:
        print(f"\n启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
