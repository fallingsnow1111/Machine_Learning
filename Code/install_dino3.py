"""
快速安装 DINO3 所需依赖
运行: python install_dino3.py
"""

import subprocess
import sys

def install_package(package):
    """安装 Python 包"""
    print(f"📦 安装 {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("="*60)
    print("🚀 DINO3-YOLO 依赖安装")
    print("="*60)
    
    # 必需的包
    packages = [
        "modelscope",           # DINO3 模型加载
        "transformers>=4.35.0", # Hugging Face 模型支持
    ]
    
    for pkg in packages:
        try:
            install_package(pkg)
            print(f"✅ {pkg} 安装成功\n")
        except Exception as e:
            print(f"❌ {pkg} 安装失败: {e}\n")
    
    print("="*60)
    print("✅ 安装完成！")
    print("="*60)
    print("\n下一步:")
    print("  python dino_yolo.py")
    print("\n⚠️  首次运行会自动下载 DINOv3-vitl16 模型 (约 1GB)")
    print("="*60)

if __name__ == "__main__":
    main()
