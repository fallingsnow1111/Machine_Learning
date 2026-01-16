"""
下载 DINOv3 模型到本地（用于 Kaggle 等离线环境）
运行: python download_dino3_model.py
"""

from modelscope import snapshot_download
from pathlib import Path
import os

def download_dino3_model():
    """下载 DINOv3-vitl16 模型到本地"""
    # 自动检测环境
    is_kaggle = os.path.exists('/kaggle')
    print("="*60)
    print("📥 下载 DINOv3-vitl16 模型")
    print("="*60)
    
    model_id = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
    
    # 根据环境选择保存路径
    if is_kaggle:
        local_dir = '/kaggle/working/models/dinov3-vitl16'
        print("\n🌐 检测到 Kaggle 环境")
    else:
        local_dir = './models/dinov3-vitl16'
        print("\n💻 本地环境")
    
    print(f"模型 ID: {model_id}")
    print(f"保存路径: {local_dir}")
    print(f"大小: 约 1GB\n")
    
    try:
        # 创建目录
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        # 下载模型
        print("开始下载...")
        cache_dir = snapshot_download(
            model_id,
            cache_dir=local_dir,
            revision='master'
        )
        
        print("\n" + "="*60)
        print("✅ 下载完成！")
        print("="*60)
        print(f"📁 模型保存在: {cache_dir}")
        
        if is_kaggle:
            print("\n📝 Kaggle 环境下一步:")
            print(f"  修改 YAML 文件中的模型路径为:")
            print(f"  '{local_dir}'")
            print("\n  或者运行后保存为 Kaggle Dataset 供下次使用")
        else:
            print("\n📝 本地环境下一步:")
            print("  1. 将 models/ 文件夹上传到 Kaggle Dataset")
            print("  2. 在 Kaggle 中修改 YAML 路径为:")
            print("     '/kaggle/input/<your-dataset-name>/dinov3-vitl16'")
            print("\n  或直接使用当前路径 './models/dinov3-vitl16' (适合本地训练)")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n备选方案:")
        print("  1. 在本地有网的机器上运行此脚本下载模型")
        print("  2. 手动从 ModelScope 下载:")
        print(f"     https://modelscope.cn/models/{model_id}")
        print("  3. 将下载的文件解压到 models/dinov3-vitl16/")

if __name__ == "__main__":
    download_dino3_model()
