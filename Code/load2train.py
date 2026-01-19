import sys
import os
# 修正：脚本在Code/Cao/，需要上溯两级（.. / ..）才能到项目根目录Machine_Learning
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 获取正确的项目根目录（上溯两级：Cao/.. → Code，再 Code/.. → Machine_Learning）
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(project_root)  # 切换到真正的项目根目录
    
    # 2. 拼接正确的模型路径（项目根目录 → pt → baseline.pt）
    model_path = os.path.join(project_root, "pt", "baseline.pt")
    
    # 可选：路径验证（关键！打印路径并判断文件是否存在，方便排查）
    print("当前拼接的模型路径：", model_path)
    print("模型文件是否存在：", os.path.exists(model_path))  # 必须返回True才对
    
    # 3. 加载训练好的模型（此时路径已正确）
    model = YOLO(model_path)

    # 4. 在 test 集上评估
    metrics = model.val(
        data="Data/Raw/dust/dataset.yaml",  # 基于项目根目录的相对路径，正确
        split="test",
        imgsz=640,
        batch=16,
        device=0
    )

    # 5. 查看核心指标
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)
    print("Precision:", metrics.box.p)
    print("Recall:", metrics.box.r)