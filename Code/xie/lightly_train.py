import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="runs/distillation/dinov3_to_yolo11_640",
        data="Data/Processed/dust_640x640_enhanced",                    # 有标注的数据集
        model="ultralytics/yolo11s",
        task="object_detection",
        
        # 关键：设置蒸馏参数
        method="distillation",                      # 使用蒸馏方法
        method_args={
            "teacher": "dinov3/vits16",            # DINOv3 教师模型
        },
        
        # 类别定义
        classes={
            0: "dust",
        },
        
        # 训练参数
        epochs=300,
        batch_size=16,
        learning_rate=0.001,
    )