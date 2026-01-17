本地生成融合数据后上传，在kaggle上拉取项目，并运行train.py即可  
1.pt文件夹：保存模型文件  
2.Code中代码:  
    synthetic_data.py：生成合成数据  
    merge_data.py：将合成数据与原数据按一定比例合并  
    train.py：训练模型并在测试集上测试    
    pred.py: 加载训练好的模型进行预测  
    proprocess_image.py: 数据预处理  
    dino_yolo.py：配置文件，包含模型配置、训练参数等  
3.Data文件夹：保存数据集，noise11表示有噪声，train中原始数据与合成数据1:1  
    Raw文件夹：保存原始数据集  
    Synthetic文件夹：保存合成数据集  
    Merged文件夹：保存合并后的数据集  
4.YAML文件夹：保存模型配置文件  
5.pt文件夹：保存模型文件  
6.README.md：说明文件  


