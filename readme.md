本地生成融合数据后上传，在kaggle上拉取项目，并运行train.py即可  
1.pt文件夹：保存模型文件  
2.Code中代码:  
    synthetic_data.py：生成合成数据  
    merge_data.py：将合成数据与原数据按一定比例合并  
    train.py：训练模型并在测试集上测试    
    pred.py: 加载训练好的模型进行预测  
