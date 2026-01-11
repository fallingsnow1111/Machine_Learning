本地生成融合数据后上传，在kaggle上拉取项目，并运行train.py即可  
1.pt文件夹：保存模型文件  
2.Code中代码:  
    &#9 synthetic_data.py：生成合成数据  
    &#9 merge_data.py：将合成数据与原数据按一定比例合并  
    &#9 train.py：训练模型并在测试集上测试    
    &#9 pred.py: 加载训练好的模型进行预测  
3.Data文件夹：保存数据集，noise11表示有噪声，train中原始数据与合成数据1:1  
    &#9 Raw文件夹：保存原始数据集  
    &#9 Synthetic文件夹：保存合成数据集    
    &#9 Merged文件夹：保存合并后的数据集  

