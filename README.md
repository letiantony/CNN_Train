### 案件分类训练与部署

####训练
##### 数据准备
*  将所有已打标数据保存在文本文件中，如“data.txt”,每行一条数据，标签和文本试用Tab键分隔
*  试用balance_and_split.py程序将数据集分成训练集、开发集、测试集，保存在文件夹data_set中, 默认命名为train.txt , dev.txt, test.txt
    
            python3 balance_and_split.py data.txt dataset_path 


##### 模型训练
* 主程序为Train 文件下的main.py 
        
                python3 main.py -data-path dataset_path
                * 训练参数保存在“model\_dataset\_path”中
                * 模型文件保存在snapshot文件下，该文件下已训练开始时间命名，找到对应文件夹下最新文件，复制到“model\_dataset\_path”


