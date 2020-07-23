## Rihgaruti
这是Kaggle上一道经典赛题：[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats)  
训练所用数据集：[猫狗图片](https://www.kaggle.com/c/dogs-vs-cats/data)  
包含两种模型：一个是model.py中自己试着写的简陋模型(准确度75%)，一个是torchvison.models中的ResNet18(准确度91%)  


## 文件结构
config.py：定义了超参数等其他重要设置  
model.py：里面的模型由两次卷积两次池化再加上三次的全连接构成  
train.py：训练并进行验证
run.py：随机抽取测试集测试，也可输入自己提供的图片  


## 环境
Win10, Python3.7, Pytorch1.5.1, Cuda11.0


## 启动
1. 下载数据集，并配置config.py中的数据集路径。注意应将数据集中train文件夹的猫狗图片分类至该目录下不同文件夹，如 'train/cat' 与 'train/dong'  
2. 运行train.py以训练模型
3. 运行run.py使用已有模型进行分类
4. 可更改train与run的相关代码启用不同模型
