# Rihgaruti  


### task1  
中文文本情感分析模型  
训练所用数据集：[weibo_senti_100k](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/weibo_senti_100k/intro.ipynb)  
分类标签：1为正面情感，0为负面情感  
模型：词嵌入+LSTM双向RNN(准确度96%)  

#### 文件结构  
dataset.py：定义了部分超参数、dataloader等  
model.py：使用LSTM单元的双向RNN模型  
train.py：训练、验证、测试  
run.py：根据输入的句子判断所含的正负情感  

### task2  
猫狗图片分类([Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats))  
训练所用数据集：[猫狗图片](https://www.kaggle.com/c/dogs-vs-cats/data)  
包含两种模型：一个是model.py中自己试着写的简陋模型(准确度75%)，一个是torchvison.models中的ResNet18(准确度91%)  

#### 文件结构  
config.py：定义了超参数等其他重要设置  
model.py：里面的模型由两次卷积两次池化再加上三次的全连接构成  
train.py：训练并进行验证  
run.py：随机抽取测试集测试，也可输入自己提供的图片  


## 环境
Win10  
Python3.7  
Pytorch1.5.1  
Cuda11.0  


## 启动
1. 下载数据集，根据config.py/dataset.py配置运行环境
2. 注意task2要求将数据集中train文件夹的猫狗图片分类至该目录下不同文件夹，如 'train/cat' 与 'train/dong'  
3. 启动train.py训练模型
4. 启动run.py使用训练好的模型处理自己的输入
