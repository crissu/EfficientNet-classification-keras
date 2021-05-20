# 使用EfficientNet训练自己的分类器

CSDN：https://blog.csdn.net/qq_42940160/article/details/117049302

## 1.requirements

```c
python=3.6  #必须
keras=2.1.5
tensorflow-gpu=1.13.1
  
其他的环境在运行代码时 缺什么，则使用conda安装什么
```

## 2.目录说明

### 1.data-训练集格式

![image-20210519223523853](https://i.loli.net/2021/05/19/vXJAaibyTeugQ2O.png)

将图片数据保存为这样的格式，然后运行 data2txt.py 生成train.txt训练集和test.txt测试集对应的txt文件

![image-20210519224047945](https://i.loli.net/2021/05/19/lr4SALR7M3GiBKt.png)

### 2.efficientnet-存放模型结构

### 3.img-存放用于测试的数据

### 4.logs-存放训练过程中保存的模型

### 5.model_data-训练好的模型将从这里加载

### 6.utils-工具文件夹，包含一些数据处理方法，学习率优化函数等工具方法文件



## 3.如何训练自己的模型

### 1.按2中准备好数据运行 data2txt.py 生成训练集，测试集

### 2.运行train.py

在该文件的底部修改自己的模型参数

```c
# 修改相应的参数
    FLAGS = {
         'data_local': './train.txt',   # 训练数据集路径
         'batch_size': 8,           # 如果爆显存 把batch_size改小一点
        'num_classes': 3,           # 期望分的类别
         'input_size': 400,         # 图片尺寸，默认为300x300，现存越大，输入尺寸也可越大
      'learning_rate': 1e-3,        # 学习率
       'class_weight': True,        # 是否开启class_weight, 当类别不均衡时建议开启

             'epochs': 300,         # 训练轮数
        'pre_weights': '',          # 使用迁移学习时，此除填写预训练权重路径，存放在model_data路径下
   'imagenet_weights': 'imagenet'   # 是否加载imagenet预训练权重，默认值为'imagenet'，使用None不加载与训练权重，数据量少时建议开启
    }

    # 标签顺序，需要和 train.txt文件中从上到下的标签顺序一致，否则class_weight的key无法和标签对应上
    str2int = {'chengchong':0, 'luan':1,'youchong':2}  #将字符标签转换成 int 标签，方便模型训练，训练自己的模型时注意修改为自己的类别

    train_model(FLAGS,str2int)
```

训练过程中的模型保存在logs文件夹下

### 3.在efficient.py中修改模型权重-pre_weights

```c
_FLAGS = {
        'num_classes': 3,
        'input_size': 300,
        'learning_rate': 1e-3,
        'pre_weights': './model_data/logsep001-loss0.787-val_loss0.608.h5',  # 预训练权重路径
    }
```

运行efficient.py，对单张图片进行预测

### 4.运行get_precious.py获取模型的预测准确率



## 3.剩下的工作自己调参去吧！！！
