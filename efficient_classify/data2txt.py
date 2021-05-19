import os
from sklearn.utils import class_weight
import numpy as np

def path2txt(path,filename):
    '''
    生成对应训练集，测试集图片的路径，以及标签
    :param path: 测试集文件夹路径 test/类别文件夹名
    :return:
    '''

    files = []
    names = []
    for f in os.listdir(path):
        if not f.endswith("~") or not f == "":  # 返回指定的文件夹包含的文件或文件夹的名字的列表
            files.append(os.path.join(path, f))  # 把目录和文件名合成一个路径
            names.append(f)

    print('图片文件夹路径', files)
    print(names)

    for file_path,name in zip(files, names):
        for f in os.listdir(file_path):
            if not f.endswith("~") or not f == "":  # 返回指定的文件夹包含的文件或文件夹的名字的列表
                img_path = os.path.join(file_path, f)
                with open('{}.txt'.format(filename), 'a', encoding='utf-8') as f:
                    f.write(img_path+' '+name+'\n')


def get_class_weights():
    '''
    因类别不均衡，故对类别的权重做一个均衡
    :return:
    '''

    with open('train.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    train_y = []
    for line in lines:
        label = line.split(' ')[1].replace('\n', '')
        train_y.append(label)

    # print('train_y:', train_y)

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(train_y),
                                                      train_y)

    class_weight_dict = dict(enumerate(class_weights))

    # print(class_weight_dict)

    return class_weight_dict



if __name__ == '__main__':
    path2txt(r'.\data\train','train')
    path2txt(r'.\data\test','test')
    # path2txt(r'.\data\eval','eval')

    # get_class_weights()