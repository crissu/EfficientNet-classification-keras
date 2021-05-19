import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image,ImageDraw, ImageFont
import os
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

import efficientnet.keras as efn


class classification_picture:

    _FLAGS = {
        'num_classes': 3,
        'input_size': 300,
        'learning_rate': 1e-3,
        'pre_weights': './model_data/logsep001-loss0.787-val_loss0.608.h5',  # 预训练权重路径
    }

    def __init__(self):
        self.__dict__.update(self._FLAGS)
        self.model = self.load_model()


    def load_model(self):
        '''
        因训练时进保存了模型参数，因此需要对模型进行构建
        :param model_path: 模型路径，因未知bug，模型必须与predict文件同目录
        :return:
        '''

        optimizer = Adam(lr=self.learning_rate)
        objective = 'categorical_crossentropy'
        metrics = ['accuracy']

        # 使用B7网络
        base_model = efn.EfficientNetB7(include_top=False,
                                        shape=(self.input_size, self.input_size, 3),
                                        n_class=self.num_classes)

        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss=objective, optimizer=optimizer, metrics=metrics)

        print('model {} is loaded'.format(self.pre_weights))
        model.load_weights(self.pre_weights)  # 加载权重

        return model


    def classification(self, img):
        '''
        对图片进行预测
        :param img: 图片路径
        :param model: 模型名称
        :return: pre_name返回预测的名字; pre_type返回预测的索引
        '''
        classes_name_list = ['chengchong', 'youchong', 'luan']

        test_img = img.resize((300, 300))
        test_img = np.array(test_img)  # 将Image实例转化为多维数组
        test_img = test_img / 255  # 此处还需要将0-255转化为0-1
        test_img = np.expand_dims(test_img, 0)  # 将三维输入图像拓展成四维张量
        pred = self.model.predict(test_img)  # 预测
        # print('pred:',pred)
        # print('分类预测结果：', end='')
        # print(classes_name_list[pred.argmax()])  # 打印结果
        pre_type = pred.argmax()
        pre_name = classes_name_list[pred.argmax()]

        return pre_name
        # return pre_type


if __name__ == '__main__':
    efficient = classification_picture()

    while True:
        try:
            img = input('please input the file path:')
            img = Image.open(img)
        except:
            print('can not open the file,please try again!')
        else:
            pre_name = efficient.classification(img)
            print('预测结果为：',pre_name)