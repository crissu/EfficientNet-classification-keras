# -*- coding: utf-8 -*-
import os

from keras import backend
from keras.callbacks import TensorBoard, Callback,ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

from utils.data_gen_label import data_flow
from utils.warmup_cosine_decay_scheduler import WarmUpCosineDecayScheduler
import efficientnet.keras as efn
from sklearn.utils import class_weight
import numpy as np
backend.set_image_data_format('channels_last')

os.environ['CUDA_VISIBLE_DEVICES'] = "2"


def get_class_weights():
    with open('train.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    train_y = []
    for line in lines:
        label = line.split(' ')[1].replace('\n','')
        train_y.append(label)

    # print('train_y:',train_y)

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(train_y),
                                                      train_y)
    # class_weight需要为字典形式
    class_weight_dict = dict(enumerate(class_weights))

    print('the class_weight is:',class_weight_dict)

    return class_weight_dict


def choose_EfficientNet(FLAGS):
    # 通过id控制选择那个版本的EfficientNet
    if FLAGS['id'] == 0:
        base_model = efn.EfficientNetB0(include_top=False, weights=FLAGS['imagenet_weights'],
                                        input_shape=(FLAGS['input_size'], FLAGS['input_size'], 3),
                                        classes=FLAGS['num_classes'])
    elif FLAGS['id'] == 1:
        base_model = efn.EfficientNetB1(include_top=False, weights=FLAGS['imagenet_weights'],
                                        input_shape=(FLAGS['input_size'], FLAGS['input_size'], 3),
                                        classes=FLAGS['num_classes'])
    elif FLAGS['id'] == 2:
        base_model = efn.EfficientNetB2(include_top=False, weights=FLAGS['imagenet_weights'],
                                        input_shape=(FLAGS['input_size'], FLAGS['input_size'], 3),
                                        classes=FLAGS['num_classes'])
    elif FLAGS['id'] == 3:
        base_model = efn.EfficientNetB3(include_top=False, weights=FLAGS['imagenet_weights'],
                                        input_shape=(FLAGS['input_size'], FLAGS['input_size'], 3),
                                        classes=FLAGS['num_classes'])
    elif FLAGS['id'] == 4:
        base_model = efn.EfficientNetB4(include_top=False, weights=FLAGS['imagenet_weights'],
                                        input_shape=(FLAGS['input_size'], FLAGS['input_size'], 3),
                                        classes=FLAGS['num_classes'])
    elif FLAGS['id'] == 5:
        base_model = efn.EfficientNetB5(include_top=False, weights=FLAGS['imagenet_weights'],
                                        input_shape=(FLAGS['input_size'], FLAGS['input_size'], 3),
                                        classes=FLAGS['num_classes'])
    elif FLAGS['id'] == 6:
        base_model = efn.EfficientNetB6(include_top=False, weights=FLAGS['imagenet_weights'],
                                        input_shape=(FLAGS['input_size'], FLAGS['input_size'], 3),
                                        classes=FLAGS['num_classes'])
    elif FLAGS['id'] == 7:
        base_model = efn.EfficientNetB7(include_top=False, weights=FLAGS['imagenet_weights'],
                                        input_shape=(FLAGS['input_size'], FLAGS['input_size'], 3),
                                        classes=FLAGS['num_classes'])

    print('EfficientNetB{} is loaded\n'.format(FLAGS['id']))
    return base_model


def model_fn(FLAGS, objective, optimizer, metrics):
    '''
    一共有 EfficientNetB0-7可以选择，数字从高到底，模型越来越大，这样意味着模型越来越准确
    weights=None, 表示不使用 官方的在imagenet数据集上的权重，如需要请改为：weights='imagenet'
    '''
    # 使用efficientnetB6
    # base_model = efn.EfficientNetB6(include_top=False, weights=FLAGS['imagenet_weights'],
    #                                 input_shape=(FLAGS['input_size'], FLAGS['input_size'], 3),
    #                                 classes=FLAGS['num_classes'])

    base_model = choose_EfficientNet(FLAGS)

    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = Dense(FLAGS['num_classes'], activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    model.summary( )
    return model


def train_model(FLAGS, str2int):
    preprocess_input = efn.preprocess_input

    # 加载数据
    train_sequence, validation_sequence = data_flow(FLAGS['data_local'], FLAGS['batch_size'],
                                                    FLAGS['num_classes'], FLAGS['input_size'], preprocess_input, str2int)

    #============================================================
    optimizer = Adam(lr=FLAGS['learning_rate'])

    objective = 'categorical_crossentropy'
    metrics = ['accuracy']

    model = model_fn(FLAGS, objective, optimizer, metrics)
    # 是否加载与训练权重
    if os.path.exists(FLAGS['pre_weights']):
        print('use pre_trained model {}'.format(FLAGS['pre_weights']))
        model.load_weights(FLAGS['pre_weights'])
    #============================================================

    log_dir = './logs/'
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_acc', save_best_only=True, save_weights_only=True)


    #============================================================
    num_train = len(train_sequence)  # 训练集个数
    sample_count = len(train_sequence) * FLAGS['batch_size']
    epochs = FLAGS['epochs']
    warmup_epoch = 10
    batch_size = FLAGS['batch_size']
    learning_rate_base = FLAGS['learning_rate']
    total_steps = int(epochs * sample_count / batch_size)
    warmup_steps = int(warmup_epoch * sample_count / batch_size)

    # 学习率
    reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                            total_steps=total_steps,
                                            warmup_learning_rate=1e-4,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=num_train,
                                            )
    #============================================================

    if FLAGS['class_weight'] == True:

        # 计算class_weight
        class_weight = get_class_weights()

        model.fit_generator(
            train_sequence,
            steps_per_epoch=len(train_sequence),
            epochs=epochs,
            verbose=1,
            callbacks=[checkpoint, logging, reduce_lr],
            validation_data=validation_sequence,
            max_queue_size=10,
            shuffle=True,
            class_weight=class_weight
        )

    else:
        model.fit_generator(
            train_sequence,
            steps_per_epoch=len(train_sequence),
            epochs=epochs,
            verbose=1,
            callbacks=[checkpoint, logging, reduce_lr],
            validation_data=validation_sequence,
            max_queue_size=10,
            shuffle=True,
        )

    print('training done!')



if __name__ == '__main__':
    # 修改相应的参数
    '''
    不同模型对应的input_size，以下将EfficientNetBX简称EN，尽管可以随意写input_size，但请按照论文中的要求去写
            ENB0：224
            ENB1：240
            ENB2：260
            ENB3：300
            ENB4：380
            ENB5：456
            ENB6：528
            ENB7：600
    '''

    FLAGS = {
         'data_local': './train.txt',   # 训练数据集路径
         'batch_size': 8,            # 如果爆显存 把batch_size改小一点
        'num_classes': 3,            # 期望分的类别
                 'id': 4,            # 选择不同版本的EfficientNet网络，值为0-7
        'input_size': 380,           # 输入网络的图片分辨率，参数在上面注释中
      'learning_rate': 1e-3,         # 学习率
       'class_weight': True,         # 是否开启class_weight, 当类别不均衡时建议开启

             'epochs': 300,          # 训练轮数
        'pre_weights': '',           # 使用迁移学习时，此除填写预训练权重路径，存放在model_data路径下
   'imagenet_weights': 'imagenet',   # 是否加载imagenet预训练权重，默认值为'imagenet'，使用None不加载与训练权重，数据量少时建议开启
    }

    # 标签顺序，需要和 train.txt文件中从上到下的标签顺序一致，否则class_weight的key无法和标签对应上
    str2int = {'chengchong':0, 'luan':1,'youchong':2}  #将字符标签转换成 int 标签，方便模型训练，训练自己的模型时注意修改为自己的类别

    train_model(FLAGS,str2int)