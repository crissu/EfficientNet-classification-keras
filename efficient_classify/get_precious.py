from efficient import classification_picture
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt

mobile = classification_picture()

def precious(path):

    with open(path, 'r', encoding='utf-8') as f:
        img_list = f.readlines()

    # print(img_list)

    total_img = len(img_list)
    num_of_right = 0
    y_true = []
    y_pred = []
    for img_and_label in tqdm(img_list):
        img_path = img_and_label.split(' ')[0]
        img_label = img_and_label.split(' ')[1].replace('\n','') #把字符串末尾的\n去掉

        img = Image.open(img_path)
        img = img.convert('RGB')
        pre_name = mobile.classification(img)

        y_true.append(img_label)
        y_pred.append(pre_name)

        if img_label == pre_name:
            num_of_right += 1

    result = classification_report(y_true, y_pred, target_names=['chengchong','youchong','luan'])
    print(result)
    # score = num_of_right / total_img
    # print('\nthe precious is %.3f'%score)

    # draw_matrix(y_true, y_pred)


def draw_matrix(y_true,y_pred):
    # 支持中文字体显示, 使用于Mac系统

    classes = ['chengchong','luan','youchong']
    confusion = confusion_matrix(y_true, y_pred)

    # 绘制热度图
    plt.imshow(confusion, cmap=plt.cm.Greens)
    indices = range(len(confusion))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('y_pred')
    plt.ylabel('y_true')

    # 显示数据
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index])

    # 显示图片
    plt.savefig('matrix.jpg')
    plt.show()


if __name__ == '__main__':
    precious('./test.txt')