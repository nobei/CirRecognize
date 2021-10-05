import itertools

import seaborn as sn

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def plot_Matrix(cm, classes, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='Times New Roman', size='12')  # 设置字体样式、大小

    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('cnnAndLSTMConfusion.pdf',dpi=300)
    plt.show()





def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

#
# df_cm = pd.DataFrame(conf_matrix.numpy(),
#                      index = [i for i in list(Attack2Index.keys())],
#                      columns = [i for i in list(Attack2Index.keys())])
# plt.figure(figsize = (10,7))
# sn.heatmap(df_cm, annot=True, cmap="BuPu")
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig("end1")

def plotCM(classes, matrix, savname):
    """classes: a list of class names"""
    # Normalize by row
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    # plot
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    cax = ax.matshow(matrix)
    #fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
    ax.set_xticklabels([''] + classes)
    ax.set_yticklabels([''] + classes)
    #save
    plt.savefig(savname)


def cm_plot(cm, pic=None):
    #cm = confusion_matrix(original_label, predict_label)   # 由原标签和预测标签生成混淆矩阵
    plt.figure()
    plt.matshow(cm, cmap=plt.cm.Blues)     # 画混淆矩阵，配色风格使用cm.Blues
    plt.colorbar()    # 颜色标签
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
            # annotate主要在图形中添加注释
            # 第一个参数添加注释
            # 第二个参数是注释的内容
            # xy设置箭头尖的坐标
            # horizontalalignment水平对齐
            # verticalalignment垂直对齐
            # 其余常用参数如下：
            # xytext设置注释内容显示的起始位置
            # arrowprops 用来设置箭头
            # facecolor 设置箭头的颜色
            # headlength 箭头的头的长度
            # headwidth 箭头的宽度
            # width 箭身的宽度
    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    plt.title('confusion matrix')
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., handleheight=1.675)

    if pic is not None:
        plt.savefig(str(pic) + '.jpg')
    plt.show()

def plot_confusion_matrixEnd(cm, labels_name, title):
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
import pandas as pd
if __name__ == '__main__':
    classes = ['push','double click','pull','rotation','grab','release','upDown']
    # f = open('C:/Users/hao/Desktop/6-12 LSTMAndCnn.txt')
    # confusionArray = np.zeros((4, 4))
    # k = 0
    # for line in f.readlines():
    #     data = line.split(',')
    #     all = float(data[0]) + float(data[1]) + float(data[2]) + float(data[3][:-1])
    #     confusionArray[k, 0] = float(data[0])
    #     confusionArray[k, 1] = float(data[1])
    #     confusionArray[k, 2] = float(data[2])
    #     confusionArray[k, 3] = float(data[3][:-1])
    #     k = k + 1

    confusionArray = [[65.0,0.0,0.0,0.0,0.0,0.0,0.0],
                      [0.0,57.0,0.0,0.0,0.0,0.0,0.0],
                      [0.0,0.0,59.0,0.0,0.0,0.0,0.0],
                      [0.0,0.0,0.0,50.0,0.0,0.0,1.0],
                      [0.0,0.0,0.0,0.0,72.0,1.0,0.0],
                      [0.0,0.0,0.0,1.0,0.0,38.0,0.0],
                      [0.0,0.0,0.0,0.0,0.0,0.0,68.0]]

    confusionArray = np.array(confusionArray)

    plot_Matrix(confusionArray,classes)
    # df_cm = pd.DataFrame(confusionArray,index=[i for i in classes],columns=[i for i in classes])
    # plt.figure(figsize=(10,7))
    # sn.heatmap(df_cm, annot=True, cmap="BuPu")

    # random_numbers = np.random.randint(6, size=50)  # 6个类别，随机生成50个样本
    # y_true = random_numbers.copy()  # 样本实际标签
    # random_numbers[:10] = np.random.randint(6, size=10)  # 将前10个样本的值进行随机更改
    # y_pred = random_numbers  # 样本预测标签
    #
    # # 获取混淆矩阵
    # #cm = confusion_matrix(y_true, y_pred)
    # plot_confusion_matrix(confusionArray, 'confusion_matrix.png', title='confusion matrix')
    # plotCM(classes,confusionArray,'LSTMAndCNN')
    # df_cm = pd.DataFrame(confusionArray.numpy(),
    #                      index=[i for i in classes],
    #                      columns=[i for i in list(classes]))
    # plt.figure(figsize=(10, 7))
    # sn.heatmap(df_cm, annot=True, cmap="BuPu")
