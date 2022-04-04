from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np


def plot_history(epochs, Acc, Loss, lr):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    epoch_list = range(1, epochs + 1)
    plt.plot(epoch_list, Loss['train_loss'])
    plt.plot(epoch_list, Loss['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('Loss Value')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('vis/history_Loss.png')
    plt.show()

    plt.plot(epoch_list, Acc['train_acc'])
    plt.plot(epoch_list, Acc['val_acc'])
    plt.xlabel('epoch')
    plt.ylabel('Acc Value')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('vis/history_Acc.png')
    plt.show()

    plt.plot(epoch_list, lr)
    plt.xlabel('epoch')
    plt.ylabel('Train LR')
    plt.savefig('vis/history_Lr.png')
    plt.show()


def get_acc(outputs, label):
    total = outputs.shape[0]
    probs, pred_y = outputs.data.max(dim=1)  # 得到概率
    correct = (pred_y == label).sum().data
    return correct / total


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def draw_confusion_matrix(device, class_kinds, conf_matrix, testloader, network, labels):
    with torch.no_grad():
        for step, (imgs, targets) in enumerate(testloader):

            targets = targets.squeeze()  # [50,1] ----->  [50]
            # 将变量转为gpu
            targets = targets.to(device)
            imgs = imgs.to(device)

            out = network(imgs)
            # 记录混淆矩阵参数
            conf_matrix = confusion_matrix(out, targets, conf_matrix)
            conf_matrix = conf_matrix.cpu()

    conf_matrix = np.array(conf_matrix.cpu())     # 将混淆矩阵从gpu转到cpu再转到np
    corrects = conf_matrix.diagonal(offset=0)     # 抽取对角线的每种分类的识别正确个数
    per_kinds = conf_matrix.sum(axis=1)           # 抽取每个分类数据总的测试条数

    print(conf_matrix)
    # 显示数据
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(class_kinds):
        for y in range(class_kinds):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")

    plt.tight_layout()  # 保证图不重叠
    plt.yticks(range(class_kinds), labels)
    plt.xticks(range(class_kinds), labels, rotation=45)  # X轴字体倾斜45°
    plt.savefig('vis/confusion_matrix.png')
    plt.show()
    plt.close()
