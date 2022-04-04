import os
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from model import AlexNet
from metrics import plot_history, get_acc,draw_confusion_matrix

if not os.path.exists('./model'):
    os.makedirs('./model')


def train(model, device, trainloader, valloader, optimizer, criterion, epochs=100, path='./model', pretrain=''):
    if pretrain != '':
        print('Load weights {}.'.format(pretrain))
        model.load_state_dict(torch.load(pretrain))

    best_acc = 0
    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []
    lr_list = []
    epoch_step = len(trainloader)
    epoch_step_val = len(valloader)

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集，或者减小batchsize")

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        if torch.cuda.is_available():
            model = model.to(device)
        model.train()
        print('Start Train')
        with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
            for step, data in enumerate(trainloader, start=0):
                img, label = data
                img = img.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                outputs = model(img)
                loss = criterion(outputs, label)
                train_loss += loss.data
                train_acc += get_acc(outputs, label)
                loss.backward()
                optimizer.step()

                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(**{'Train Loss': train_loss.item() / (step + 1),
                                    'Train Acc': train_acc.item() / (step + 1),
                                    'Lr': lr})
                pbar.update(1)
        train_loss = train_loss.item() / len(trainloader)
        train_acc = train_acc.item() * 100 / len(trainloader)

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        # 记录学习率
        lr = optimizer.param_groups[0]['lr']

        lr_list.append(lr)

        print('Finish Train')

        model.eval()
        print('Start Validation')
        with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar2:
            for step, data in enumerate(valloader, start=0):
                img, label = data
                img = img.to(device)
                label = label.to(device)
                with torch.no_grad():
                    if step >= epoch_step_val:
                        break

                    # 释放内存
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()

                    outputs = model(img)
                    loss = criterion(outputs, label)
                    val_loss += loss.data
                    val_acc += get_acc(outputs, label)

                    pbar2.set_postfix(**{'Val Acc': val_acc.item() / (step + 1),
                                         'Val Loss': val_loss.item() / (step + 1)})
                    pbar2.update(1)

        val_loss = val_loss.item() / len(valloader)
        val_acc = val_acc.item() * 100 / len(valloader)

        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        print('Finish Validation')

        print(
            "Epoch [{:>3d}/{:>3d}]  Train Acc: {:>3.2f}%  Train Loss: {:>.6f} || Val Acc: {:>3.2f}% Val Loss: {:>.6f} || Learning Rate:{:>.6f}"
            .format(epoch + 1, epochs, train_acc, train_loss, val_acc, val_loss, lr))

        if (epoch+1) % 10 == 0:
            torch.save(model, path + '/AlexNet_epoch_' + str(epoch+1) + '.pth')
    torch.cuda.empty_cache()

    Acc = {}
    Loss = {}
    Acc['train_acc'] = train_acc_list
    Acc['val_acc'] = val_acc_list
    Loss['train_loss'] = train_loss_list
    Loss['val_loss'] = val_loss_list
    Lr = lr_list
    return Acc, Loss, Lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=100, help="epoch number in training")
    parser.add_argument('-b', '--batch_size', type=int, default=256, help="batch size")
    parser.add_argument('-p', '--path', type=str, default='./model', help="save model path")

    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # 数据增广
        transforms.RandomHorizontalFlip(),      # 数据增广
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    traindataset = datasets.CIFAR10(root='./dataset', train=True, download=False, transform=transform)
    testdataset = datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    net = AlexNet().to(device)
    if device == 'cuda':
        net = nn.DataParallel(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    Acc, Loss, Lr = train(net, device=device, trainloader=trainloader, valloader=testloader, epochs=args.epoch,
                          optimizer=optimizer,
                          criterion=criterion,
                          path=args.path)

    plot_history(args.epoch, Acc, Loss, Lr)

    # 绘制混淆矩阵
    class_kinds = len(classes)
    conf_matrix = torch.zeros(class_kinds, class_kinds)
    labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # 每种类别的标签
    draw_confusion_matrix(device=device,
                          class_kinds=class_kinds,
                          conf_matrix=conf_matrix,
                          testloader=testloader,
                          network=net,
                          labels=labels)

