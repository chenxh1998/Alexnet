# Read me

### This is a pytorch implementation of Alexnet. The dataset adopts cifar-10. After the training is completed, the loss curve and the accuracy curve can be generated, and the corresponding confusion matrix can be generated at the same time.

## How to use

## 1. Prepare dataset

### Firstly, you shouled download the cifar-10 from the website.Then put the cifar-10 into data.

## 2.Training

### `python --epoch 100 --batch_size 256`

## 3.Predict

### `python predict.py --img ./test_picture/cat.jpg --path ./model/AlexNet_epoch_100.pth`