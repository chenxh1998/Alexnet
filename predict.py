import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
from model import AlexNet

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img', type=str, default='./test_picture/cat/mao2.jpg', help="image path")
parser.add_argument('-p', '--path', type=str, default='./model/AlexNet_epoch_100.pth', help="model path")

args = parser.parse_args()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(args.path)  # 加载模型
# model = model.to('cuda')
model.eval()  # 把模型转为test模式
# 读取要预测的图片
img = Image.open(args.img).convert('RGB')  # 读取图像
trans = transforms.Compose([transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                 std=(0.5, 0.5, 0.5)),
                            ])

img = trans(img)
img = img.to(device)
# 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
img = img.unsqueeze(0)
# 扩展后，为[1，1，28，28]
output = model(img)
prob = F.softmax(output, dim=1)  # prob是10个分类的概率
print("概率", prob)
value, predicted = torch.max(output.data, 1)
print("类别", predicted.item())
print(value)
pred_class = classes[predicted.item()]
print("分类", pred_class)




