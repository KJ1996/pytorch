from torchvision import models
from torch import nn
import torch as t
import torchvision as tv
# 加载预训练好的模型，如果不存在会进行下载
# 预训练好的模型保存在 ~/.torch/models/下面
net = models.resnet18(pretrained=True)
net.fc=nn.Linear(512, 10)
print(net)
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch as t
show = ToPILImage() # 可以把Tensor转成Image，方便可视化
# 定义转换方式，transforms.Compose将多个转换函数组合起来使用
'''
transforms.Compose([transforms.ToTensor(),
transforms.Normalize(std=(0.5,0.5,0.5),mean=(0.5,0.5,0.5))])，
则其作用就是先将输入归一化到(0,1)，再使用公式”(x-mean)/std”，将每个元素分布到(-1,1) 
ToTensor,shape=(H x W x C)的像素值范围为[0, 255]
的PIL.Image或者numpy.ndarray转换成shape=(C x H x W)的像素值范围为[0.0, 1.0]的torch.FloatTensor
'''
# 定义对数据的预处理
transform = transforms.Compose([
        transforms.RandomResizedCrop(224),#先将给定的PIL.Image随机切，然后再resize成给定的size大小。

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
                             ])

# 训练集
trainset = tv.datasets.CIFAR10(
                    root='/home/hyj/data/tmp1',
                    train=True,
                    download=True,
                    transform=transform)
trainloader = t.utils.data.DataLoader(
                    trainset,
                    batch_size=4,
                    shuffle=True,
                    )

# 测试集
testset = tv.datasets.CIFAR10(
                    '/home/hyj/data/tmp1',
                    train=False,
                    download=True,
                    transform=transform)

testloader = t.utils.data.DataLoader(
                    testset,
                    batch_size=4,
                    shuffle=False,
                    )

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

from torch import optim

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
test_input=t.rand(1,3,224,224)
output=net(test_input)
print(output.shape)

for epoch in range(10):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为
        # 一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。

        # 例如：
        # >>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        # >>> list(enumerate(seasons))
        # [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
        # >>> list(enumerate(seasons, start=1))       # 下标从 1 开始
        # [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]

        # 输入数据
        inputs, labels = data

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)  # forward
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印log信息
        # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
        running_loss += loss.item()
        if i % 20 == 19:  # 每2000个batch打印一下训练状态
            print('[%d, %5d] loss: %.3f' \
                  % (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
print('Finished Training')
