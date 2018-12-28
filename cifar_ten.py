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
                    num_workers=2)

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
                    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')






'''
utils.data.DataLoader:
dataset：包含所有数据的数据集

batch_size :每一小组所包含数据的数量

Shuffle : 是否打乱数据位置，当为Ture时打乱数据，全部抛出数据后再次dataloader时重新打乱。

sampler : 自定义从数据集中采样的策略，如果制定了采样策略，shuffle则必须为False.

Batch_sampler:和sampler一样，但是每次返回一组的索引，和batch_size, shuffle, sampler, drop_last 互斥。

num_workers : 使用线程的数量，当为0时数据直接加载到主程序，默认为0。

'''

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.conv2 = nn.Conv2d(6, 16, 5)  
        self.fc1   = nn.Linear(16*5*5, 120)  
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x): 
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) 
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
        x = x.view(x.size()[0], -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return x


net = Net()
print(net)

from torch import optim
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

t.set_num_threads(8)#设定用于并行化CPU操作的OpenMP线程数
for epoch in range(2):  
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
    #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为
    #一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。

    #例如：
    #>>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    #>>> list(enumerate(seasons))
    #[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
    #>>> list(enumerate(seasons, start=1))       # 下标从 1 开始
    #[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]


        # 输入数据
        inputs, labels = data
        
        # 梯度清零
        optimizer.zero_grad()
        
        # forward + backward 
        outputs = net(inputs)#forward
        loss = criterion(outputs, labels)
        loss.backward()   
        
        # 更新参数 
        optimizer.step()
        
        # 打印log信息
        # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
        running_loss += loss.item()
        if i % 2000 == 1999: # 每2000个batch打印一下训练状态
            print('[%d, %5d] loss: %.3f' \
                  % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

correct = 0 # 预测正确的图片数
total = 0 # 总共的图片数


# 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
with t.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = t.max(outputs, 1)#找到每行最大值即预测的分类
        #当max函数中有维数参数的时候，它的返回值为两个，一个为最大值，另一个为最大值的索引
        total += labels.size(0)
        correct += (predicted == labels).sum()

print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))

# 保存模型
t.save(net.state_dict(), 'net.pth')

# 加载已保存的模型
net2 = Net()
net2.load_state_dict(t.load('net.pth'))