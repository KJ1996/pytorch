from torchvision import models
from torch import nn
import torch as t
import torch.nn.functional as F
import numpy as np
from itertools import combinations
from torch.nn import DataParallel
device_ids = [0,1,2,3]
net = models.resnet18(pretrained=True)
net.fc=nn.Linear(512, 40)
device = t.device("cuda:1" if t.cuda.is_available() else "cpu")
net.to(device)

import torchvision as tv
import torchvision.transforms as transforms

#定义triple类
class triple:


    def __init__(self, input1, input2,input3):
        self.input1 = input1
        self.input2 = input2
        self.input3 = input3
    def show(self):
        print("1:",self.input1)
        print("2:", self.input2)
        print("3:", self.input3)


#
path1='/home/hyj/data/tmp1'
path2='/home/disk3/huyujie/data'
# 定义对数据的预处理
transform = transforms.Compose([

        transforms.RandomResizedCrop(224),#先将给定的PIL.Image随机切，然后再resize成给定的size大小。

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
                             ])

# 训练集
trainset = tv.datasets.CIFAR10(
                    root=path1,

                    train=True,
                    download=True,
                    transform=transform)
trainloader = t.utils.data.DataLoader(
                    trainset,
                    batch_size=32,
                    shuffle=True,
                    #num_workers=2
)

# 测试集
testset = tv.datasets.CIFAR10(
                    root=path1,

                    train=False,
                    download=True,
                    transform=transform)

testloader = t.utils.data.DataLoader(
                    testset,
                    batch_size=32,
                    shuffle=False,
                    #num_workers=2
                    )

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#定义Average precision(AveP)
def Average_precision(retrieved_list,label):
    sum=len(retrieved_list)#检索所得列表的总长,检索序列中仅含标签
    index=0.0#当前为第几个相关结果
    count =0.0#求和
    for c in range(sum):
        if(retrieved_list[c]==label):
            index=index+1
            count=count+index/(c+1)

    res=count/sum
    return res

#定义map:Mean average precision(MAP)
def Mean_average_precision(q_num,AveP_num):
    print("map:",AveP_num/q_num)
    return AveP_num/q_num

def get_triplets(labels):
    labels = labels.cpu().data.numpy()
    triplets = []
    for label in set(labels):
        label_mask = (labels == label)
        label_indices = np.where(label_mask)[0]
        if len(label_indices) < 2:
            continue
        negative_indices = np.where(np.logical_not(label_mask))[0]
        anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

        # Add all negatives for all positive pairs
        temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                         for neg_ind in negative_indices]
        triplets += temp_triplets

    return t.LongTensor(np.array(triplets))

def triplet_hashing_loss_regu(embeddings, cls, margin):
    triplets = get_triplets(cls)

    if embeddings.is_cuda:
        triplets = triplets.cuda()

    ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
    an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)

    losses = F.relu(ap_distances - an_distances + margin)

    return losses.mean()
for i, data in enumerate(trainloader, 0):
        # 输入数据
            inputs, labels = data
            print(labels)

'''
from torch import optim
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
if __name__ == '__main__':
    for epoch in range(10):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
        # 输入数据
            inputs, labels = data
            print(labels.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)
        # 梯度清零
            optimizer.zero_grad()

        # forward + backward
            outputs = net(inputs)  # forward
            hash_out = t.sigmoid(outputs)
            loss = triplet_hashing_loss_regu(hash_out, labels, 1)
            loss.backward()

        # 更新参数
            optimizer.step()

        # 打印log信息
        # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
            running_loss += loss.item()
            if i % 50 == 49:  # 每50个batch打印一下训练状态
                print('[%d, %5d] loss: %.3f' \
                    % (epoch + 1, i/50 + 1, running_loss / 50))
                running_loss = 0.0
    print('Finished Training')

# 保存模型
    t.save(net.state_dict(), 'net.pth')

'''



