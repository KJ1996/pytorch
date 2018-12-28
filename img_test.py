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

def get_triplet(hash_out,labels,class_num):
    nums = []  # 用于存放每一个分类的所有下标的
    triplet_list=[]#最终的返回的所有三元组的集合
    for c in range(class_num):
        nums.append([])#扩展维度
    for i, label in enumerate(labels, 0):
        nums[label].append(i)
    #print(nums)
    for i in range(class_num):
        if(len(nums[i])>1):
            index_num=len(nums[i])#每一个分类的总样本数，大于1才能作为input1,2
            for j in range(index_num):

                for k in range(j+1,index_num,1):
                    for index,label in enumerate(labels,0):
                        if(labels[index]!=i):
                            temp_tri=triple((hash_out[nums[i][j]]).resize(1,40),(hash_out[nums[i][k]]).resize(1,40),(hash_out[index]).resize(1,40))
                            #temp_tri=triple(nums[i][j],nums[i][k],index)
                            triplet_list.append(temp_tri)

    return triplet_list

def get_loss(triple_list):
    input1=t.zeros(1,40).cuda()
    input2= t.zeros(1, 40).cuda()
    input3= t.zeros(1, 40).cuda()
    loss =triplet_loss(input1,input2,input3)
    #print(triple_list[0].input3.shape)
    for i in triple_list:

        loss=loss+triplet_loss(i.input1,i.input2,i.input3)

    return loss/len(triple_list)
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

(data, label) = trainset[100]
print(classes[label])

# (data + 1) / 2是为了还原被归一化的数据
(show((data + 1) / 2).resize((400, 100))).show()