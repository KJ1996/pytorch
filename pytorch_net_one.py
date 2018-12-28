import torch as t
import torch.nn as nn
import torch.nn.functional as F
'''
把网络中具有可学习参数的层放在构造函数__init__中。
如果某一层(如ReLU)不具有可学习的参数，则既可以放在构造函数中，
也可以不放，但建议不放在其中，而在forward中使用nn.functional代替。
'''
class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(Net,self).__init__()
        # 卷积层 '1'表示输入图片为单通道(灰度图）, '6'表示输出通道数（卷积核数），'5'表示卷积核为5*5
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射层/全连接层，y = Wx + b
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):# 卷积 -> 激活 -> 池化

        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        print("x:",x.size())

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print(x.size())
        # reshape，‘-1’表示自适应
        x = x.view(x.size()[0], -1)
        print(x.size())
        x = F.relu(self.fc1(x))
        print(x.size())
        x = F.relu(self.fc2(x))
        print(x.size())
        x = self.fc3(x)
        print(x.size())
        return x
net = Net()
#print(net)

params = list(net.parameters())
print(len(params))
for name,parameters in net.named_parameters():
    print(name,':',parameters.size())

input = t.randn(1, 1, 32, 32)
print(input.size())
out = net(input)
print(out.size())
'''
data = t.FloatTensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
#t1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
#y=t1(input)
print("y:",y.size())
res = F.max_pool2d(data, kernel_size=(2, 2), )

print(res)

net.zero_grad()# 所有参数的梯度清零
out.backward(t.ones(1,10)) # 反向传播
'''
output = net(input)
target = t.arange(0,10).view(1,10)
#torch.arange(start, end, step=1, out=None) → Tensor,包含从start到end，以step为步长的一组序列值
#view,在不改变张量数据的情况下随意改变张量的大小和形状(10,1)->(1,10)
'''
k=t.arange(0,3)
print("k1:",k)
print("k2",k.view(1,3))
'''
target =target.float()#强制转化为float
criterion = nn.MSELoss()
loss = criterion(output, target)
#例如nn.MSELoss用来计算均方误差，nn.CrossEntropyLoss用来计算交叉熵损失
#print(loss)

# 运行.backward，观察调用之前和调用之后的grad
net.zero_grad() # 把net中所有可学习参数的梯度清零
print('反向传播之前 conv1.bias的梯度')
print(net.conv1.bias.grad)
loss.backward()
print('反向传播之后 conv1.bias的梯度')
print(net.conv1.bias.grad)

import torch.optim as optim
#新建一个优化器，指定要调整的参数和学习率
optimizer = optim.SGD(net.parameters(), lr = 0.01)
#torch.optim中实现了深度学习中绝大多数的优化方法，例如RMSProp、Adam、SGD等

# 在训练过程中
# 先梯度清零(与net.zero_grad()效果一样)
optimizer.zero_grad()

# 计算损失
output = net(input)
loss = criterion(output, target)

#反向传播
loss.backward()

#更新参数
optimizer.step()