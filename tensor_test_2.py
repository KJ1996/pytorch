import torch as t
a = t.arange(0, 6)
a.view(2, 3)
b = a.view(-1, 3) # 当某一维为-1的时候，会自动计算它的大小
print(b.shape)#shape==size()

from matplotlib import pyplot as plt
from IPython import display
'''
b.unsqueeze(1) # 注意形状，在第1维（下标从0开始）上增加“１”
#等价于 b[:,None]
print(b.shape)
print(b[:, None].shape)
'''
b.resize_(3, 3)
print(b)
'''
resize是另一种可用来调整size的方法，但与view不同，它可以修改tensor的大小。
如果新大小超过了原大小，会自动分配新的内存空间，
而如果新大小小于原大小，则之前的数据依旧会被保存
'''
# None类似于np.newaxis, 为a新增了一个轴
# 等价于a.view(1, a.shape[0], a.shape[1])

device = t.device('cpu')
t.manual_seed(1000)

def get_fake_data(batch_size=8):
    ''' 产生随机数据：y=x*2+3，加上了一些噪声'''
    x = t.rand(batch_size, 1, device=device) * 5
    y = x * 2 + 3 +  t.randn(batch_size, 1, device=device)
    return x, y

x, y = get_fake_data(batch_size=16)
plt.scatter(x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy())

# 随机初始化参数
w = t.rand(1, 1).to(device)
b = t.zeros(1, 1).to(device)
w=w.float()
b=b.float()
lr = 0.02  # 学习率
'''
unsqueeze(0) 
squeeze压缩的意思 就是在第几维为1 去掉

unsqueeze 解缩 在第几维增加 变成*1


'''
for ii in range(500):
    x, y = get_fake_data(batch_size=4)
    x=x.float()
    y=y.float()
    # forward：计算loss
    y_pred = x.mm(w) + b.expand_as(y)  # x@W等价于x.mm(w);for python3 only
    y_pred=y_pred.float()
    loss = 0.5 * (y_pred - y) ** 2  # 均方误差
    loss = loss.mean()

    # backward：手动计算梯度
    dloss = 1
    dy_pred = dloss * (y_pred - y)

    dw = x.t().mm(dy_pred)
    db = dy_pred.sum()

    # 更新参数
    w.sub_(lr * dw)
    b.sub_(lr * db)

    if ii % 50 == 0:
        # 画图
        display.clear_output(wait=True)
        x = t.arange(0, 6).view(-1, 1)
        x=x.float()
        y = x.mm(w) + b.expand_as(x)
        plt.plot(x.cpu().numpy(), y.cpu().numpy())  # predicted

        x2, y2 = get_fake_data(batch_size=32)
        plt.scatter(x2.numpy(), y2.numpy())  # true data

        plt.xlim(0, 5)
        plt.ylim(0, 13)
        plt.show()
        plt.pause(0.5)

print('w: ', w.item(), 'b: ', b.item())