#from __future__ import print_function#让print函数加括号使用

import torch as t
import numpy as np

print(t.__version__)#返回torch的版本号

'''
x=t.Tensor(5,3)#仅分配空间，未初始化
x=t.Tensor([[1,3],])#初始化
print(x)
'''

'''
x=t.rand(5,2)#生成随机的[0,1]均匀分布
print(x)
print(x[:,1])

print(x.size()) # 查看x的形状
print(x.size(1))#查看第一列的长度


y=t.rand(5,2)
print(y)
print('第一种加法，y的结果')
y.add(x) # 普通加法，不改变y的内容
print(y)

print('第二种加法，y的结果')
y.add_(x) # inplace 加法，y变了
print(y)

注意，函数名后面带下划线_ 的函数会修改Tensor本身。例如，x.add_(y)和x.t_()会改变 x，但x.add(y)和x.t()返回一个新的Tensor， 而x不变。
'''
'''
a = t.ones(5) # 新建一个全1的Tensor
print(a)
b = a.numpy()
print(b)

a = np.ones(5)
b = t.from_numpy(a) # Numpy->Tensor
print(a)
print(b)

b.add_(1) # 以`_`结尾的函数会修改自身
print(a)
print(b) # Tensor和Numpy共享内存
scalar = b[0]
print(scalar)#直接tensor[idx]得到的还是一个tensor: 一个0-dim 的tensor，一般称为scalar.
print(scalar.item())#使用scalar.item()能从中取出python对象的数值
ten=t.Tensor([2])
print(scalar.size())
print(ten.size())#ten是1唯的，scalar是0维的

# 只有一个元素的tensor也可以调用`tensor.item()`
print(ten.item())
'''
'''
tensor = t.tensor([3,4])#此外在pytorch中还有一个和np.array 很类似的接口: torch.tensor, 二者的使用十分类似。
scalar = t.tensor(3)
print(scalar)

old_tensor = tensor
new_tensor = t.tensor(old_tensor)
new_tensor[0] = 1111
print(old_tensor,new_tensor)

#需要注意的是，t.tensor()总是会进行数据拷贝，新tensor和原来的数据不再共享内存。所以如果你想共享内存的话，建议使用torch.from_numpy()或者tensor.detach()来新建一个tensor, 二者共享内存。
new_tensor = old_tensor.detach()
new_tensor[0] = 1111
print(old_tensor, new_tensor)
'''
x = t.ones(2, 2, requires_grad=True)
print(x)
y=x.sum()
print(y)
y.backward() # 反向传播,计算梯度
print(x.grad)
y.backward() # 反向传播,计算梯度
print(x.grad)
#注意：grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以反向传播之前需把梯度清零。
x.grad.data.zero_()
y.backward() # 反向传播,计算梯度
print(x.grad)