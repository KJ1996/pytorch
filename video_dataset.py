import os
import pandas as pd
import torch

"""
torch.utils.data.Dataset 是一个表示数据集的抽象类.
你自己的数据集一般应该继承``Dataset``, 并且重写下面的方法:
    1. __len__ 使用``len(dataset)`` 可以返回数据集的大小
    2. __getitem__ 支持索引, 以便于使用 dataset[i] 可以 获取第i个样本(0索引)
"""
from torch.utils.data import Dataset
import imageio
import random
import resnet_3d
import pylab
import skimage
#import cv2
import torch.nn.functional as F
import numpy as np
from itertools import combinations

"""
torch.utils.data中的DataLoader提供为Dataset类对象提供了:
    1.批量读取数据
    2.打乱数据顺序
    3.使用multiprocessing并行加载数据

    DataLoader中的一个参数collate_fn：可以使用它来指定如何精确地读取一批样本，
     merges a list of samples to form a mini-batch.
    然而，默认情况下collate_fn在大部分情况下都表现很好
"""
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from skimage import io, transform
import tranfroms
from tranfroms import ToTensor
from tranfroms import Normalize
import numpy as np
'''
读取根目录下的所有文件，并将文件路径收集为一个list并返回
'''


transform = transforms.Compose([
    ToTensor(),
     # 归一化
                             ])
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

    return torch.LongTensor(np.array(triplets))

def triplet_hashing_loss_regu(embeddings, cls, margin):
    triplets = get_triplets(cls)

    if embeddings.is_cuda:
        triplets = triplets.cuda()

    ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
    an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)

    losses = F.relu(ap_distances - an_distances + margin)

    return losses.mean()
def getFile(filepath):
    pathDir = os.listdir(filepath)
    # print(len(pathDir))

    video_list=[]
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        child_list=os.listdir(child)
        for allfile in child_list:
            video_file_path=os.path.join('%s/%s' % (child, allfile))
            if(os.path.isfile(video_file_path)):
                video_list.append(video_file_path)


    return video_list
'''
根据文件名，确定标签
'''
def get_lable(filename,classes):
    for i in range(len(classes)):
        if (filename.find(classes[i]) >= 0):
            return int(i)

#获取打乱后的所有batch
def get_minibatches_idx( len, minibatch_size, shuffle=True):
    """
    :param n: len of data
    :param minibatch_size: minibatch size of data
    :param shuffle: shuffle the data
    :return: len of minibatches and minibatches
    """

    idx_list = np.arange(len, dtype="int32")

    # shuffle
    if shuffle:
        random.shuffle(idx_list)  # also use torch.randperm()

    # segment
    minibatches = []
    minibatch_start = 0
    for i in range(len // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    # processing the last batch
    if (minibatch_start !=len):
        minibatches.append(idx_list[minibatch_start:])

    return minibatches

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None,batch_size=1,shuffle=True):
        self.video_list = getFile(root_dir)
        self.root_dir = root_dir
        self.transform = transform
        self.classes=os.listdir(root_dir)
        self.batch_size=batch_size
        self.minibatches=get_minibatches_idx(len(self.video_list),shuffle=shuffle,minibatch_size=batch_size)
    def __len__(self):
        """
        继承 Dataset 类后,必须重写的一个方法
        返回数据集的大小
        :return:
        """
        return len(self.video_list)

    def __getitem__(self, idx):
        """
        继承 Dataset 类后,必须重写的一个方法
        返回第 idx个batch及相关信息
        :param idx:
        :return:
        """
        batch=self.minibatches[idx]
        batch_seq=np.zeros((self.batch_size,3, 16, 32, 32), dtype="float32")
        batch_lable=np.zeros((self.batch_size), dtype="int32")
        #print(batch)
        for index,i in enumerate(batch):

            filename=self.video_list[i]
            lable=get_lable(filename,self.classes)
            batch_lable[index] =lable
            vid = imageio.get_reader(filename, 'ffmpeg')

            L=len(vid)#视频总长
            step=int(L/16)#每次取样的步长
            #print(L,step)
            seq = np.zeros((3, 16, 32, 32), dtype="float32")#格式爲np保存視頻,均勻抽取16帧
            for num in range(16):

                    #fig = pylab.figure()
                    #fig.suptitle('image #{}'.format(num), fontsize=20)
                    #pylab.show(im)
                    s = vid.get_data((num)*step)
                    resize_img=  transforms.Compose([transforms.ToTensor(),
                    transforms.ToPILImage(),
                    transforms.RandomCrop((32,32)),
                    transforms.ToTensor()])

                    image = np.asarray(s).astype(np.float32)
                    new_image = resize_img(image)
                    im_np=new_image.numpy()
                    #print(im_np.shape)
                    #print(s)

                    if(num==16):
                        print(index,' is ok')
                    seq[:,num,:,:]=im_np[:,:,:]
            batch_seq[index, :, :, :, :] = seq[:, :, :, :]

        if self.transform:#轉換爲tensor
            vid_tensor = self.transform(batch_seq)
            lable_tensor=self.transform(batch_lable)
            sample = {'video': vid_tensor, 'lable': lable_tensor}
            return sample
        else:
            sample = {'video': batch_seq, 'lable': batch_lable}
            return sample


path1='/home/hyj/UCF-101'
path2='/home/disk3/huyujie/data/UCF-101'

test=VideoDataset(path2,batch_size=128,transform=transform)
#print(test[10]['video'])
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
net=resnet_3d.resnet18(sample_size=32,sample_duration=16,num_classes=100)
net.to(device)
print(net)

from torch import optim
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
if __name__ == '__main__':
    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(test, 0):
        # 输入数据
            inputs = data['video']
            labels=data['lable']
            #print(labels.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)

        # 梯度清零
            optimizer.zero_grad()

        # forward + backward
            outputs = net(inputs)  # forward
            hash_out = torch.sigmoid(outputs)
            loss = triplet_hashing_loss_regu(hash_out, labels, 2)
            loss.backward()

        # 更新参数
            optimizer.step()

        # 打印log信息
        # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
            running_loss += loss.item()
            if i % 10 == 9:  # 每50个batch打印一下训练状态
                print('[%d, %5d] loss: %.3f' \
                    % (epoch + 1, i/10 + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')

# 保存模型
    #torch.save(net.state_dict(), 'net.pth')





