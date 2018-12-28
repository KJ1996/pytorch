#!/usr/bin/env python
# encoding: utf-8
'''
1、读取指定目录下的所有文件
2、读取文件，正则匹配出需要的内容，获取文件名
3、打开此文件(可以选择打开可以选择复制到别的地方去)
'''
import os.path
import imageio

import pylab

import skimage
import numpy as np
min = 200
video_list=[]
def readfile(filepath):
    vid = imageio.get_reader(filepath, 'ffmpeg')
    print(len(vid))
    return len(vid)
# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath,classes):
    pathDir = os.listdir(filepath)
    #print(len(pathDir))
    global min
    global video_list
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        if os.path.isfile(child):
            video_list.append(child)
            temp = readfile(child)#读取视频文件，并返回帧长
            if (min > temp):
                min =temp
            for i in range(len(classes)):

                if(child.find(classes[i])>=0):
                    
                    print(i)


            continue
        eachFile(child,classes)
        







if __name__ == "__main__":
    filenames = '/home/hyj/UCF-101'  # refer root dir
    pathDir = os.listdir(filenames)
    classes = pathDir
    eachFile(filenames,classes)
    print(video_list=[])
    print("total:",len(video_list))
    print("min is",min)

'''
#视频的绝对路径
filename = '/home/hyj/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'
#可以选择解码工具
vid = imageio.get_reader(filename,  'ffmpeg')
for num,im in enumerate(vid):
    #image的类型是mageio.core.util.Image可用下面这一注释行转换为arrary
    print (im.mean())
    image = skimage.img_as_float(im).astype(np.float64)
    fig = pylab.figure()
    fig.suptitle('image #{}'.format(num), fontsize=20)
    pylab.show()
'''

