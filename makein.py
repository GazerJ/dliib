#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 13:54:52 2022

@author: gazer
"""
import dlib
import numpy as np
import cv2
import os
import json
detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
imagePath = './'                                                                           #图像的目录
data = np.zeros((1,128))                                                                            #定义一个128维的空向量data
label = []        #定义空的list存放人脸的标签

for file in os.listdir(imagePath):
    try:                                                                  #开始一张一张索引目录中的图像
        if '.jpg' in file or '.png' in file:
            fileName = file
            labelName = file.split('_')[0]                                                              #获取标签名
            print('current image: ', file)
            print('current label: ', labelName)
            
            img = cv2.imread(imagePath + file)                                                          #使用opencv读取图像数据
            if img.shape[0]*img.shape[1] > 500000:                                                      #如果图太大的话需要压缩，这里像素的阈值可以自己设置
                img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            dets = detector(img, 1)                                                                     #使用检测算子检测人脸，返回的是所有的检测到的人脸区域
            for k, d in enumerate(dets):
                rec = dlib.rectangle(d.rect.left(),d.rect.top(),d.rect.right(),d.rect.bottom())
                shape = sp(img, rec)                                                                    #获取landmark
                face_descriptor = facerec.compute_face_descriptor(img, shape)                           #使用resNet获取128维的人脸特征向量
                faceArray = np.array(face_descriptor).reshape((1, 128))                                 #转换成numpy中的数据结构
                data = np.concatenate((data, faceArray))                                                #拼接到事先准备好的data当中去
                label.append(labelName)                                                                 #保存标签
                cv2.rectangle(img, (rec.left(), rec.top()), (rec.right(), rec.bottom()), (0, 255, 0), 2)       #显示人脸区域
            cv2.waitKey(2)
            cv2.imshow('image', img)
    except:
        pass

data = data[1:, :]                                                                                  #因为data的第一行是空的128维向量，所以实际存储的时候从第二行开始
np.savetxt('faceData.txt', data, fmt='%f')                                                          #保存人脸特征向量合成的矩阵到本地

labelFile=open('label.txt','w')                                      
json.dump(label, labelFile)                                                                         #使用json保存list到本地
labelFile.close()

cv2.destroyAllWindows()                                                                             #关闭所有的窗口