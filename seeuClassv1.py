#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:03:12 2022

@author: gazer
"""


import cv2
import time
import dlib
import json
import numpy as np
class capter:
    COUNT,left,top,right,bottom,FPS=0,0,0,0,0,0
    lasttime=time.time()
    N=10
    def __init__(self):
        labelFile=open('label.txt','r')
        self.label = json.load(labelFile)                                                   #载入本地人脸库的标签
        labelFile.close()
        self.data = np.loadtxt('faceData.txt',dtype=float)                                  #载入本地人脸特征向量

        self.cap = cv2.VideoCapture (0)#打开本地摄像头# 0:本地摄像头 ，文件路径则为读文件/
        print(self.cap.isOpened()) #检测是否 打开
        self.cap.set(3,480)
        self.cap.set(4,640)#设置分辨率
        self.detector = dlib.get_frontal_face_detector()# Dlib库提供的检测人脸的库函数
        self.predictor = dlib.shape_predictor("/home/gazer/dlib_shape_predictor_68_face_landmarks/shape_predictor_68_face_landmarks.dat")
        self.facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat') 
        self.go()
    def go(self):
        
        while self.cap.isOpened():
            ret,frame = self.cap.read() #获取数据帧
            
            self.findFace(frame)
            self.recoFace(frame)
            self.drewFace(frame)
            self.count()
            self.wait()
            
    def findFace(self,frame):
        if self.COUNT!=self.N:
            return 0
        self.getFPS()
        self.getFace(frame)
    def drewFace(self,frame):

        cv2.rectangle(frame,(self.left,self.top),(self.right,self.bottom),(222,123,123))
        cv2.putText(frame, 'FPS: '+str(self.FPS//1.0), (100,100), cv2.FONT_HERSHEY_PLAIN, 5,(0,255,255), 5)
        cv2.imshow("frame",frame)
    def recoFace(self,frame):
        if self.COUNT!=self.N:
            return 0
        try:
            face_descriptor = self.facerec.compute_face_descriptor(frame, self.shape)
            class_pre = self.findNearestClassForImage(face_descriptor, self.label)
        except:
            pass
        print(class_pre)
        cv2.rectangle(frame, (self.left, self.top+10), (self.right, self.bottom), (0, 255, 0), 2)
        cv2.putText(frame, class_pre , (self.left,self.top), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        return 0
    
    def count(self):
        if self.COUNT==self.N:
            self.COUNT=0
        else:
            self.COUNT+=1
    def wait(self):
        
        a=cv2.waitKey(1)
        if(a==113):
            self.close()
     
    def findNearestClassForImage(self,face_descriptor, faceLabel):
        threshold=0.5
        temp =  face_descriptor - self.data
        e = np.linalg.norm(temp,axis=1,keepdims=True)
        min_distance = e.min() 
        print('distance: ', min_distance)
        if min_distance > threshold:
            return 'other'
        index = np.argmin(e)
        return faceLabel[index] 
    def getFPS(self):
        self.FPS=self.N/(time.time()-self.lasttime)
        self.lasttime=time.time()
        return 0
    def getFace(self,frame):
        faces=self.detector(frame,1)
        print(1111)
        for index,face in enumerate(faces):
            self.left,self.top,self.right,self.bottom=face.left(),face.top(),face.right(),face.bottom()
            self.shape = self.predictor(frame, face)# 68点数据
            print(2222)
            for i in range(68):
                cv2.circle(frame, (self.shape.part(i).x, self.shape.part(i).y), 2, (0, 255, 0), -1, 8)
    def close(self):
        self.cap.release()# 
        cv2.destroyAllWindows()
capter()