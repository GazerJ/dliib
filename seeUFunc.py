#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:01:01 2022

@author: gazer
"""

import cv2
import time
import dlib

class capter:
    count=0
    left=0
    top=0
    right=0
    bottom=0
    lasttime=time.time()
    FPS=0
    N=2
    def __init__(self):
        self.cap = cv2.VideoCapture (0)#打开本地摄像头# 0:本地摄像头 ，文件路径则为读文件/
        print(self.cap.isOpened()) #检测是否 打开
        self.cap.set(3,480)
        self.cap.set(4,640)#设置分辨率
        self.detector=dlib.get_frontal_face_detector()# Dlib库提供的检测人脸的库函数
        self.go()
    def go(self):
        count=0
        while self.cap.isOpened():
            ret,frame = self.cap.read() #获取数据帧
            count+=1
            if count==self.N:
                count=0
                self.FPS=self.getFPS()
                self.getFace(frame)
            cv2.rectangle(frame,(self.left,self.top),(self.right,self.bottom),(222,123,123))
            cv2.putText(frame, 'FPS: '+str(self.FPS//1.0), (100,100), cv2.FONT_HERSHEY_PLAIN, 5,(0,255,255), 5)
            cv2.imshow("frame",frame)
            a=cv2.waitKey(1)
            if(a==113):
                self.close()
                break;
    def getFPS(self):
        FPS=self.N/(time.time()-self.lasttime)
        self.lasttime=time.time()
        return FPS
    def getFace(self,frame):
        faces=self.detector(frame,0)
        for index,face in enumerate(faces):
            self.left=face.left()
            self.top=face.top()
            self.right=face.right()
            self.bottom=face.bottom()
    def close(self):
        self.cap.release()# 
        cv2.destroyAllWindows()
capter()