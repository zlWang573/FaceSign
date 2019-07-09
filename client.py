# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import cv2
import pyttsx3
from PIL import Image
import json
import numpy as np
import os
import sys
import socket
import time
from tkinter import *

HOST = 'localhost'
PORT = 9500
BUFSIZ = 10000
ADDR = (HOST,PORT)

cliSockfd = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
facexml = r"haarcascade_frontalface_default.xml"

def Say(string):
    audio = pyttsx3.init()
    audio.say(string)
    audio.runAndWait()

def main():
    SIGN = ''
    MODE = 'None'
    INFO = 'push \'i\': sign in, \'o\': sign back, \'q\': exit'
    try:
        cliSockfd.connect(ADDR)
        print('connect to server')
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(facexml) # 加载人脸特征库

        recoginzer = cv2.face.LBPHFaceRecognizer_create()
        if (os.path.exists('./data/trainer.yml')): # 判断是否存在目录
            recoginzer.read(r'./data/trainer.yml')  #加载分类器
        else:
            print("no trained information! please contact adminstrator!")
            return

        display = 0
        display_info = ''

        while(True):
            ret, frame = cap.read() # 读取一帧的图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 转灰
            cv2.putText(frame, INFO, (0 ,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),1)
            cv2.putText(frame, 'Mode: '+MODE, (0 ,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            if display > 0:
                cv2.putText(frame, display_info, (0 ,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)
                display -= 1

            faces = face_cascade.detectMultiScale(gray,
                scaleFactor = 1.15,
                minNeighbors = 5,
                minSize = (5, 5)
                ) # 检测人脸

            Area=0.0
            test_img=[]
            xx,yy,ww,hh = 0,0,0,0
            for(x, y, w, h) in faces:
                if (w*h > Area and float(w + h) > 300.0): #只挑选最大的人脸，且有足够大小
                    test_img=gray[y:y+h, x:x+w]
                    xx,yy,ww,hh = x,y,w,h
                    Area = w*h
            flag = False
            if len(test_img) != 0 and MODE != 'None':
                cv2.rectangle(frame, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2) # 用矩形圈出人脸
                idnum, confidence = recoginzer.predict(gray[y:y+h, x:x+w])
                confidence = 100 - confidence
                if confidence > 50: #置信度高于50再确认
                    confidence = "{0}%".format(round(confidence))
                    send_msg = str(idnum) + ' ' + SIGN;
                    cliSockfd.send(send_msg.encode())
                    recv_msg = cliSockfd.recv(BUFSIZ)
                    recv_msg = str(recv_msg)
                    recv_msg = recv_msg[2:-1]
                    list = recv_msg.split(' ')
                    if len(list) < 3:
                        continue
                    idnum = list[2]
                    if list[1] == '1':
                        flag = True
                    else:
                        idnum += ':already sign ' + SIGN
                    cv2.putText(frame, str(idnum), (xx+5, yy-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)
                    cv2.putText(frame, 'id: '+list[0], (xx+5, yy+hh+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)

            cv2.imshow('video', frame)

            if flag:
                print(list[2]+', sign '+SIGN+' sucess!')
                Say('hello,'+list[2]+', sign '+SIGN+' sucess!')
                display_info = 'hello,'+list[2]+', sign '+SIGN+' sucess! id: ' + list[0]
                display = 40 #延迟显示信息系数
            op = cv2.waitKey(1)
            if op & 0xFF == ord('q'):
                break
            if op & 0xFF == ord('i'):
                SIGN = 'in'
                MODE = 'Sign In'
            if op & 0xFF == ord('o'):
                SIGN = 'out'
                MODE = 'Sign Out'
            if op & 0xFF == ord('n'):
                SIGN = ''
                MODE = 'None'
        cliSockfd.send(b'close')
        cap.release() # 释放摄像头
        cv2.destroyAllWindows()

    except socket.error:
        print ('error: connect fail')
    finally:
        cliSockfd.close()

if __name__ == '__main__':
    main()
