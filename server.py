# -*- coding: utf-8 -*-

import os
import cv2
from PIL import Image
import json
import numpy as np

from socket import *
import time
import sys
import os

#json 读取写入
def json_write(json_str, JsonPath=r"./data/name.json"):
    with open(JsonPath,"w") as f:
        json.dump(json_str, f)

def json_read(JsonPath=r"./data/name.json"):
    with open(JsonPath,"r") as f:
        json_str = json.load(fp=f)
        return json_str
#写入签到信息
def write_sign_in(id):
    Date = time.strftime("%Y-%m-%d", time.localtime())
    Time = time.strftime("%H:%M:%S", time.localtime())
    sign_info = {}
    if (os.path.exists('./data/SignInfo/'+Date+'.in.json')): # 判断是否存在目录
        sign_info = json_read('./data/SignInfo/'+Date+'.in.json')
    else:
        open('./data/SignInfo/'+Date+'.in.json','w')
    it = iter(sign_info)
    flag = True
    for x in it:
        if x == id:
            flag = False
            break
    if not flag: #已签到
        return False
    sign_info[id] = Time
    json_write(sign_info, './data/SignInfo/'+Date+'.in.json')
    return True
#写入签退信息
def write_sign_out(id):
    Date = time.strftime("%Y-%m-%d", time.localtime())
    Time = time.strftime("%H:%M:%S", time.localtime())
    sign_info = {}
    if (os.path.exists('./data/SignInfo/'+Date+'.out.json')): # 判断是否存在目录
        sign_info = json_read('./data/SignInfo/'+Date+'.out.json')
    else:
        open('./data/SignInfo/'+Date+'.out.json','w')
    it = iter(sign_info)
    flag = True
    for x in it:
        if x == id:
            flag = False
            break
    if not flag: #已签退
        return False
    sign_info[id] = Time
    json_write(sign_info, './data/SignInfo/'+Date+'.out.json')
    return True

facexml = r"haarcascade_frontalface_default.xml"

#获取特征图
def GetFeature():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(facexml) # 加载人脸特征库

    name2id = {}

    if (os.path.exists('./data/name.json')): # 判断是否存在文件
        name2id=json_read()
    else:
        open(r"./data/name.json","w")

    face_name = input('\n enter user name:')
    face_id = input('\n enter user id:')

    # id 和 name 一一对应
    name2id_flag = True
    it = iter(name2id)
    for x in it:
        if name2id[x] == face_id:
            name2id_flag = False
            break

    if name2id_flag:
        name2id[face_name]= face_id#添加映射

    json_write(name2id) #写入json

    if (os.path.exists('./data/FaceData/'+str(face_id))): # 判断是否存在目录
        pass
    else:
        os.mkdir('./data/FaceData/'+str(face_id)+'/')

    count = (len(os.listdir('./data/FaceData/'+face_id)))
    tcount = 0
    while(True):
        ret, frame = cap.read() # 读取一帧的图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 转灰

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

        if len(test_img) != 0:
            cv2.rectangle(frame, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2) # 用矩形圈出人脸
            cv2.imwrite('./data/FaceData/'+str(face_id)+'/User.'+str(face_id)+'.'+str(count)+'.jpg', test_img) #保存最大特征图
            count = count + 1
            tcount = tcount + 1

        cv2.imshow('video', frame)
        print(tcount)
        if tcount == 100:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release() # 释放摄像头
    cv2.destroyAllWindows()

#获取特征和id
def GetFeatureAndLabels(path):
    Features = [f for f in os.listdir(path)]
    imgPaths = []
    FaceSamples = []
    ids = []
    names = []

    name2id = json_read()
    for feature in Features:
        test_id = -1
        it = iter(name2id)
        for x in it:
            test_id += 1
            if name2id[x] == feature:
                break

        test_path = os.path.join(path, feature)
        imgPaths.append([os.path.join(test_path, f) for f in os.listdir(test_path)])
        for imgpath in os.listdir(test_path):

            PIL_img = Image.open(os.path.join(test_path,imgpath)).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            FaceSamples.append(img_numpy)
            names.append(feature)
            ids.append(int(test_id))

    return FaceSamples, ids, names

# 训练
def FaceTrain():
    print ("Train Face .... ")

    faces,ids,names=GetFeatureAndLabels('./data/FaceData')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))   #开始训练
    print(recognizer)

    recognizer.write(r'./data/trainer.yml')

    print("{0} faces is trained".format(len(np.unique(ids))))

def FaceDetection():

    HOST = 'localhost'
    PORT = 9500
    BUFSIZ = 10000
    ADDR = (HOST,PORT)

    sockfd = socket(AF_INET,SOCK_STREAM)
    sockfd.bind(ADDR)
    sockfd.listen(5)

    name2id=json_read()

    print('waiting for connection...')
    cliSockfd,addr = sockfd.accept()
    print('...connected sucess')

    while True:
        msg = cliSockfd.recv(BUFSIZ)
        msg = str(msg)
        msg = msg[2:-1]
        if msg == 'close':
            print('client closed')
            break
        list = str(msg).split(' ')
        if len(list) < 2:
            continue
        #将list[0]转化为名字
        cnt = int(list[0])
        it = iter(name2id)
        for x in it:
            if cnt == 0:
                list[0] = x
                break
            cnt -= 1

        if list[1] == 'in':
            if write_sign_in(name2id[list[0]]):
                send_msg = name2id[list[0]] + ' 1 ' + list[0]
                print(name2id[list[0]]+' sign in...')
            else:
                send_msg = name2id[list[0]] + ' 0 ' + list[0]
        elif list[1] == 'out':
            if write_sign_out(name2id[list[0]]):
                send_msg = name2id[list[0]] + ' 1 ' + list[0]
                print(name2id[list[0]]+' sign out...')
            else:
                send_msg = name2id[list[0]] + ' 0 ' + list[0]
        cliSockfd.send(send_msg.encode())
    cliSockfd.close()

def main():
    while (True):

        order = input("\n please input a order(0(open server),1(adminstrator),q(exit))\n")
        if order == "q":
            break

        elif order == "0":
            if (os.path.exists('./data//trainer.yml')):# 判断是否存在文件
                if (os.path.exists('./data//name.json')):# 判断是否存在文件
                    FaceDetection()
                else:
                    print("no name and id information!")
            else:
                print("no trained information!")

        elif order == "1":
            mod = input("\n please input a mod, t(training), g(get face)")
            if mod == "t":
                if (os.path.exists('./data/') & (int)(len([os.listdir('./data/')])) >= 1):# 判断是否存在目录
                    if (os.path.exists('./data/name.json')):# 判断是否存在文件
                        FaceTrain()
                    else:
                        print("no name and id information!")
                else:
                    print("no face information!")
            elif mod == "g":
                GetFeature()
                FaceTrain()

def init():
    if (os.path.exists('./data/')): # 判断是否存在目录
        pass
    else:
        os.mkdir('./data/')
    if (os.path.exists('./data/FaceData/')): # 判断是否存在目录
        pass
    else:
        os.mkdir('./data/FaceData/')
    if (os.path.exists('./data/SignInfo')): # 判断是否存在目录
        pass
    else:
        os.mkdir('./data/SignInfo')

if __name__ == '__main__':
    init()
    main()
