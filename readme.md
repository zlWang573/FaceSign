# FaceSign 人脸识别签到系统：

### 简介：
    人脸识别签到系统，可以用摄像头识别人脸并辨别身份，将签到/签退信息保存在文件中
    CS架构:
    server负责采集人脸，训练，开启服务
    client负责签到/签退界面

### 环境：
    开发时使用deepin(linux)操作系统，python3.5
    所需环境包括但不限于：
    python3
    opencv-contrib-python
    pyttsx3
    numpy
    matplotlib
    espeak

### 如何使用：
    先运行server.py
    在server.py中可以采集人脸，训练，开启服务
    当server.py开启服务后可以运行client.py识别人脸
    不同的人要保证id不同，系统靠id识别人的身份
    如果重复录入人脸不会覆盖之前的，而是追加
    需要删除之前的人脸可以手动去删除
    暂时没有删除人的操作，如果想的话可以手动删除人脸文件夹，和name.json中的元组

### 文件：
    server.py 服务器程序
    client.py 客户端程序
    haarcascade_frontalface_default.xml opencv人脸分类器
    data 保存数据的文件夹， 运行server.py也会自动创建
    每天的签到签退信息将会保存在./data/SignInfo文件夹中
    ./data/name.json保存了姓名与id的对应
    训练好的分类器将会保存为./data/trainer.yml
    人脸图片保存在./data/FaceData文件夹中
    client.py运行需要haarcascade_frontalface_default.xml， trainer.yml（保证与server同步）
    server.py运行也需要haarcascade_frontalface_default.xml

### 其他：
    本来想用mysql数据库实现数据保存及通信，搞了半天没装下mysql，遂暂时放弃，用json文件保存信息
    client.py有语音功能，但是很难听，并且导致图像显示延迟，不想要可以删除Say函数
    不要让json文件为空，为空的话直接删除