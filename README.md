YOLOv3_Keras:使用Keras实现YOLO v3目标检测。
================================

# 步骤：

### 1、下载yolov3.weights到项目目录下；

[yolov3.weights](https://link.jianshu.com/?t=https%3A%2F%2Fpjreddie.com%2Fmedia%2Ffiles%2Fyolov3.weights)<br />  

### 2、运行如下命令进行权值文件转换；

python yad2k.py cfg\yolo.cfg yolov3.weights data\yolo.h5

### 3、将测试图片放到images/test目录下，运行demo.py，即可在images/res下看到运行结果。

### 测试图片：

![github](https://github.com/MrJoeyM/YOLOv3_Keras/blob/master/images/test/00071.png)  

### 运行结果：

![github](https://github.com/MrJoeyM/YOLOv3_Keras/blob/master/images/res/00071.png)  
