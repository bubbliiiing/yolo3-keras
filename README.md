## YOLOV3：You Only Look Once目标检测模型在Keras当中的实现
---

## 目录
1. [仓库更新 Top News](#仓库更新)
2. [相关仓库 Related code](#相关仓库)
3. [性能情况 Performance](#性能情况)
4. [所需环境 Environment](#所需环境)
5. [文件下载 Download](#文件下载)
6. [训练步骤 How2train](#训练步骤)
7. [预测步骤 How2predict](#预测步骤)
8. [评估步骤 How2eval](#评估步骤)
9. [参考资料 Reference](#Reference)

## Top News
**`2022-04`**:**支持多GPU训练，新增各个种类目标数量计算，新增heatmap。**  

**`2022-03`**:**进行了大幅度的更新，修改了loss组成，使得分类、目标、回归loss的比例合适、支持step、cos学习率下降法、支持adam、sgd优化器选择、支持学习率根据batch_size自适应调整、新增图片裁剪。**  
BiliBili视频中的原仓库地址为：https://github.com/bubbliiiing/yolo3-keras/tree/bilibili

**`2021-10`**:**进行了大幅度的更新，增加了大量注释、增加了大量可调整参数、对代码的组成模块进行修改、增加fps、视频预测、批量预测等功能。**   

## 相关仓库
| 模型 | 路径 |
| :----- | :----- |
YoloV3 | https://github.com/bubbliiiing/yolo3-keras  
Efficientnet-Yolo3 | https://github.com/bubbliiiing/efficientnet-yolo3-keras  
YoloV4 | https://github.com/bubbliiiing/yolov4-keras
YoloV4-tiny | https://github.com/bubbliiiing/yolov4-tiny-keras
Mobilenet-Yolov4 | https://github.com/bubbliiiing/mobilenet-yolov4-keras
YoloV5-V5.0 | https://github.com/bubbliiiing/yolov5-keras
YoloV5-V6.1 | https://github.com/bubbliiiing/yolov5-v6.1-keras
YoloX | https://github.com/bubbliiiing/yolox-keras

## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| COCO-Train2017 | [yolo_weights.h5](https://github.com/bubbliiiing/yolo3-keras/releases/download/v1.0/yolo_weights.h5) | COCO-Val2017 | 416x416 | 38.1 | 66.8

## 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5  

## 文件下载
训练所需的yolo_weights.h5可以在Release里面下载。  
https://github.com/bubbliiiing/yolo3-keras/releases  
也可以去百度网盘下载  

链接: https://pan.baidu.com/s/1N5TyQTWDeVsN5h9v-ucpbg    
提取码: q42w   

VOC数据集下载地址如下，里面已经包括了训练集、测试集、验证集（与测试集一样），无需再次划分：  
链接: https://pan.baidu.com/s/19Mw2u_df_nBzsC2lg20fQA    
提取码: j5ge    

## 训练步骤
### a、训练VOC07+12数据集
1. 数据集的准备   
**本文使用VOC格式进行训练，训练前需要下载好VOC07+12的数据集，解压后放在根目录**  

2. 数据集的处理   
修改voc_annotation.py里面的annotation_mode=2，运行voc_annotation.py生成根目录下的2007_train.txt和2007_val.txt。   

3. 开始网络训练   
train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。   

4. 训练结果预测   
训练结果预测需要用到两个文件，分别是yolo.py和predict.py。我们首先需要去yolo.py里面修改model_path以及classes_path，这两个参数必须要修改。   
**model_path指向训练好的权值文件，在logs文件夹里。   
classes_path指向检测类别所对应的txt。**   
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。   

### b、训练自己的数据集
1. 数据集的准备  
**本文使用VOC格式进行训练，训练前需要自己制作好数据集，**    
训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。   
训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。   

2. 数据集的处理  
在完成数据集的摆放之后，我们需要利用voc_annotation.py获得训练用的2007_train.txt和2007_val.txt。   
修改voc_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。   
训练自己的数据集时，可以自己建立一个cls_classes.txt，里面写自己所需要区分的类别。   
model_data/cls_classes.txt文件内容为：      
```python
cat
dog
...
```
修改voc_annotation.py中的classes_path，使其对应cls_classes.txt，并运行voc_annotation.py。  

3. 开始网络训练  
**训练的参数较多，均在train.py中，大家可以在下载库后仔细看注释，其中最重要的部分依然是train.py里的classes_path。**  
**classes_path用于指向检测类别所对应的txt，这个txt和voc_annotation.py里面的txt一样！训练自己的数据集必须要修改！**  
修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中。  

4. 训练结果预测  
训练结果预测需要用到两个文件，分别是yolo.py和predict.py。在yolo.py里面修改model_path以及classes_path。  
**model_path指向训练好的权值文件，在logs文件夹里。  
classes_path指向检测类别所对应的txt。**  
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。  

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载yolo_weights.pth，放入model_data，运行predict.py，输入  
```python
img/street.jpg
```
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/yolo_weights.h5',
    "classes_path"      : 'model_data/coco_classes.txt',
    #---------------------------------------------------------------------#
    #   anchors_path代表先验框对应的txt文件，一般不修改。
    #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    #---------------------------------------------------------------------#
    "anchors_path"      : 'model_data/yolo_anchors.txt',
    "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    #---------------------------------------------------------------------#
    #   输入图片的大小，必须为32的倍数。
    #---------------------------------------------------------------------#
    "input_shape"       : [416, 416],
    #---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来，置信度门限
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #---------------------------------------------------------------------#
    #   最大目标数量
    #---------------------------------------------------------------------#
    "max_boxes"         : 100,
    #---------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    #---------------------------------------------------------------------#
    "letterbox_image"   : False,
}
```
3. 运行predict.py，输入  
```python
img/street.jpg
```
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。  

## 评估步骤 
### a、评估VOC07+12的测试集
1. 本文使用VOC格式进行评估。VOC07+12已经划分好了测试集，无需利用voc_annotation.py生成ImageSets文件夹下的txt。
2. 在yolo.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
3. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

### b、评估自己的数据集
1. 本文使用VOC格式进行评估。  
2. 如果在训练前已经运行过voc_annotation.py文件，代码会自动将数据集划分成训练集、验证集和测试集。如果想要修改测试集的比例，可以修改voc_annotation.py文件下的trainval_percent。trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1。train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1。
3. 利用voc_annotation.py划分测试集后，前往get_map.py文件修改classes_path，classes_path用于指向检测类别所对应的txt，这个txt和训练时的txt一样。评估自己的数据集必须要修改。
4. 在yolo.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
5. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

## Reference
https://github.com/qqwweee/keras-yolo3  
https://github.com/eriklindernoren/PyTorch-YOLOv3   
https://github.com/BobLiu20/YOLOv3_PyTorch
