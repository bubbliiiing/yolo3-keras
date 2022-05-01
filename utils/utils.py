from functools import reduce

import numpy as np
from PIL import Image


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   获得先验框
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
    
#-------------------------------------------------------------------------------------------------------------------------------#
#   From https://github.com/ckyrkou/Keras_FLOP_Estimator 
#   Fix lots of bugs
#-------------------------------------------------------------------------------------------------------------------------------#
def net_flops(model, table=False, print_result=True):
    if (table == True):
        print("\n")
        print('%25s | %16s | %16s | %16s | %16s | %6s | %6s' % (
            'Layer Name', 'Input Shape', 'Output Shape', 'Kernel Size', 'Filters', 'Strides', 'FLOPS'))
        print('=' * 120)
        
    #---------------------------------------------------#
    #   总的FLOPs
    #---------------------------------------------------#
    t_flops = 0
    factor  = 1e9

    for l in model.layers:
        try:
            #--------------------------------------#
            #   所需参数的初始化定义
            #--------------------------------------#
            o_shape, i_shape, strides, ks, filters = ('', '', ''), ('', '', ''), (1, 1), (0, 0), 0
            flops   = 0
            #--------------------------------------#
            #   获得层的名字
            #--------------------------------------#
            name    = l.name
            
            if ('InputLayer' in str(l)):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]
                
            #--------------------------------------#
            #   Reshape层
            #--------------------------------------#
            elif ('Reshape' in str(l)):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]

            #--------------------------------------#
            #   填充层
            #--------------------------------------#
            elif ('Padding' in str(l)):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]

            #--------------------------------------#
            #   平铺层
            #--------------------------------------#
            elif ('Flatten' in str(l)):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]
                
            #--------------------------------------#
            #   激活函数层
            #--------------------------------------#
            elif 'Activation' in str(l):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]
                
            #--------------------------------------#
            #   LeakyReLU
            #--------------------------------------#
            elif 'LeakyReLU' in str(l):
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    flops   += i_shape[0] * i_shape[1] * i_shape[2]
                    
            #--------------------------------------#
            #   池化层
            #--------------------------------------#
            elif 'MaxPooling' in str(l):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]
                    
            #--------------------------------------#
            #   池化层
            #--------------------------------------#
            elif ('AveragePooling' in str(l) and 'Global' not in str(l)):
                strides = l.strides
                ks      = l.pool_size
                
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    flops   += o_shape[0] * o_shape[1] * o_shape[2]

            #--------------------------------------#
            #   全局池化层
            #--------------------------------------#
            elif ('AveragePooling' in str(l) and 'Global' in str(l)):
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    flops += (i_shape[0] * i_shape[1] + 1) * i_shape[2]
                
            #--------------------------------------#
            #   标准化层
            #--------------------------------------#
            elif ('BatchNormalization' in str(l)):
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]

                    temp_flops = 1
                    for i in range(len(i_shape)):
                        temp_flops *= i_shape[i]
                    temp_flops *= 2
                    
                    flops += temp_flops
                
            #--------------------------------------#
            #   全连接层
            #--------------------------------------#
            elif ('Dense' in str(l)):
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                
                    temp_flops = 1
                    for i in range(len(o_shape)):
                        temp_flops *= o_shape[i]
                        
                    if (i_shape[-1] == None):
                        temp_flops = temp_flops * o_shape[-1]
                    else:
                        temp_flops = temp_flops * i_shape[-1]
                    flops += temp_flops

            #--------------------------------------#
            #   普通卷积层
            #--------------------------------------#
            elif ('Conv2D' in str(l) and 'DepthwiseConv2D' not in str(l) and 'SeparableConv2D' not in str(l)):
                strides = l.strides
                ks      = l.kernel_size
                filters = l.filters
                bias    = 1 if l.use_bias else 0
                
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    if (filters == None):
                        filters = i_shape[2]
                    flops += filters * o_shape[0] * o_shape[1] * (ks[0] * ks[1] * i_shape[2] + bias)

            #--------------------------------------#
            #   逐层卷积层
            #--------------------------------------#
            elif ('Conv2D' in str(l) and 'DepthwiseConv2D' in str(l) and 'SeparableConv2D' not in str(l)):
                strides = l.strides
                ks      = l.kernel_size
                filters = l.filters
                bias    = 1 if l.use_bias else 0
            
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    if (filters == None):
                        filters = i_shape[2]
                    flops += filters * o_shape[0] * o_shape[1] * (ks[0] * ks[1] + bias)
                
            #--------------------------------------#
            #   深度可分离卷积层
            #--------------------------------------#
            elif ('Conv2D' in str(l) and 'DepthwiseConv2D' not in str(l) and 'SeparableConv2D' in str(l)):
                strides = l.strides
                ks      = l.kernel_size
                filters = l.filters
                
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    if (filters == None):
                        filters = i_shape[2]
                    flops += i_shape[2] * o_shape[0] * o_shape[1] * (ks[0] * ks[1] + bias) + \
                             filters * o_shape[0] * o_shape[1] * (1 * 1 * i_shape[2] + bias)
            #--------------------------------------#
            #   模型中有模型时
            #--------------------------------------#
            elif 'Model' in str(l):
                flops = net_flops(l, print_result=False)
                
            t_flops += flops

            if (table == True):
                print('%25s | %16s | %16s | %16s | %16s | %6s | %5.4f' % (
                    name[:25], str(i_shape), str(o_shape), str(ks), str(filters), str(strides), flops))
                
        except:
            pass
    
    t_flops = t_flops * 2
    if print_result:
        show_flops = t_flops / factor
        print('Total GFLOPs: %.3fG' % (show_flops))
    return t_flops