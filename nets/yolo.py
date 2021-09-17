from keras.layers import Concatenate, Input, Lambda, UpSampling2D
from keras.models import Model
from utils.utils import compose

from nets.darknet import DarknetConv2D, DarknetConv2D_BN_Leaky, darknet_body
from nets.yolo_training import yolo_loss

#---------------------------------------------------#
#   特征层->最后的输出
#---------------------------------------------------#
def make_five_conv(x, num_filters):
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    return x

def make_yolo_head(x, num_filters, out_filters):
    y = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    y = DarknetConv2D(out_filters, (1,1))(y)
    return y

#---------------------------------------------------#
#   FPN网络的构建，并且获得预测结果
#---------------------------------------------------#
def yolo_body(input_shape, anchors_mask, num_classes):
    inputs      = Input(input_shape)
    #---------------------------------------------------#   
    #   生成darknet53的主干模型
    #   获得三个有效特征层，他们的shape分别是：
    #   C3 为 52,52,256
    #   C4 为 26,26,512
    #   C5 为 13,13,1024
    #---------------------------------------------------#
    C3, C4, C5  = darknet_body(inputs)

    #---------------------------------------------------#
    #   第一个特征层
    #   y1=(batch_size,13,13,3,85)
    #---------------------------------------------------#
    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    x   = make_five_conv(C5, 512)
    P5  = make_yolo_head(x, 512, len(anchors_mask[0]) * (num_classes+5))

    # 13,13,512 -> 13,13,256 -> 26,26,256
    x   = compose(DarknetConv2D_BN_Leaky(256, (1,1)), UpSampling2D(2))(x)

    # 26,26,256 + 26,26,512 -> 26,26,768
    x   = Concatenate()([x, C4])
    #---------------------------------------------------#
    #   第二个特征层
    #   y2=(batch_size,26,26,3,85)
    #---------------------------------------------------#
    # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    x   = make_five_conv(x, 256)
    P4  = make_yolo_head(x, 256, len(anchors_mask[1]) * (num_classes+5))

    # 26,26,256 -> 26,26,128 -> 52,52,128
    x   = compose(DarknetConv2D_BN_Leaky(128, (1,1)), UpSampling2D(2))(x)
    # 52,52,128 + 52,52,256 -> 52,52,384
    x   = Concatenate()([x, C3])
    #---------------------------------------------------#
    #   第三个特征层
    #   y3=(batch_size,52,52,3,85)
    #---------------------------------------------------#
    # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
    x   = make_five_conv(x, 128)
    P3  = make_yolo_head(x, 128, len(anchors_mask[2]) * (num_classes+5))
    return Model(inputs, [P5, P4, P3])


def get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask):
    y_true = [Input(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], \
                                len(anchors_mask[l]), num_classes + 5)) for l in range(len(anchors_mask))]
    model_loss  = Lambda(
        yolo_loss, 
        output_shape    = (1, ), 
        name            = 'yolo_loss', 
        arguments       = {'input_shape' : input_shape, 'anchors' : anchors, 'anchors_mask' : anchors_mask, 'num_classes' : num_classes, 'ignore_thresh': 0.7}
    )([*model_body.output, *y_true])
    model       = Model([model_body.input, *y_true], model_loss)
    return model
