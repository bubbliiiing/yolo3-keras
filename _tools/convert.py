from operator import itemgetter

import numpy as np

from utils.utils import get_anchors, get_classes
from yolo import YOLO


class Yolo4(YOLO):
    def load_yolo(self):
        self.yolo4_model = self.yolo_model

        print('Loading weights.')
        weights_file            = open("model_data/darknet_weights/yolov3.weights", 'rb')
        major, minor, revision  = np.ndarray(shape=(3, ), dtype='int32', buffer=weights_file.read(12))
        if (major*10+minor)>=2 and major<1000 and minor<1000:
            seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
        else:
            seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
        print('Weights Header: ', major, minor, revision, seen)

        convs_to_load   = []
        bns_to_load     = []
        for i in range(len(self.yolo4_model.layers)):
            layer_name  = self.yolo4_model.layers[i].name
            if layer_name.startswith('conv2d_'):
                convs_to_load.append((int(layer_name[7:]), i))
            if layer_name.startswith('batch_normalization_'):
                bns_to_load.append((int(layer_name[20:]), i))

        convs_sorted    = sorted(convs_to_load, key=itemgetter(0))
        bns_sorted      = sorted(bns_to_load, key=itemgetter(0))

        bn_index = 0
        for i in range(len(convs_sorted)):
            print('Converting ', i)
            if i == 58 or i == 66 or i == 74:
                weights_shape   = self.yolo4_model.layers[convs_sorted[i][1]].get_weights()[0].shape
                bias_shape      = self.yolo4_model.layers[convs_sorted[i][1]].get_weights()[0].shape[3]
                filters         = bias_shape
                size            = weights_shape[0]
                darknet_w_shape = (filters, weights_shape[2], size, size)
                weights_size    = np.product(weights_shape)

                conv_bias = np.ndarray(
                    shape   = (filters, ),
                    dtype   = 'float32',
                    buffer  = weights_file.read(filters * 4)
                )
                conv_weights = np.ndarray(
                    shape   = darknet_w_shape,
                    dtype   = 'float32',
                    buffer  = weights_file.read(weights_size * 4)
                )
                conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
                self.yolo4_model.layers[convs_sorted[i][1]].set_weights([conv_weights, conv_bias])
            else:
                weights_shape   = self.yolo4_model.layers[convs_sorted[i][1]].get_weights()[0].shape
                size            = weights_shape[0]

                bn_shape        = self.yolo4_model.layers[bns_sorted[bn_index][1]].get_weights()[0].shape
                filters         = bn_shape[0]
                darknet_w_shape = (filters, weights_shape[2], size, size)
                weights_size    = np.product(weights_shape)

                conv_bias   = np.ndarray(
                    shape   = (filters, ),
                    dtype   = 'float32',
                    buffer  = weights_file.read(filters * 4))
                bn_weights  = np.ndarray(
                    shape   = (3, filters),
                    dtype   = 'float32',
                    buffer  = weights_file.read(filters * 12))

                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,  # shift beta
                    bn_weights[1],  # running mean
                    bn_weights[2]  # running var
                ]
                self.yolo4_model.layers[bns_sorted[bn_index][1]].set_weights(bn_weight_list)

                conv_weights = np.ndarray(
                    shape   = darknet_w_shape,
                    dtype   = 'float32',
                    buffer  = weights_file.read(weights_size * 4)
                )
                conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
                self.yolo4_model.layers[convs_sorted[i][1]].set_weights([conv_weights])

                bn_index += 1
        weights_file.close()
        self.yolo4_model.save_weights("model_data/yolov3.h5")

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors     = get_anchors(self.anchors_path)

        self.generate()
        self.load_yolo()

if __name__ == '__main__':
    model_image_size = (608, 608)

    yolo4_model = Yolo4()

    yolo4_model.close_session()
