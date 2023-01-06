from tensorflow.keras.layers import (
    Conv2D,
    Add,
    ZeroPadding2D,
    UpSampling2D,
    Concatenate,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Flatten,
    Softmax,
    Reshape,
    Input,
    ReLU
)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape



from yolo3.models.layers import (
    compose,
    DarknetConv2D,
    DarknetConv2D_BN,
    DarknetConv2D_BN_Leaky,
    DarknetConv2D_BN_custom_Leaky,
    Depthwise_Separable_Conv2D_BN_Leaky,
    Darknet_Depthwise_Separable_Conv2D_BN_Leaky,
)

# from yolo3.models.layers import make_last_layers, make_depthwise_separable_last_layers, make_spp_last_layers
#from yolo3.models.layers import (
#    yolo3_predictions,
#    yolo3lite_predictions,
#    tiny_yolo3_predictions,
#    tiny_yolo3lite_predictions,
#)

from yolo3.models.yolo3_darknet import tiny_yolo3_body


def yolox_head(inputs, num_anchors, num_classes,head_chann = 256, width = 1.0):
    in_channels = inputs.get_shape()[-1]
    pad_name = "_head_" + str(in_channels)

    init_name = "conv2d_12" + pad_name
    cls_names = ["conv2d_13" + pad_name,
                 "conv2d_14" + pad_name,
                 "pred_cls" + pad_name]

    reg_names = ["conv2d_15" + pad_name,
                 "conv2d_16" + pad_name,
                 "pred_reg" + pad_name]

    obj_name = "pred_obj" + pad_name

    head_chann = int(width*head_chann)
        
    x = DarknetConv2D(head_chann, (1,1), name = init_name)(inputs)

    # classification features.............
    cls_out = DarknetConv2D(head_chann, (3,3), name = cls_names[0])(x)
    cls_out = DarknetConv2D(head_chann, (3,3), name = cls_names[1])(cls_out)

    cls_out = DarknetConv2D(num_anchors*num_classes, (1,1), name = cls_names[2])(cls_out)

    # regression-IoU features...........
    reg_out = DarknetConv2D(head_chann, (3,3), name = reg_names[0])(x)
    reg_out = DarknetConv2D(head_chann, (3,3), name = reg_names[1])(reg_out)

    # obj output..........
    obj_out = DarknetConv2D(num_anchors, (1,1), name = obj_name)(reg_out)
    # regression output..................
    reg_out = DarknetConv2D(num_anchors*4, (1,1), name = reg_names[2])(reg_out)

    out = Concatenate(axis=-1)([reg_out,obj_out, cls_out])


    return out




def tiny_yolox_darknet(inputs, num_anchors, num_classes, weights_path):
    """ Tiny-Yolox model with yolov3-tiny backbone.................."""
     
    # TODO: get darknet class number from class file
    num_classes_coco = 80
    base_model = tiny_yolo3_body(inputs, num_anchors, num_classes_coco)
    base_model.load_weights(weights_path, by_name=False)
    print("Load weights {}.".format(weights_path))

    # get conv output in original network
    y1 = base_model.layers[40].output
    y2 = base_model.layers[41].output

    y1 = yolox_head(y1, num_anchors = num_anchors,
                    num_classes = num_classes,
                    head_chann = 256, width = 0.3)
    y2 = yolox_head(y2, num_anchors = num_anchors,
                    num_classes = num_classes,
                    head_chann = 256, width = 0.3)
    
    
    return Model(inputs, [y1, y2])
