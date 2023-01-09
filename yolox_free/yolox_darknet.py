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

#from yolo3.models.yolo3_darknet import tiny_yolo3_body

def tiny_yolo3_body(inputs, num_anchors, num_classes):
    """Create Tiny YOLO_v3 model CNN body in keras."""
    # feature map 2 (26x26x256 for 416 input)
    f2 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3), name = 'conv2d'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        DarknetConv2D_BN_Leaky(32, (3, 3), name = 'conv2d_1'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        DarknetConv2D_BN_Leaky(64, (3, 3), name = 'conv2d_2'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        DarknetConv2D_BN_Leaky(128, (3, 3), name = 'conv2d_3'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        DarknetConv2D_BN_Leaky(256, (3, 3), name = 'conv2d_4'),
    )(inputs)

    # feature map 1 (13x13x1024 for 416 input)
    f1 = compose(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        DarknetConv2D_BN_Leaky(512, (3, 3), name = 'conv2d_5'),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        DarknetConv2D_BN_Leaky(1024, (3, 3),name = 'conv2d_6'),
    )(f2)

    # feature map 1 transform
    x1 = DarknetConv2D_BN_Leaky(256, (1, 1), name = 'conv2d_7')(f1)

    # feature map 1 output (13x13 for 416 input)
    y1 = compose(
        DarknetConv2D_BN_Leaky(512, (3, 3), name = "conv2d_8"),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name="predict_conv_1"),
    )(x1)

    # upsample fpn merge for feature map 1 & 2
    x2 = compose(DarknetConv2D_BN_Leaky(128, (1, 1), name = 'conv2d_10'), UpSampling2D(2))(x1)

    # feature map 2 output (26x26 for 416 input)
    y2 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(256, (3, 3), name = "conv2d_11"),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name="predict_conv_2"),
    )([x2, f2])

    return Model(inputs, [y1, y2])

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
    base_model.load_weights(weights_path, by_name=True,skip_mismatch = True)
    print("Load weights {}.".format(weights_path))

    # get conv output in original network
    y1 = base_model.layers[40].output
    y2 = base_model.layers[41].output

    y1 = yolox_head(y1, num_anchors = num_anchors,
                    num_classes = num_classes,
                    head_chann = 256, width = 0.6)
    y2 = yolox_head(y2, num_anchors = num_anchors,
                    num_classes = num_classes,
                    head_chann = 256, width = 0.6)
    
    
    return Model(inputs, [y1, y2])
