#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 Darknet Model Defined in Keras."""



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
    ReLU,
    Dropout
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
from yolo3.models.layers import (
    yolo3_predictions,
    yolo3lite_predictions,
    tiny_yolo3_predictions,
    tiny_yolo3lite_predictions,
)


# Truncated darknet53 model...................................
def resblock_body_custom(x, num_filters, num_blocks):
    """Returns each residual block output"""
    # Darknet uses left and top padding instead of 'same' mode
    init_blocks = [ZeroPadding2D(((1, 0), (1, 0))), \
                   DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))]

    x = compose(init_blocks[0], init_blocks[1])(x)
    blocks = [init_blocks]
    first_conv = x
    for i in range(num_blocks):
        block_layers = [DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)), \
                        DarknetConv2D_BN_Leaky(num_filters, (3, 3))]
        #y = compose(
        #    block_layers[0],
        #    block_layers[1]
        #)(x)
        #x = Add()([x, y])
        blocks.append(block_layers)
    
    return first_conv, blocks


def darknet53_truncated_body(x, stages = [1,2,8,8,4], prune_method = 'last',
                          indexes = None):
    """Darknet53 body having 52 Convolution2D layers"""
    fr_stages = [1,2,8,8,4]
    filters = [64,128,256,512,1024]
    stages_outputs = []
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    for i,fr_blocks in enumerate(fr_stages):
        num_blocks = stages[i]
        first_conv, block_layers = resblock_body_custom(x, filters[i], fr_blocks)
        # indexes of selected residual blocks........................
        if prune_method == 'last':
            indexes = range(0,num_blocks)
        elif prune_method == 'first':
            indexes = range(fr_blocks-1,fr_blocks-num_blocks,-1)

        # first block of the stage...............
        x = compose(block_layers[0][0], block_layers[0][1])(x)
        for idx in indexes:
            y = compose(block_layers[idx+1][0], block_layers[idx+1][1])(x)
            x = Add()([x,y])


        stages_outputs.append(x)
    return x, stages_outputs


def yolo3_truncated_body(inputs, num_anchors, num_classes,
                         stages = [1,2,8,8,4], prune_method = 'last',
                         weights_path=None):

    """Create YOLO_V3 model CNN body in Keras."""
    final_output, stages_outputs = darknet53_truncated_body(inputs, stages = stages, prune_method = prune_method)
    darknet = Model(inputs, outputs = final_output)
    if weights_path is not None:
        darknet.load_weights(weights_path, by_name=True)
        print("Load weights {}.".format(weights_path))

    # f1: 13 x 13 x 1024
    f1 = darknet.output
    # f2: 26 x 26 x 512
    #f2 = darknet.layers[152].output
    f2 = stages_outputs[3]
    # f3: 52 x 52 x 256
    #f3 = darknet.layers[92].output
    f3 = stages_outputs[2]


    
    f1_channel_num = 1024
    f2_channel_num = 512
    f3_channel_num = 256

    y1, y2, y3 = yolo3_predictions(
        (f1, f2, f3),
        (f1_channel_num, f2_channel_num, f3_channel_num),
        num_anchors,
        num_classes,
    )

    return Model(inputs, [y1, y2, y3])


# Darknet53 with dropout in residual blocks............

def resblock_body_dropout(x, num_filters, num_blocks, dropout = 0.2):
    """A series of resblocks starting with a downsampling Convolution2D"""
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)),
        )(x)
        y = Dropout(rate = dropout)(y)
        x = Add()([x, y])
    return x



def darknet53_dropout_body(x):
    """Darknet53 body having 52 Convolution2D layers"""
    dropout = 0.2
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body_dropout(x, 64, 1, dropout)
    x = resblock_body_dropout(x, 128, 2, dropout)
    x = resblock_body_dropout(x, 256, 8, dropout)
    x = resblock_body_dropout(x, 512, 8, dropout)
    x = resblock_body_dropout(x, 1024, 4, dropout)
    return x


def yolo3_dropout_body(inputs, num_anchors, num_classes, weights_path=None):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet53_dropout_body(inputs))
    if weights_path is not None:
        darknet.load_weights(weights_path, by_name=True)
        print("Load weights {}.".format(weights_path))

    # f1: 13 x 13 x 1024
    f1 = darknet.output
    # f2: 26 x 26 x 512
    f2 = darknet.layers[152].output
    # f3: 52 x 52 x 256
    f3 = darknet.layers[92].output

    f1_channel_num = 1024
    f2_channel_num = 512
    f3_channel_num = 256

    y1, y2, y3 = yolo3_predictions(
        (f1, f2, f3),
        (f1_channel_num, f2_channel_num, f3_channel_num),
        num_anchors,
        num_classes,
    )

    model = Model(inputs, [y1, y2, y3])
    #readjust final conv names........
    final_conv_1 = model.get_layer(name = 'conv2d_58')
    final_conv_1.__name = 'predict_conv1'

    final_conv_2 = model.get_layer(name = 'conv2d_66')
    final_conv_2.__name = 'predict_conv2'

    final_conv_3 = model.get_layer(name = 'conv2d_74')
    final_conv_3.__name = 'predict_conv3'
    

    return model











