#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 ResNet18 Model Defined in Keras."""

from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from Resnet18 import *

from yolo3.models.layers import yolo3_predictions, yolo3lite_predictions, tiny_yolo3_predictions, tiny_yolo3lite_predictions


def tiny_yolo3_resnet18_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 ResNet50 model CNN body in keras.'''
    
    
    resnet18 = ResNet18(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(resnet50.layers)))

    # input: 416 x 416 x 3
    # stage4_unit2_output: 13 x 13 x 512
    # stage3_unit2_output: 26 x 26 x 256

    # f1 :13 x 13 x 512
    f1 = resnet18.get_layer('stage4_unit2_output').output
    # f2: 26 x 26 x 256
    f2 = resnet18.get_layer('stage3_unit2_output').output

    f1_channel_num = 512
    f2_channel_num = 256

    y1, y2 = tiny_yolo3_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])

