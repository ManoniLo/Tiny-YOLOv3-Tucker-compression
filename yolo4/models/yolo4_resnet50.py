#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v4 ResNet50 Model Defined in Keras."""

from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50

from yolo4.models.layers import yolo4_predictions, yolo4lite_predictions, tiny_yolo4_predictions, tiny_yolo4lite_predictions


def yolo4_resnet50_body(inputs, num_anchors, num_classes):
    """Create YOLO_V4 ResNet50 model CNN body in Keras."""
    resnet50 = ResNet50(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(resnet50.layers)))

    # input: 416 x 416 x 3
    # conv5_block3_out: 13 x 13 x 2048
    # conv4_block6_out: 26 x 26 x 1024
    # conv3_block4_out: 52 x 52 x 512

    # f1 :13 x 13 x 2048
    f1 = resnet50.get_layer('conv5_block3_out').output
    # f2: 26 x 26 x 1024
    f2 = resnet50.get_layer('conv4_block6_out').output
    # f3 : 52 x 52 x 512
    f3 = resnet50.get_layer('conv3_block4_out').output

    f1_channel_num = 1024
    f2_channel_num = 512
    f3_channel_num = 256

    y1, y2, y3 = yolo4_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo4lite_resnet50_body(inputs, num_anchors, num_classes):
    '''Create YOLO_v4 Lite ResNet50 model CNN body in keras.'''
    resnet50 = ResNet50(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(resnet50.layers)))

    # input: 416 x 416 x 3
    # conv5_block3_out: 13 x 13 x 2048
    # conv4_block6_out: 26 x 26 x 1024
    # conv3_block4_out: 52 x 52 x 512

    # f1 :13 x 13 x 2048
    f1 = resnet50.get_layer('conv5_block3_out').output
    # f2: 26 x 26 x 1024
    f2 = resnet50.get_layer('conv4_block6_out').output
    # f3 : 52 x 52 x 512
    f3 = resnet50.get_layer('conv3_block4_out').output

    f1_channel_num = 1024
    f2_channel_num = 512
    f3_channel_num = 256

    y1, y2, y3 = yolo4lite_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def tiny_yolo4_resnet50_body(inputs, num_anchors, num_classes, use_spp=True):
    '''Create Tiny YOLO_v4 ResNet50 model CNN body in keras.'''
    resnet50 = ResNet50(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(resnet50.layers)))

    # input: 416 x 416 x 3
    # conv5_block3_out: 13 x 13 x 2048
    # conv4_block6_out: 26 x 26 x 1024
    # conv3_block4_out: 52 x 52 x 512

    # f1 :13 x 13 x 2048
    f1 = resnet50.get_layer('conv5_block3_out').output
    # f2: 26 x 26 x 1024
    f2 = resnet50.get_layer('conv4_block6_out').output

    f1_channel_num = 1024
    f2_channel_num = 512

    y1, y2 = tiny_yolo4_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes, use_spp)

    return Model(inputs, [y1,y2])


def tiny_yolo4lite_resnet50_body(inputs, num_anchors, num_classes, use_spp=True):
    '''Create Tiny YOLO_v4 Lite ResNet50 model CNN body in keras.'''
    resnet50 = ResNet50(input_tensor=inputs, weights='imagenet', include_top=False)
    print('backbone layers number: {}'.format(len(resnet50.layers)))

    # input: 416 x 416 x 3
    # conv5_block3_out: 13 x 13 x 2048
    # conv4_block6_out: 26 x 26 x 1024
    # conv3_block4_out: 52 x 52 x 512

    # f1 :13 x 13 x 2048
    f1 = resnet50.get_layer('conv5_block3_out').output
    # f2: 26 x 26 x 1024
    f2 = resnet50.get_layer('conv4_block6_out').output

    f1_channel_num = 1024
    f2_channel_num = 512

    y1, y2 = tiny_yolo4lite_predictions((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes, use_spp)

    return Model(inputs, [y1,y2])

