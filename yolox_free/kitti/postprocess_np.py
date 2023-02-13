#!/usr/bin/python3
# -*- coding=utf-8 -*-

import numpy as np
from scipy.special import expit, softmax
from common.yolo_postprocess_np import yolo_handle_predictions, yolo_adjust_boxes

def yolo_correct_boxes(predictions, img_shape, model_image_size, letterbox_padd = False):
    '''rescale predicition boxes back to original image shape'''
    box_xy = predictions[..., :2]
    box_wh = predictions[..., 2:4]
    objectness = np.expand_dims(predictions[..., 4], -1)
    class_scores = predictions[..., 5:]
    image_shape = np.array(img_shape, dtype='float32')

    # padding by mantaining the aspect ratio, otherwise stretch image to model input size..................
    if letterbox_padd:
        # model_image_size & image_shape should be (height, width) format
        model_image_size = np.array(model_image_size, dtype='float32')
        image_shape = np.array(img_shape, dtype='float32')
        height, width = image_shape

        new_shape = np.round(image_shape * np.min(model_image_size/image_shape))
        offset = (model_image_size-new_shape)/2./model_image_size
        scale = model_image_size/new_shape
        #reverse offset/scale to match (w,h) order
        offset = offset[..., ::-1]
        scale = scale[..., ::-1]

        box_xy = (box_xy - offset) * scale
        box_wh *= scale

    # Convert centoids to top left coordinates
    box_xy -= box_wh / 2

    # Scale boxes back to original image shape.
    image_wh = image_shape[..., ::-1]
    box_xy *= image_wh
    box_wh *= image_wh

    return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)


def yolo_decode(prediction, anchors, num_classes, input_dims, scale_x_y=None, use_softmax=False):
    '''Decode final layer features to bounding box parameters.'''
    batch_size = np.shape(prediction)[0]
    num_anchors = len(anchors)

    grid_size = np.shape(prediction)[1:3]
    #check if stride on height & width are same
    assert input_dims[0]//grid_size[0] == input_dims[1]//grid_size[1], 'model stride mismatch.'
    stride = input_dims[0] // grid_size[0]

    prediction = np.reshape(prediction,
                            (batch_size, grid_size[0] * grid_size[1] * num_anchors, num_classes + 5))

    ################################
    # generate x_y_offset grid map
    grid_y = np.arange(grid_size[0])
    grid_x = np.arange(grid_size[1])
    x_offset, y_offset = np.meshgrid(grid_x, grid_y)

    x_offset = np.reshape(x_offset, (-1, 1))
    y_offset = np.reshape(y_offset, (-1, 1))

    x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
    x_y_offset = np.tile(x_y_offset, (1, num_anchors))
    x_y_offset = np.reshape(x_y_offset, (-1, 2))
    x_y_offset = np.expand_dims(x_y_offset, 0)

    ################################

    # Log space transform of the height and width
    anchors = np.tile(anchors, (grid_size[0] * grid_size[1], 1))
    anchors = np.expand_dims(anchors, 0)

    '''
    if scale_x_y:
        # Eliminate grid sensitivity trick involved in YOLOv4
        #
        # Reference Paper & code:
        #     "YOLOv4: Optimal Speed and Accuracy of Object Detection"
        #     https://arxiv.org/abs/2004.10934
        #     https://github.com/opencv/opencv/issues/17148
        #
        box_xy_tmp = expit(prediction[..., :2]) * scale_x_y - (scale_x_y - 1) / 2
        box_xy = (box_xy_tmp + x_y_offset) / np.array(grid_size)[::-1]
    else:
        box_xy = (expit(prediction[..., :2]) + x_y_offset) / np.array(grid_size)[::-1]
    '''

    #max_grid_size = np.array([26,26])
    #grid_scale = max_grid_scale / grid_size
    #grid_scale_tensor = np.tile(grid_scale,(grid_size[0]*grid_size[1],1))
    #grid_scale_tensor = np.expand_dims(grid_scale_tensor,axis=0)
    
    box_xy = (expit(prediction[..., :2]) + x_y_offset) / np.array(grid_size)[::-1]
    box_wh = np.exp(prediction[...,2:4])
    
        
    # Sigmoid objectness scores
    objectness = expit(prediction[..., 4])  # p_o (objectness score)
    objectness = np.expand_dims(objectness, -1)  # To make the same number of values for axis 0 and 1

    if use_softmax:
        # Softmax class scores
        class_scores = softmax(prediction[..., 5:], axis=-1)
    else:
        # Sigmoid class scores
        class_scores = expit(prediction[..., 5:])

    return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)


def yolox_decode(predictions, anchors, num_classes, input_dims, elim_grid_sense=False):
    """
    YOLOv3 Head to process predictions from YOLOv3 models

    :param num_classes: Total number of classes
    :param anchors: YOLO style anchor list for bounding box assignment
    :param input_dims: Input dimensions of the image
    :param predictions: A list of three tensors with shape (N, 19, 19, 255), (N, 38, 38, 255) and (N, 76, 76, 255)
    :return: A tensor with the shape (N, num_boxes, 85)
    """

    anchors_per_layer = 1
    assert len(predictions) == len(anchors)// anchors_per_layer, 'anchor numbers does not match prediction.'

    anchor_mask = [[1], [1]]
    if len(predictions) == 3: # assume 3 set of predictions is YOLOv3
        #anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
        scale_x_y = [1.05, 1.1, 1.2] if elim_grid_sense else [None, None, None]
    elif len(predictions) == 2: # 2 set of predictions is YOLOv3-tiny
        #anchor_mask = [[3,4,5], [0,1,2]]
        scale_x_y = [1.05, 1.05] if elim_grid_sense else [None, None]
    else:
        raise ValueError('Unsupported prediction length: {}'.format(len(predictions)))

    results = []
    for i, prediction in enumerate(predictions):
        results.append(yolo_decode(prediction, anchors[anchor_mask[i]], num_classes, input_dims, scale_x_y=scale_x_y[i], use_softmax=False))

    return np.concatenate(results, axis=1)


def yolox_postprocess_np(yolo_outputs, image_shape, anchors, num_classes, model_image_size, max_boxes=100, confidence=0.1, iou_threshold=0.4, elim_grid_sense=False):
    predictions = yolox_decode(yolo_outputs, anchors, num_classes, input_dims=model_image_size, elim_grid_sense=elim_grid_sense)
    predictions = yolo_correct_boxes(predictions, image_shape, model_image_size)

    boxes, classes, scores = yolo_handle_predictions(predictions,
                                                     image_shape,
                                                     max_boxes=max_boxes,
                                                     confidence=confidence,
                                                     iou_threshold=iou_threshold)

    boxes = yolo_adjust_boxes(boxes, image_shape)

    return boxes, classes, scores

