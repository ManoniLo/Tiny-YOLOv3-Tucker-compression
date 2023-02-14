#!/usr/bin/python3
# -*- coding=utf-8 -*-

import numpy as np
from common.yolo_postprocess_np import yolo_decode, yolo_handle_predictions, yolo_adjust_boxes

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


def yolo_lite_postprocess_np(yolo_outputs, image_shape, anchors, num_classes, model_image_size, max_boxes=100, confidence=0.1, iou_threshold=0.4, elim_grid_sense=False):

    scale_x_y = 1.05 if elim_grid_sense else None
    predictions = yolo_decode(yolo_outputs, anchors, num_classes, input_dims=model_image_size, scale_x_y=scale_x_y, use_softmax=True)
    predictions = yolo_correct_boxes(predictions, image_shape, model_image_size, letterbox_padd = False)

    boxes, classes, scores = yolo_handle_predictions(predictions,
                                                     image_shape,
                                                     max_boxes=max_boxes,
                                                     confidence=confidence,
                                                     iou_threshold=iou_threshold)

    boxes = yolo_adjust_boxes(boxes, image_shape)

    return boxes, classes, scores

