#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create YOLOv3/v4 models with different backbone & head
"""
import warnings
from functools import partial

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from yolo3.models.layers import (
    compose,
    DarknetConv2D,
    DarknetConv2D_BN,
    DarknetConv2D_BN_Leaky,
    DarknetConv2D_BN_custom_Leaky,
    Depthwise_Separable_Conv2D_BN_Leaky,
    Darknet_Depthwise_Separable_Conv2D_BN_Leaky,
)

from common.model_utils import add_metrics, get_pruning_model
from common.utils import get_anchors
from yolo_lite.loss import yolo_lite_loss
#from yolo3.data import yolo3_data_generator
from yolo_lite.data import yolo_lite_data_generator


def get_yolo_lite_model(model_type,
                        num_feature_layers,
                        num_anchors,
                        num_classes,
                        input_tensor=None,
                        input_shape=None,
                        model_pruning=False,
                        pruning_end_step=10000):
    
    base_model = load_model('weights/yolo_lite_coco.h5')
    inputs = base_model.inputs
    
    y1 = base_model.layers[23].output
    y1 = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name="predict_conv_1")(
        y1)

    return Model(inputs, [y1]), None



def get_yolo_lite_train_model(
    model_type,
    anchors,
    num_classes,
    weights_path=None,
    freeze_level=1,
    optimizer=Adam(lr=1e-3, decay=0),
    label_smoothing=0,
    elim_grid_sense=False,
    model_pruning=False,
    pruning_end_step=10000,
):
    """create the training model, for YOLOv3"""
    # K.clear_session() # get a new session
    anchors = get_anchors("./configs/yolo_lite_anchors.txt")
    num_anchors = len(anchors)
    # model has 3 anchors and 1 feature layer,
    # so we can calculate feature layers number to get model type
    num_feature_layers = num_anchors // 5

    # feature map target value, so its shape should be like:
    # [
    #  (image_height/32, image_width/32, 3, num_classes+5),
    #  (image_height/16, image_width/16, 3, num_classes+5),
    #  (image_height/8, image_width/8, 3, num_classes+5)
    # ]
    y_true = [
        Input(shape=(None, None, 5, num_classes + 5), name="y_true_{}".format(l))
        for l in range(num_feature_layers)
    ]

    model_body, backbone_len = get_yolo_lite_model(
        model_type,
        num_feature_layers,
        num_anchors//num_feature_layers,
        num_classes,
        model_pruning=model_pruning,
        pruning_end_step=pruning_end_step
    )
    print(
        "Create {} {} model with {} anchors and {} classes.".format(
            "Tiny" if num_feature_layers == 2 else "",
            "yolo_fastest",
            num_anchors,
            num_classes,
        )
    )
    print("model layer number:", len(model_body.layers))

    if weights_path:
        model_body.load_weights(weights_path, by_name=True)  # , skip_mismatch=True)
        print("Load weights {}.".format(weights_path))

    if freeze_level in [1, 2]:
        # Freeze the backbone part or freeze all but final feature map & input layers
        num = (backbone_len, len(model_body.layers) - 1)[freeze_level - 1]
        for i in range(num):
            model_body.layers[i].trainable = False
        print(
            "Freeze the first {} layers of total {} layers.".format(
                num, len(model_body.layers)
            )
        )
    elif freeze_level == 0:
    #    # Unfreeze all layers.
        for i in range(len(model_body.layers)):
            model_body.layers[i].trainable = True
        print("Unfreeze all of the layers.")

    model_loss, location_loss, confidence_loss, class_loss = Lambda(
        yolo_lite_loss,
        name="yolo_loss",
        arguments={
            "anchors": anchors,
            "num_classes": num_classes,
            "label_smoothing": label_smoothing,
            "elim_grid_sense": False,
        },
    )([*model_body.outputs, *y_true])

    model = Model([model_body.inputs, *y_true], model_loss)

    loss_dict = {
        "location_loss": location_loss,
        "confidence_loss": confidence_loss,
        "class_loss": class_loss,
    }
    add_metrics(model, loss_dict)

    model.compile(
        optimizer=optimizer, loss={"yolo_loss": lambda y_true, y_pred: y_pred}
    )  # use custom yolo_loss Lambda layer

    return model


def yolo_lite_data_generator_wrapper(
    annotation_lines,
    batch_size,
    input_shape,
    anchors,
    num_classes,
    enhance_augment=None,
    rescale_interval=-1,
    multi_anchor_assign=False,
    **kwargs
):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    input_shape = (416, 416)
    return yolo_lite_data_generator(
        annotation_lines,
        batch_size,
        input_shape,
        anchors,
        num_classes,
        enhance_augment,
        rescale_interval,
        multi_anchor_assign,
    )
    




