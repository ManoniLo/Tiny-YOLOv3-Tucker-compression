import sys
from .tucker_decomp import *
from .tucker_utils import *
#sys.path.append('../')

from yolo3.model import get_yolo3_model
from yolo3.models.yolo3_darknet import custom_tiny_yolo3_body
from yolo3.loss import yolo3_loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input,Lambda
from tensorflow.keras.models import Model

from common.model_utils import add_metrics, get_pruning_model


comp_name = '_tucker'
yolo3_tiny_model_map = {'tiny_yolo3_darknet': [custom_tiny_yolo3_body,20,'weights/yolov3-tiny.h5']}

def get_yolo3_comp_model(
    model_type,
    num_feature_layers,
    num_anchors,
    num_classes,
    layer_cfvalues,
    pretrained_weights_path = None,
    compute_decomp_weights = True,
    input_tensor=None,
    input_shape=None,
    model_pruning=False,
    pruning_end_step=10000,
):
    # prepare input tensor
    if input_shape:
        input_tensor = Input(shape=input_shape, name="image_input")

    if input_tensor is None:
        input_tensor = Input(shape=(None, None, 3), name="image_input")

    name_len = len(model_type) - len(comp_name)
    model_type = model_type[:name_len]
    

    # Tiny YOLOv3 model has 6 anchors and 2 feature layers
    if num_feature_layers == 2:
        if model_type in yolo3_tiny_model_map:
            model_function = yolo3_tiny_model_map[model_type][0]
            backbone_len = yolo3_tiny_model_map[model_type][1]
            weights_path = yolo3_tiny_model_map[model_type][2]

            if weights_path:
                model_body = model_function(
                    input_tensor,
                    num_anchors // 2,
                    num_classes,
                    weights_path=weights_path,
                )
            else:
                model_body = model_function(input_tensor, num_anchors // 2, num_classes)
        else:
            raise ValueError("This model type is not supported now")

    # YOLOv3 model has 9 anchors and 3 feature layers
    elif num_feature_layers == 3:
        if model_type in yolo3_model_map:
            model_function = yolo3_model_map[model_type][0]
            backbone_len = yolo3_model_map[model_type][1]
            weights_path = yolo3_model_map[model_type][2]

            if weights_path:
                model_body = model_function(
                    input_tensor,
                    num_anchors // 3,
                    num_classes,
                    weights_path=weights_path,
                )
            else:
                model_body = model_function(input_tensor, num_anchors // 3, num_classes)
        else:
            raise ValueError("This model type is not supported now")
    else:
        raise ValueError("model type mismatch anchors")

    if model_pruning:
        model_body = get_pruning_model(
            model_body, begin_step=0, end_step=pruning_end_step
        )

    if pretrained_weights_path is not None:
        print('Loading weights from ',pretrained_weights_path)
        model_body.load_weights(pretrained_weights_path,by_name = False)
        

    comp_info = get_tucker_comp_info(model_body,layer_cfvalues)
    if compute_decomp_weights:
        tucker_model_body = tucker_comp_model(model_body,comp_info)
    else:
        tucker_model_body = tucker_nonseq_init(model_body,comp_info)
        


    return tucker_model_body, None


def get_yolo3_comp_train_model(
    model_type,
    anchors,
    num_classes,
    layer_cfvalues,
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
    num_anchors = len(anchors)
    # YOLOv3 model has 9 anchors and 3 feature layers but
    # Tiny YOLOv3 model has 6 anchors and 2 feature layers,
    # so we can calculate feature layers number to get model type
    num_feature_layers = num_anchors // 3

    # feature map target value, so its shape should be like:
    # [
    #  (image_height/32, image_width/32, 3, num_classes+5),
    #  (image_height/16, image_width/16, 3, num_classes+5),
    #  (image_height/8, image_width/8, 3, num_classes+5)
    # ]
    y_true = [
        Input(shape=(None, None, 3, num_classes + 5), name="y_true_{}".format(l))
        for l in range(num_feature_layers)
    ]

    comp_model_body, backbone_len = get_yolo3_comp_model(
        model_type,
        num_feature_layers,
        num_anchors,
        num_classes,
        layer_cfvalues,
        pretrained_weights_path = weights_path,
        model_pruning=model_pruning,
        pruning_end_step=pruning_end_step,
    )
    
    
    print(
        "Create {} {} model with {} anchors and {} classes.".format(
            "Tiny" if num_feature_layers == 2 else "",
            model_type,
            num_anchors,
            num_classes,
        )
    )
    print("model layer number:", len(comp_model_body.layers))


    if freeze_level in [1, 2]:
        # Freeze the backbone part or freeze all but final feature map & input layers.
        num = (backbone_len, len(comp_model_body.layers) - 3)[freeze_level - 1]
        for i in range(num):
            comp_model_body.layers[i].trainable = False
        print(
            "Freeze the first {} layers of total {} layers.".format(
                num, len(comp_model_body.layers)
            )
        )
    elif freeze_level == 0:
        # Unfreeze all layers.
        for i in range(len(comp_model_body.layers)):
            comp_model_body.layers[i].trainable = True
        print("Unfreeze all of the layers.")


    model_loss, location_loss, confidence_loss, class_loss = Lambda(
        yolo3_loss,
        name="yolo_loss",
        arguments={
            "anchors": anchors,
            "num_classes": num_classes,
            "ignore_thresh": 0.5,
            "label_smoothing": label_smoothing,
            "elim_grid_sense": elim_grid_sense,
        },
    )([*comp_model_body.output, *y_true])

    model = Model([comp_model_body.input, *y_true], model_loss)

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
