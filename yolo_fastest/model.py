import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.optimizers import Adam
from yolo3.loss import yolo3_loss
from yolo3.data import yolo3_data_generator
from common.model_utils import add_metrics, get_pruning_model
from common.utils import get_anchors

from yolo3.models.layers import DarknetConv2D


def get_yolo_fastest_model(
    model_type,
    num_feature_layers,
    num_anchors,
    num_classes,
    input_tensor=None,
    input_shape=None,
    model_pruning=False,
    pruning_end_step=10000,
):

    model_body = tf.keras.models.load_model("yolo_fastest/yolo-fastest-xl.h5")
    inputs = model_body.inputs

    y1 = model_body.layers[263].output
    y2 = model_body.layers[264].output
    
    backbone_len = 237


    y1 = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name="predict_conv_1")(y1)
    y2 = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name="predict_conv_2")(y2)


    model_body = Model(inputs = inputs, outputs = [y1,y2])

    return model_body, backbone_len


def get_yolo_fastest_train_model(
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
    anchors = get_anchors("./configs/yolo_fastest_anchors.txt")
    num_anchors = 6
    # model has 6 anchors and 2 feature layers,
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

    model_body, backbone_len = get_yolo_fastest_model(
        model_type,
        num_feature_layers,
        num_anchors//num_feature_layers,
        num_classes,
        model_pruning=model_pruning,
        pruning_end_step=pruning_end_step,
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
        num = (backbone_len, len(model_body.layers) - 2)[freeze_level - 1]
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
        yolo3_loss,
        name="yolo_loss",
        arguments={
            "anchors": anchors,
            "num_classes": num_classes,
            "ignore_thresh": 0.5,
            "label_smoothing": label_smoothing,
            "elim_grid_sense": False,
        },
    )([*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)

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


def yolo_fastest_data_generator_wrapper(
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
    return yolo3_data_generator(
        annotation_lines,
        batch_size,
        input_shape,
        anchors,
        num_classes,
        enhance_augment,
        rescale_interval,
        multi_anchor_assign,
    )
