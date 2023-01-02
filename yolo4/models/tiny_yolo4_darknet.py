
from tensorflow.keras.layers import Concatenate, Input, Lambda, UpSampling2D
from tensorflow.keras.models import Model
#from utils.utils import compose

#from nets.attention import cbam_block, eca_block, se_block, ca_block
#from nets.CSPdarknet53_tiny import (DarknetConv2D, DarknetConv2D_BN_Leaky,
#                                    darknet_body)
#from nets.yolo_training import yolo_loss

#attention = [se_block, cbam_block, eca_block, ca_block]

from CSPdarknet53_tiny import (DarknetConv2D, DarknetConv2D_BN_Leaky,
                                tiny_yolo_darknet_body)


def tiny_yolo4_body(input_shape, anchors_mask, num_classes):
    inputs = Input(input_shape)
    
    feat1, feat2 = tiny_yolo_darknet_body(inputs)
    #if phi >= 1 and phi <= 4:
    #    feat1 = attention[phi - 1](feat1, name='feat1')
    #    feat2 = attention[phi - 1](feat2, name='feat2')

    # 13,13,512 -> 13,13,256
    P5 = DarknetConv2D_BN_Leaky(256, (1,1))(feat2)
    # 13,13,256 -> 13,13,512 -> 13,13,255
    P5_output = DarknetConv2D_BN_Leaky(512, (3,3))(P5)
    P5_output = DarknetConv2D(len(anchors_mask[0]) * (num_classes+5), (1,1))(P5_output)
    
    # 13,13,256 -> 13,13,128 -> 26,26,128
    P5_upsample = compose(DarknetConv2D_BN_Leaky(128, (1,1)), UpSampling2D(2))(P5)
    if phi >= 1 and phi <= 4:
        P5_upsample = attention[phi - 1](P5_upsample, name='P5_upsample')

    # 26,26,256 + 26,26,128 -> 26,26,384
    P4 = Concatenate()([P5_upsample, feat1])
    
    # 26,26,384 -> 26,26,256 -> 26,26,255
    P4_output = DarknetConv2D_BN_Leaky(256, (3,3))(P4)
    P4_output = DarknetConv2D(len(anchors_mask[1]) * (num_classes+5), (1,1))(P4_output)
    
    return Model(inputs, [P5_output, P4_output])

