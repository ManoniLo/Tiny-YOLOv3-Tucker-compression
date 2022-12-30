import tensorflow as tf
from tensorflow.keras.initializers import random_normal
from tensorflow.keras.layers import (Concatenate, Conv2D, Lambda, MaxPooling2D,
                          ZeroPadding2D)
from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.regularizers import l2
from utils.utils import compose


def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]

#------------------------------------------------------#
#   单次卷积DarknetConv2D
#   如果步长为2则自己设定padding方式。
#------------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : random_normal(stddev=0.02)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)
    
#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


#---------------------------------------------------#
#   CSPdarknet_tiny
#---------------------------------------------------#
def resblock_body(x, num_filters):
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3))(x)
    route = x

    x = Lambda(route_group, arguments={'groups':2, 'group_id':1})(x) 
    x = DarknetConv2D_BN_Leaky(int(num_filters/2), (3,3))(x)
    route_1 = x
    x = DarknetConv2D_BN_Leaky(int(num_filters/2), (3,3))(x)
    x = Concatenate()([x, route_1])

    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    feat = x
    x = Concatenate()([route, x])

    x = MaxPooling2D(pool_size=[2,2],)(x)

    return x, feat

#---------------------------------------------------#
#   CSPdarknet_tiny
#---------------------------------------------------#
def darknet_body(x):
    # 416,416,3 -> 208,208,32 -> 104,104,64
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(32, (3,3), strides=(2,2))(x)
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(64, (3,3), strides=(2,2))(x)
    
    # 104,104,64 -> 52,52,128
    x, _ = resblock_body(x,num_filters = 64)
    # 52,52,128 -> 26,26,256
    x, _ = resblock_body(x,num_filters = 128)
    # 26,26,256 -> x 13,13,512
    #           -> feat1 26,26,256
    x, feat1 = resblock_body(x,num_filters = 256)
    # 13,13,512 -> 13,13,512
    x = DarknetConv2D_BN_Leaky(512, (3,3))(x)

    feat2 = x
    return feat1, feat2
  
  
  attention = [se_block, cbam_block, eca_block, ca_block]


def tiny_yolo4_body(input_shape, anchors_mask, num_classes, phi = 0):
    inputs = Input(input_shape)
    
    feat1, feat2 = darknet_body(inputs)
    if phi >= 1 and phi <= 4:
        feat1 = attention[phi - 1](feat1, name='feat1')
        feat2 = attention[phi - 1](feat2, name='feat2')

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
