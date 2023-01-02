from functools import wraps

import tensorflow as tf
from tensorflow.keras.initializers import random_normal
from tensorflow.keras.layers import (Concatenate, Conv2D, Lambda, MaxPooling2D,
                                     ZeroPadding2D)
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.regularizers import l2


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')



#from utils.utils import compose


def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]


  
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : random_normal(stddev=0.02)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)
    

    
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

'''
                    input
                      |
            DarknetConv2D_BN_Leaky
                      -----------------------
                      |                     |
                 route_group              route
                      |                     |
            DarknetConv2D_BN_Leaky          |
                      |                     |
    -------------------                     |
    |                 |                     |
 route_1    DarknetConv2D_BN_Leaky          |
    |                 |                     |
    -------------Concatenate                |
                      |                     |
        ----DarknetConv2D_BN_Leaky          |
        |             |                     |
      feat       Concatenate-----------------
                      |
                 MaxPooling2D
'''

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


def tiny_yolo_darknet_body(x):
  
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(32, (3,3), strides=(2,2))(x)
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(64, (3,3), strides=(2,2))(x)
    
    # 104,104,64 -> 52,52,128
    x, _ = resblock_body(x,num_filters = 64)
    # 52,52,128 -> 26,26,256
    x, _ = resblock_body(x,num_filters = 128)
    # 26,26,256 -> x为13,13,512
    #           -> feat1为26,26,256
    x, feat1 = resblock_body(x,num_filters = 256)
    # 13,13,512 -> 13,13,512
    x = DarknetConv2D_BN_Leaky(512, (3,3))(x)

    feat2 = x
    return feat1, feat2
  
  
