
from __future__ import print_function
import tensorflow as tf
#import keras
import numpy as np

import tensorflow.keras.layers as Layers
import tensorflow.keras.models
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
import matplotlib

from sklearn.metrics import classification_report
#from keras.utils import np_utils
#from imutils import build_montages
import warnings

import tensorly as tl
from tensorly.decomposition import tucker,partial_tucker

# Tucker decomposition of models for Pedestrian Tracking........................
# Added support for kernel regularization...................................
# Models are compiled only with metrics....................

def tucker_sequential_init(model,comp_info,model_metric="accuracy"):
  model_layers=model.layers
  comp_model=Sequential()

  comp_layer_names=comp_info.keys()

  for lay_indx,layer in enumerate(model_layers):
    layer_name=layer.name

    if layer_name in comp_layer_names:
      # Standard Convolutional Layer....................
      if isinstance(layer,Layers.Conv2D):
        Weights=layer.get_weights()[0]
        Wshape=Weights.shape
        filter_size=np.asarray(Wshape[:2])

        ranks=comp_info[layer_name]['rank']

        core_size=np.append(filter_size,np.array(ranks))
        prev_layer=Layers.Conv2D(ranks[0],kernel_size=(1,1),
                                 kernel_regularizer=layer.kernel_regularizer,
                                 input_shape=layer.input_shape,
                                 use_bias=False,name=layer_name+'_prev')
        # Core layer is depth-wise instead of a standard convolutional layer...
        core_layer=Layers.Conv2D(ranks[1],kernel_size=filter_size,
                                 kernel_regularizer=layer.kernel_regularizer,
                                 padding=layer.padding,strides=layer.strides,
                                 use_bias=False,name=layer_name+'_core')
        post_layer=Layers.Conv2D(Wshape[3],kernel_size=(1,1),
                                 kernel_regularizer=layer.kernel_regularizer,
                                 use_bias=layer.use_bias, activation=layer.activation,
                                 name=layer_name+'_post')


        comp_model.add(prev_layer)
        comp_model.add(core_layer)
        comp_model.add(post_layer)


      # Compression of a fully-connected layer...............
      elif isinstance(layer,Layers.Dense):
        rank=comp_info[layer_name]['rank'][0]

        prev_layer=Layers.Dense(rank,kernel_regularizer=layer.kernel_regularizer,
                                use_bias=False,name=layer_name+'_prev')
        post_layer=Layers.Dense(layer.units,use_bias=layer.use_bias,
                                kernel_regularizer=layer.kernel_regularizer,
                                activation=layer.activation,name=layer_name+'_post')
        
        
        comp_model.add(prev_layer)
        comp_model.add(post_layer)   

      elif isinstance(layer,Layers.Conv2DTranspose):
        ranks=comp_info[layer_name]['rank']
        Weights=layer.get_weights()[0]
        Wshape=Weights.shape
        filter_size=np.asarray(Wshape[:2])
          
        core_size=np.append(filter_size,ranks)
          
        prev_layer=Layers.Conv2DTranspose(ranks[0],kernel_size=(1,1),
                                          kernel_regularizer=layer.kernel_regularizer,
                                          input_shape=layer.input_shape,use_bias=False,
                                          name=layer_name+'_prev')
        # Core layer is depth-wise instead of a standard convolutional layer...
        core_layer=Layers.Conv2D(ranks[1],kernel_size=filter_size,
                                 kernel_regularizer=layer.kernel_regularizer,
                                 padding=layer.padding,strides=layer.strides,
                                 use_bias=False,name=layer_name+'_core')
        post_layer=Layers.Conv2D(Wshape[3],kernel_size=(1,1),
                                 kernel_regularizer=layer.kernel_regularizer,
                                 activation=layer.activation,name=layer_name+'_post')
          
        comp_model.add(prev_layer)
        comp_model.add(core_layer)
        comp_model.add(post_layer)
    

    # if we do not choose to compress layer.......  
    else:
      comp_model.add(layer)

  
  comp_model.compile(loss=model.loss,
              optimizer=model.optimizer,
              metrics=[model_metric])  
  
  return comp_model


def tucker_nonseq_init(model,comp_info,model_metric="acc"):
  
  # model must be loaded in order to delete node history................
  layers=model.layers
  comp_layer_names=comp_info.keys()
  network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}
  
  # mantaining order for multiple inputs layers................
  for layer in layers:
    layer_name=layer.name
    if layer_name not in network_dict['input_layers_of']:
      network_dict['input_layers_of'].update({layer_name: []})

    node=layer.inbound_nodes[0]
    input_layers=node.inbound_layers
    if not isinstance(input_layers,list):
      input_layers=[input_layers]
    for input_layer in input_layers:
      input_layer_name=input_layer.name
      network_dict['input_layers_of'][layer_name].append(input_layer_name)

  # This code does not mantain inputs' order.................. 
  # for layer in layers:
  #  input_layer_name=layer.name    
  #  for node in layer.outbound_nodes:
  #    layer_name = node.outbound_layer.name
  #    if layer_name not in network_dict['input_layers_of']:
  #      network_dict['input_layers_of'].update(
  #          {layer_name: [input_layer_name]})
  #    else:
  #      network_dict['input_layers_of'][layer_name].append(input_layer_name)

  
  network_dict['new_output_tensor_of'].update(
      {layers[0].name: model.input})

  outputs=[]
  
  for layer in layers[1:]:
    layer_name=layer.name
    layer_inputs=[network_dict['new_output_tensor_of'][layer_in_name] 
                  for layer_in_name in network_dict['input_layers_of'][layer_name]]

                  
    if len(layer_inputs)==1:
      layer_inputs=layer_inputs[0]
    
    if layer_name in comp_layer_names:
      if isinstance(layer,Layers.Conv2D):

        ranks=comp_info[layer_name]['rank']
        Weights=layer.get_weights()[0]


        if isinstance(layer,Layers.Conv2DTranspose):
          # Transposed Convolution........................
          Weights=np.transpose(Weights,(0,1,3,2))
          Wshape=Weights.shape
          filter_size=np.asarray(Wshape[:2])
          core_size=np.append(filter_size,ranks)

          prev_layer=Layers.Conv2DTranspose(ranks[0],kernel_size=(1,1),
                                            kernel_regularizer=layer.kernel_regularizer,
                                            input_shape=layer.input_shape, strides=layer.strides,
                                            use_bias=False,name=layer_name+'_prev')
          # Core layer is depth-wise instead of a standard convolutional layer...
          core_layer=Layers.Conv2D(ranks[1],kernel_size=filter_size,
                                   kernel_regularizer=layer.kernel_regularizer,
                                   padding=layer.padding,#strides=layer.strides,
                                   use_bias=False,name=layer_name+'_core')
          post_layer=Layers.Conv2D(Wshape[3],kernel_size=(1,1),
                                   kernel_regularizer=layer.kernel_regularizer,
                                   activation=layer.activation,
                                   name=layer_name+'_post')

          
          
        else:
          # Standard Convolutional Layer.............
          Wshape=Weights.shape
          filter_size=np.asarray(Wshape[:2])
          core_size=np.append(filter_size,ranks)

          
          prev_layer=Layers.Conv2D(ranks[0],kernel_size=(1,1),
                                   input_shape=layer.input_shape,
                                   kernel_regularizer=layer.kernel_regularizer,
                                   use_bias=False, name=layer_name+'_prev')
          
          core_layer=Layers.Conv2D(ranks[1],kernel_size=filter_size,
                                  padding=layer.padding,strides=layer.strides,
                                  kernel_regularizer = layer.kernel_regularizer,
                                  use_bias=False,name=layer_name+'_core')
          post_layer=Layers.Conv2D(Wshape[3],kernel_size=(1,1),
                                   kernel_regularizer=layer.kernel_regularizer,
                                   use_bias=layer.use_bias, activation=layer.activation,
                                   name=layer_name+'_post')

        
        x=prev_layer(layer_inputs)
        x=core_layer(x)
        new_output=post_layer(x)

      elif isinstance(layer,Layers.Dense):
        
        rank=comp_info[layer_name]['rank'][0]
        prev_layer=Layers.Dense(rank,input_shape=layer.input_shape,
                                kernel_regularizer=layer.kernel_regularizer,
                                use_bias=False,name=layer_name+'_prev')
        post_layer=Layers.Dense(layer.units,use_bias=layer.use_bias,
                                kernel_regularizer=layer.kernel_regularizer,
                                activation=layer.activation,name=layer_name+'_post')

        x=prev_layer(layer_inputs)
        new_output=post_layer(x)

      else: 
        warnings.warn("Selected layer to be compressed is neither Convolutional nor Dense Layer.\n It is possible that you have selected the wrong layer or you have mispelled it")

    else: 
      # simply skipping layer................
      new_output=layer(layer_inputs)

    # Adding output tensor of current layer to new model output tensors............... 
    network_dict['new_output_tensor_of'].update({layer_name: new_output})

    # Adding current output to create model..............
    if layer_name in model.output_names:
      outputs.append(new_output)
  
  comp_model=tf.keras.models.Model(inputs=model.inputs,outputs=outputs)
  model_metrics = comp_model.metrics
  comp_model.compile(metrics = model_metrics)
  #comp_model.compile(metrics=[model_metric])

  # save and reload model to reset layers' connections..........
  tmp_model_file = 'comp_model_tmp.h5'
  tf.keras.models.save_model(comp_model,tmp_model_file)
  comp_model = tf.keras.models.load_model(tmp_model_file)

  
  return comp_model



def tucker_comp_model(model,comp_info,return_info=False):

  if isinstance(model,Sequential):
    comp_model=tucker_sequential_init(model,comp_info)
  else:
    comp_model=tucker_nonseq_init(model,comp_info)
  
  comp_layer_names=comp_info.keys()
  tucker_max_iter_key="tucker_max_iter"
  tucker_tol_key="tucker_tol"
  max_iter_default=500
  tol_default=1e-10
  layers_info={}
  model_layer_names=[layer.name for layer in model.layers]

  for layer_name in comp_layer_names:

    if tucker_max_iter_key in comp_info[layer_name].keys():
      tucker_max_iter=comp_info[layer_name][tucker_max_iter_key]
    else:
      tucker_max_iter=max_iter_default

    if tucker_tol_key in comp_info[layer_name].keys():
      tucker_tol=comp_info[layer_name][tucker_tol_key]
    else:
      tucker_tol=tol_default

    if layer_name in model_layer_names:
      layer=model.get_layer(name=layer_name)
      if isinstance(layer,Layers.Conv2D):
        ranks=comp_info[layer_name]['rank']
        Weights=layer.get_weights()[0]
        print('setting weights........')
        
        if isinstance(layer,Layers.Conv2DTranspose):
          # Transposed Convolution........................
          print('Conv2Dtranspose setting weights.......')
          Weights=np.transpose(Weights,(0,1,3,2))
          Wshape=Weights.shape
          modes=[2,3]
        
          # Computing partial Tucker...............................
          Gtuck,Utuck=partial_tucker(Weights,modes=modes,rank=ranks,n_iter_max=tucker_max_iter,tol=tucker_tol)
          Weights_prev=np.expand_dims(np.expand_dims(np.transpose(Utuck[0]),axis=0),axis=0)
          Weights_core=Gtuck
          Weights_post=np.expand_dims(np.expand_dims(np.transpose(Utuck[1]),axis=0),axis=0)

          prev_layer=comp_model.get_layer(name=layer_name+'_prev')
          core_layer=comp_model.get_layer(name=layer_name+'_core')
          post_layer=comp_model.get_layer(name=layer_name+'_post')

          if return_info:
            Weights_hat=tl.tenalg.multi_mode_dot(Gtuck,Utuck,modes)
            error=tl.norm(Weights-Weights_hat)/tl.norm(Weights)
            layers_info.update({layer_name: {}})
            layers_info[layer_name]["Weights_approx"]=Weights_hat
            layers_info[layer_name]["error"]=error
            layers_info[layer_name]["G"]=Gtuck
            layers_info[layer_name]["Umatrices"]=Utuck


        else:    
          # Standard Convolutional layer...........
          print('Conv2D setting weights....')
          Wshape=Weights.shape
          modes=[2,3]
        
          # Computing partial Tucker...............................
          Gtuck,Utuck=partial_tucker(Weights,modes=modes,rank=ranks,n_iter_max=tucker_max_iter,tol=tucker_tol)
              
          Weights_prev=np.expand_dims(np.expand_dims(Utuck[0],axis=0),axis=0)
          Weights_core=Gtuck
          Weights_post=np.expand_dims(np.expand_dims(np.transpose(Utuck[1]),axis=0),axis=0)

          prev_layer=comp_model.get_layer(name=layer_name+'_prev')
          core_layer=comp_model.get_layer(name=layer_name+'_core')
          post_layer=comp_model.get_layer(name=layer_name+'_post')



          # Setting prev-convolutional layer weights...........
          prev_layer.set_weights([Weights_prev])
          # Depth-Wise core convolutional weights.............
          core_layer.set_weights([Weights_core])
          # Post convolutional layer..........................
          if layer.use_bias:
            post_bias=layer.get_weights()[1]
            post_layer.set_weights([Weights_post,post_bias])
          else:
            post_layer.set_weights([Weights_post])
          
          if return_info:
            Weights_hat=tl.tenalg.multi_mode_dot(Gtuck,Utuck,modes)
            error=tl.norm(Weights-Weights_hat)/tl.norm(Weights)
            layers_info.update({layer_name: {}})
            layers_info[layer_name]["Weights_approx"]=Weights_hat
            layers_info[layer_name]["error"]=error
            layers_info[layer_name]["G"]=Gtuck
            layers_info[layer_name]["Umatrices"]=Utuck


          
      elif isinstance(layer,Layers.Dense):
        
        rank=comp_info[layer_name]['rank']
        Weights=np.transpose(layer.get_weights()[0])
        post_bias=layer.get_weights()[1]
        Wshape=Weights.shape
        modes=[1]
        
        G,U=partial_tucker(Weights,modes=modes,rank=rank)

        prev_layer=comp_model.get_layer(name=layer_name+'_prev')
        post_layer=comp_model.get_layer(name=layer_name+'_post')
        
        prev_layer.set_weights([U[0]])
        post_layer.set_weights([np.transpose(G),post_bias])
        
        if return_info:
          Weights_hat=tl.tenalg.multi_mode_dot(G,U,modes)
          error=tl.norm(Weights-Weights_hat)/tl.norm(Weights)
          layers_info.update({layer_name: {}})
          layers_info[layer_name]["Weights_approx"]=Weights_hat
          layers_info[layer_name]["error"]=error
          layers_info[layer_name]["G"]=G
          layers_info[layer_name]["Umatrices"]=U

      


    
  if not return_info:
    return comp_model
  else:
    return comp_model,layers_info

