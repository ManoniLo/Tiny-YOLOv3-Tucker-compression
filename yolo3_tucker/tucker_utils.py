import tensorflow as tf
import numpy as np


def get_tucker_comp_info(model,cfvalues):
    comp_info={}

    for name in cfvalues.keys():
        layer=model.get_layer(name=name)
        Weights=layer.get_weights()[0]

        Wsize=np.array(Weights.shape)
        Wnumel=np.prod(Wsize)
        cf=cfvalues[name]

        if isinstance(layer,tf.keras.layers.Dense):
            tuck_modes=[1]
            tucker_ranks=[int(round(cf*Wnumel/np.sum(Wsize)))]

        
        elif isinstance(layer,tf.keras.layers.Conv2D) or isinstance(layer,tf.keras.layers.Conv2DTranspose):
            tuck_modes=[2,3]
            gamma = 1.0
            #Bf=np.sum(np.square(Wsize[2:]))/Wnumel
            Bf = (Wsize[2]**2 + Wsize[3]**2*gamma)/Wnumel
            alpha=(-Bf+np.sqrt(np.square(Bf)+4*cf*gamma))/(2*gamma)
            beta=alpha*gamma
            tucker_ranks=np.round(np.array([alpha*Wsize[2],beta*Wsize[3]])).astype(np.int)
            print('layer {} ranks {}'.format(name,tucker_ranks))

        comp_info.update({name:{}})
        comp_info[name]['rank']=tucker_ranks

    return comp_info
