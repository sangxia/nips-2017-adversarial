import tensorflow as tf
from tensorflow.contrib import slim

from keras import backend as K
from keras.engine.topology import Layer

import inception_v4
import incres_v2
import resnet_v1
import inception_v3

def load_ckpt(ckpt_name, var_scope_name, scope, constructor, input_tensor, label_offset, load_weights, **kwargs):
    """ kwargs are is_training and create_aux_logits """
    print(var_scope_name)
    with slim.arg_scope(scope):
        logits, endpoints = constructor(\
                input_tensor, num_classes=1000+label_offset, \
                scope=var_scope_name, **kwargs)
    if load_weights:
        init_fn = slim.assign_from_checkpoint_fn(\
                ckpt_name, slim.get_model_variables(var_scope_name))
        init_fn(K.get_session())
    return logits, endpoints

class TFModel(Layer):

    def __init__(self, \
            ckpt_file_name, \
            var_scope_name, \
            scope, \
            constructor, \
            label_offset, \
            **kwargs):
        self._ckpt_file_name = ckpt_file_name
        self._var_scope_name = var_scope_name
        self._scope = scope
        self._constructor = constructor
        self._label_offset = label_offset
        self._kwparams = {}
        for kw in ['is_training', 'create_aux_logits']:
            if kw in kwargs:
                self._kwparams[kw] = kwargs[kw]
                del kwargs[kw]
        if 'is_prediction' in kwargs:
            self._is_prediction = kwargs['is_prediction']
            del kwargs['is_prediction']
        else:
            self._is_prediction = False
        if 'load_weights' in kwargs:
            self._load_weights = kwargs['load_weights']
            del kwargs['load_weights']
        print(self._kwparams)
        self._model_logits = None
        self._model_endpoints = None
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1000)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        if self._model_logits is None:
            self._model_logits, self._model_endpoints = load_ckpt(\
                    self._ckpt_file_name, self._var_scope_name, self._scope, \
                    self._constructor, x, self._label_offset, self._load_weights, **self._kwparams)

        if self._label_offset:
            slice_func = lambda x: \
                    tf.slice(x, [0,1], [-1,-1], name='imagenet_offset')
        else:
            slice_func = lambda x: x

        if self._is_prediction:
            return slice_func(self._model_endpoints['Predictions'])
        else:
            return slice_func(self._model_logits)

class InceptionV4(TFModel):

    def __init__(self, **kwargs):
        super().__init__('inception_v4.ckpt', 'InceptionV4', \
                inception_v4.inception_v4_arg_scope(), \
                inception_v4.inception_v4, 1, **kwargs)

class IncResV2(TFModel):

    def __init__(self, **kwargs):
        super().__init__('incres_v2.ckpt', 'InceptionResnetV2', \
                incres_v2.inception_resnet_v2_arg_scope(), \
                incres_v2.inception_resnet_v2, 1, **kwargs)


class ResNet101(TFModel):

    def __init__(self, **kwargs):
        super().__init__('resnet101.ckpt', 'resnet_v1_101', \
                resnet_v1.resnet_arg_scope(), \
                resnet_v1.resnet_v1_101, 0, **kwargs)

class IncResV2EnsAdv(TFModel):

    def __init__(self, **kwargs):
        super().__init__('incres_v2_ens_adv.ckpt', 'InceptionResnetV2', \
                incres_v2.inception_resnet_v2_arg_scope(), \
                incres_v2.inception_resnet_v2, 1, **kwargs)


class InceptionV3Adv(TFModel):

    def __init__(self, **kwargs):
        super().__init__('inception_v3_adv.ckpt', 'InceptionV3', \
                inception_v3.inception_v3_arg_scope(), \
                inception_v3.inception_v3, 1, **kwargs)


class InceptionV3Ens3Adv(TFModel):

    def __init__(self, **kwargs):
        super().__init__('inceptionv3ens3adv.ckpt', 'InceptionV3Ens3Adv', \
                inception_v3.inception_v3_arg_scope(), \
                inception_v3.inception_v3, 1, **kwargs)


class InceptionV3Ens4Adv(TFModel):

    def __init__(self, **kwargs):
        super().__init__('inceptionv3ens4adv.ckpt', 'InceptionV3Ens4Adv', \
                inception_v3.inception_v3_arg_scope(), \
                inception_v3.inception_v3, 1, **kwargs)


