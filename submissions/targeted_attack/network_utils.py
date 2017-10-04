import pickle
from itertools import combinations

import tensorflow as tf
from tensorflow.contrib.image import transform as images_transform
from tensorflow.contrib.image import rotate as images_rotate

from keras.engine.topology import Layer
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.layers import Input
from keras.layers.merge import Add, Average
from keras.layers.merge import _Merge
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3

import model_wrappers

INPUT_SIZE_299 = set(['inceptionv3', 'xception', 'inceptionv4', 'incresv2', \
        'incresv2ensadv', 'inceptionv3adv','inceptionv3ens3adv','inceptionv3ens4adv'])
INPUT_SIZE_224 = set(['resnet50'])
#INPUT_SIZE_224 = set(['resnet50', 'vgg16'])
INPUT_SIZE_224_TF = set(['resnet101'])
SUPPORTED_MODELS = INPUT_SIZE_299.union(INPUT_SIZE_224).union(INPUT_SIZE_224_TF)

#def total_variation_layer(imgs):
#    return Lambda(lambda x: tf.image.total_variation(x))(imgs)

class Gradients(Layer):

    def __init__(self, grad_sign, **kwargs):
        self._sign = grad_sign
        super().__init__(**kwargs)

    def call(self, x):
        return self._sign*tf.gradients(x[1], [x[0]])[0]

class Equal(Layer):

    def call(self, x):
        return K.equal(x[0], x[1])

class CategoricalCrossEntropy(Layer):

    def __init__(self, from_logits=False, **kwargs):
        self._from_logits = from_logits
        super().__init__(**kwargs)

    def call(self, x):
        # label, logit
        return K.categorical_crossentropy(x[1], x[0], from_logits=self._from_logits)

class ImageAugmentation(Layer):
    
    def call(self, x):
        # img, noise
        one = tf.fill([tf.shape(x[0])[0],1], 1.)
        zero = tf.fill([tf.shape(x[0])[0],1], 0.)
        transforms = tf.concat([one, zero, zero, zero, one, zero, zero, zero], axis=1)
        rands = tf.concat([tf.truncated_normal([tf.shape(x[0])[0],6], stddev=x[1]), zero, zero], axis=1)
        return images_transform(x[0], transforms+rands, interpolation='BILINEAR')

class ImageRotation(Layer):

    def call(self, x):
        """ imgs, scale, scale is in radians """
        rands = tf.truncated_normal([tf.shape(x[0])[0]], stddev=x[1])
        return images_rotate(x[0], rands, interpolation='BILINEAR')

class Subtract(_Merge):
    """Layer that subtracts two inputs.
    It takes as input a list of tensors of size 2,
    both of the same shape, and returns a single tensor, (inputs[0] - inputs[1]),
    also of the same shape.
    # Examples
    ```python
        import keras
        input1 = keras.layers.Input(shape=(16,))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(32,))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        # Equivalent to subtracted = keras.layers.subtract([x1, x2])
        subtracted = keras.layers.Subtract()([x1, x2])
        out = keras.layers.Dense(4)(subtracted)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    ```
    """

    def build(self, input_shape):
        super().build(input_shape)

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('`Subtract` layer should be called '
                             'on exactly 2 inputs')
        if inputs[0]._keras_shape != inputs[1]._keras_shape:
            raise ValueError('`Subtract` layer should be called '
                             'on inputs of the same shape')

        return inputs[0] - inputs[1]

class TotalVariationDistance(Layer):

    def call(self, x):
        diff_dist = Subtract()([x[0],x[1]])
        return K.sum(K.abs(diff_dist), axis=1)

def InceptionV3_pre(imgs, scope):
    return Lambda(lambda x: (x/255.-0.5)*2., name=scope+'inceptionv3_pre')(imgs)

def ResNet50_resize(imgs, scope):
    # bicubic doesn't have gradient defined
    return Lambda(lambda x: tf.image.resize_images(x, (224,224), method=tf.image.ResizeMethod.BILINEAR), \
            name=scope+'resnet50_resize')(imgs)

def ResNet50_TF_pre(imgs, scope):
    return Lambda(lambda x: x - [123.68,116.779,103.939], name=scope+'resnet50_tf_pre')(imgs)

def ResNet50_pre(imgs, scope):
    return Lambda(lambda x: K.reverse(x, len(x.shape)-1) - [103.939,116.779,123.68], name=scope+'resnet50_pre')(imgs)

def load_target_model(target_model='inceptionv3'):
    print('preparing', target_model)
    input_img = Input(shape=(299,299,3), name='target_input_img')
    return load_branch(target_model, input_img, None, None, True)

def restore_source_model(saved_ckpt_name, grad_dict=None, from_frozen=False):
    print('restoring', saved_ckpt_name)
    with open(saved_ckpt_name + '.pickle', 'rb') as f:
        info = pickle.load(f)
    print(info[0])
    print(info[1])
    sess = K.get_session()
    if not from_frozen:
        print('restoring graph')
        saver = tf.train.import_meta_graph(saved_ckpt_name + '.ckpt.meta')
        print('restoring variables')
        saver.restore(sess, saved_ckpt_name + '.ckpt')
    else:
        print('restoring frozen graph def')
        with open(saved_ckpt_name + '.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    ops = tf.get_default_graph().get_operations()
    tensor_search_name = set(info[0] + list(info[1].values()))
    found_tensors = {}
    for op in ops:
        if len(op.outputs) != 1:
            continue
        if op.outputs[0].name.startswith('src_input'):
            print(op.outputs[0].name)
        if op.outputs[0].name in tensor_search_name:
            found_tensors[op.outputs[0].name] = op.outputs[0]
    input_tensors = [found_tensors[nm] for nm in info[0]]
    pred_input_tensors = input_tensors[:2] + [input_tensors[3]]
    print(input_tensors)
    for model_name, tensor_name in info[1].items():
        grad_dict[model_name] = K.function(\
                input_tensors if model_name != 'PRED' else pred_input_tensors, \
                [found_tensors[tensor_name]])
    print('restore finished')

def load_defense_model(models=['xception','resnet50'], \
        dist_pairs = [('xception', 'resnet50')], \
        aug_or_rotate='aug', \
        make_model=False, \
        grad_dict=None):

    if grad_dict is not None:
        grad_dict.clear()

    input_img = Input(shape=(299,299,3), name='src_input_img')
    noise_input = Input(batch_shape=(1,), name='augmentation_scale')

    if aug_or_rotate not in ['aug', 'rot', 'none']:
        raise ValueError('invalid value for aug_or_rotate')
    
    model_outputs = []
    model_outputs_dict = {}
    for model_name in models:
        print('preparing', model_name)
        model_pred = load_branch(model_name, input_img, aug_or_rotate, noise_input, False, load_weights=True)
        model_outputs.append(model_pred)
        model_outputs_dict[model_name] = model_pred

        # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    print('preparing avg pred')
    model_outputs_avg = Average()(model_outputs)

    print('preparing gradients')
    pred_diff = None
    for m1, m2 in dist_pairs:
        tv = TotalVariationDistance(name='dist_{0}_{1}'.format(m1,m2))(\
                [model_outputs_dict[m1], model_outputs_dict[m2]])
        grad = Gradients(-1, name='grad_{0}_{1}'.format(m1,m2))([input_img, tv])
        if grad_dict is not None:
            if make_model:
                grad_dict[(m1,m2)] = grad
            else:
                grad_dict[(m1,m2)] = K.function(\
                        [input_img, noise_input, K.learning_phase()], \
                        [grad])

    if make_model:
        grad_dict['PRED'] = model_outputs_avg 
    else:
        grad_dict['PRED'] = K.function([input_img, noise_input, K.learning_phase()], [model_outputs_avg])

    if make_model:
        return Model(inputs=[input_img, noise_input], \
                outputs=[grad_dict[(m1,m2)] for m1,m2 in dist_pairs]+[grad_dict['PRED']])
    else:
        return None

def load_source_model(models=['xception','resnet50'], \
        pred_models=None, \
        is_targeted=False, \
        # output_avg_pred=False, \
        loss_pow=None, \
        aug_or_rotate='aug', \
        grad_dict=None, \
        make_model=False, \
        load_weights=True):

    # the gradients are supposed to be subtracted
    grad_sign = 1 if is_targeted else -1

    if grad_dict is not None:
        grad_dict.clear()

    input_img = Input(shape=(299,299,3), name='src_input_img')
    noise_input = Input(batch_shape=(1,), name='augmentation_scale')

    if aug_or_rotate not in ['aug', 'rot', 'none']:
        raise ValueError('invalid value for aug_or_rotate')

    target_labels = Input(shape=(1000,), name='target_labels')
    pred_max = Lambda(lambda x:K.max(x, axis=1, keepdims=True), name='target_labels_pred_max')(target_labels)
    y = Equal(name='target_pred_equal')([target_labels, pred_max])
    y = Lambda(lambda x:K.cast(x, K.floatx()), name='type_cast')(y)
    y = Lambda(lambda x:x / K.sum(x, axis=1, keepdims=True), name='pseudo_label')(y)

    model_outputs = []
    for model_name in models:
        print('preparing', model_name)
        model_pred = load_branch(model_name, input_img, aug_or_rotate, noise_input, False, load_weights=load_weights)
        if pred_models is not None and model_name in pred_models:
            model_outputs.append(model_pred)
        print('preparing gradients', model_name)
        branch_loss = CategoricalCrossEntropy(from_logits=False, name=model_name+'_ce')([y, model_pred])
        #if loss_pow:
        #    branch_loss = K.pow(branch_loss, loss_pow)

        branch_grads = Gradients(grad_sign, name='grad_'+model_name)([input_img, branch_loss])
        if grad_dict is not None:
            if make_model:
                grad_dict[model_name] = branch_grads
            else:
                grad_dict[model_name] = K.function(\
                        [input_img, noise_input, target_labels, K.learning_phase()], \
                        [branch_grads])
        print('finished', model_name)

    print('loading of source models finished')
    if pred_models and grad_dict is not None:
        if make_model:
            if len(model_outputs) == 1:
                grad_dict['PRED'] = Lambda(lambda x:x, name='PRED')(model_outputs[0])
            else:
                grad_dict['PRED'] = Average(name='PRED')(model_outputs)
        else:
            pred_func = K.function([input_img, noise_input, K.learning_phase()], [sum(model_outputs) / len(model_outputs)])
            grad_dict['PRED'] = pred_func

    if make_model:
        output_list = models
        if pred_models:
            output_list.append('PRED')
        #learning_phase = Input(batch_shape=(), \
        #    dtype='bool', \
        #    tensor=K.learning_phase(), \
        #    name='learning_phase')
        #grad_inputs = [input_img, noise_input, target_labels, learning_phase]
        grad_inputs = [input_img, noise_input, target_labels]
        model = Model(inputs=grad_inputs, \
                outputs=[grad_dict[m] \
                for m in output_list])
    else:
        model = None

    print('gradient computation finished')
    return model

def load_branch(model_name, input_img, aug_or_rotate, noise_input, return_model, load_weights=True, keras_rename=True):
    """
    return_model - return a keras Model if True, else return the 
        softmax activated output tensor 
    keras_rename - whether to rename layers if loading a keras model, 
        not sure if renaming is a good idea if return_model is True
    """

    def rename_layers(model, prefix):
        for m in model.layers:
            m.name = prefix + m.name

    if not (model_name in SUPPORTED_MODELS):
        raise FileNotFoundError(model_name + ' not found')

    input_resized = input_img if model_name in INPUT_SIZE_299 \
            else ResNet50_resize(input_img, model_name+'_')

    if aug_or_rotate == 'aug':
        input_aug = ImageAugmentation()([input_resized, noise_input])
    elif aug_or_rotate == 'rot':
        input_aug = ImageRotation()([input_resized, noise_input])
    else:
        input_aug = input_resized
    #if aug_layer_func is not None:
    #    input_aug = aug_layer_func(input_resized, noise_input)
    #else:
    #    input_aug = input_resized

    if model_name in INPUT_SIZE_299:
        input_preprocessed = InceptionV3_pre(input_aug, model_name + '_')
    elif model_name in INPUT_SIZE_224:
        input_preprocessed = ResNet50_pre(input_aug, model_name + '_')
    else:
        input_preprocessed = ResNet50_TF_pre(input_aug, model_name + '_')

    if model_name == 'xception':
        with tf.name_scope('xception'):
            model = Xception(weights=None, input_tensor=input_preprocessed)
            if load_weights:
                model.load_weights('xception.h5')
        if keras_rename:
            rename_layers(model, 'xception/')
        if not return_model:
            return model.output
        else:
            return model
    elif model_name == 'inceptionv3':
        with tf.name_scope('inceptionv3'):
            model = InceptionV3(weights=None, input_tensor=input_preprocessed)
            if load_weights:
                model.load_weights('inceptionv3.h5')
        if keras_rename:
            rename_layers(model, 'inceptionv3/')
        if not return_model:
            return model.output
        else:
            return model
    elif model_name == 'resnet50':
        with tf.name_scope('resnet50'):
            model = ResNet50(weights=None, input_tensor=input_preprocessed)
            if load_weights:
                model.load_weights('resnet50.h5')
        if keras_rename:
            rename_layers(model, 'resnet50/')
        if not return_model:
            return model.output
        else:
            return model
    elif model_name == 'inceptionv4':
        # JUST TO BE CLEAR: this is not logits that is the output here
        model_logits = model_wrappers.InceptionV4(\
                load_weights=load_weights, \
                # don't remember why this is needed
                is_prediction=True,\
                is_training=False, \
                create_aux_logits=False)(input_preprocessed)
        if not return_model:
            return model_logits
        else:
            return Model(inputs=input_img, outputs=model_logits, name='inception_v4')
    elif model_name == 'incresv2':
        model_logits = model_wrappers.IncResV2(\
                load_weights=load_weights, \
                is_prediction=True,\
                is_training=False, \
                create_aux_logits=False)(input_preprocessed)
        if not return_model:
            return model_logits
        else:
            return Model(inputs=input_img, outputs=model_logits, name='incres_v2')
    elif model_name == 'incresv2ensadv':
        model_logits = model_wrappers.IncResV2EnsAdv(\
                load_weights=load_weights, \
                is_prediction=True,\
                is_training=False, \
                create_aux_logits=False)(input_preprocessed)
        if not return_model:
            return model_logits
        else:
            return Model(inputs=input_img, outputs=model_logits, name='incres_v2_ens_adv')
    elif model_name == 'inceptionv3adv':
        model_logits = model_wrappers.InceptionV3Adv(\
                load_weights=load_weights, \
                is_prediction=True,\
                is_training=False)(input_preprocessed)
        if not return_model:
            return model_logits
        else:
            return Model(inputs=input_img, outputs=model_logits, name='inception_v3_adv')
    elif model_name == 'inceptionv3ens3adv':
        model_logits = model_wrappers.InceptionV3Ens3Adv(\
                load_weights=load_weights, \
                is_prediction=True,\
                is_training=False)(input_preprocessed)
        if not return_model:
            return model_logits
        else:
            return Model(inputs=input_img, outputs=model_logits, name='inception_v3_ens3_adv')
    elif model_name == 'inceptionv3ens4adv':
        model_logits = model_wrappers.InceptionV3Ens4Adv(\
                load_weights=load_weights, \
                is_prediction=True,\
                is_training=False)(input_preprocessed)
        if not return_model:
            return model_logits
        else:
            return Model(inputs=input_img, outputs=model_logits, name='inception_v3_ens4_adv')
    elif model_name == 'resnet101':
        model_logits = model_wrappers.ResNet101(\
                load_weights=load_weights, \
                is_prediction=True,\
                is_training=False)(input_preprocessed)
        if not return_model:
            return model_logits
        else:
            return Model(inputs=input_img, outputs=model_logits, name='resnet101')
    elif model_name == 'vgg16':
        print('WARNING: DO NOT USE VGG FOR SUBMISSION')
        model = VGG16(input_tensor=input_preprocessed)
#        model = VGG16(weights=None, input_tensor=input_preprocessed)
#        model.load_weights('vgg16.h5')
        if not return_model:
            return model.output
        else:
            return model


