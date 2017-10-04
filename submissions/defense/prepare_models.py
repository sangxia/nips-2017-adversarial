import pickle
import network_utils
import tensorflow as tf
from keras import backend as K
from tensorflow.contrib import slim
from tensorflow.python.framework import graph_util

sess = K.get_session()

file_name = 'def'
grad_dict = {}

# TODO for some reason i3a MUST be loaded before the other two i3 adv models, why?
defense_models = ['incresv2ensadv', 'inceptionv3adv', 'inceptionv3ens4adv', 'inceptionv3ens3adv', ]
dist_pairs = [('inceptionv3ens3adv', 'inceptionv3ens4adv'), \
        ('inceptionv3ens3adv', 'incresv2ensadv'), \
        ('inceptionv3ens4adv', 'incresv2ensadv'), \
        ('inceptionv3adv', 'incresv2ensadv'), \
        ('inceptionv3ens3adv', 'inceptionv3adv'), \
        ('inceptionv3ens4adv', 'inceptionv3adv'), \
        ]
model = network_utils.load_defense_model(\
        models = defense_models, \
        make_model = True, \
        dist_pairs = dist_pairs, \
        grad_dict = grad_dict)

#is_targeted = True
#file_name = 'ir2ea-r101-i3a-' + ('nt' if not is_targeted else 't')
#source_models = ['incresv2ensadv', 'resnet101', 'inceptionv3adv']
#pred_models = source_models
#grad_dict = {}
#model = network_utils.load_source_model(models=source_models, \
#        pred_models=pred_models, \
#        is_targeted=is_targeted, \
#        grad_dict=grad_dict, \
#        make_model=True, \
#        load_weights=True)

input_tensor_names = [t.name for t in model.inputs] + [K.learning_phase().name]
grad_dict_names = dict([(no, o.name) for no,o in grad_dict.items()])

print('Saving metadata')
print(input_tensor_names)
print(grad_dict_names)
with open(file_name + '.pickle', 'wb') as f:
    pickle.dump((input_tensor_names,grad_dict_names), f)

print('Saving graph')
saver = tf.train.Saver()

graph_def = tf.get_default_graph().as_graph_def()
out_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, \
        [t.op.name for t in grad_dict.values()])
with open(file_name + '.pb', 'wb') as f:
    f.write(out_graph_def.SerializeToString())

#save_path = saver.save(K.get_session(), file_name + '.ckpt')
#print(save_path)

