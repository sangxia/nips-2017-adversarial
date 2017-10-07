import pickle
import network_utils
import tensorflow as tf
from keras import backend as K
from tensorflow.contrib import slim
from tensorflow.python.framework import graph_util

sess = K.get_session()

######################################
# Code for preparing defense models
file_name = 'def'
grad_dict = {}

# NOTE for some reason i3a MUST be loaded before the other two i3 adv models, why?
defense_models = ['incresv2ensadv', 'inceptionv3adv', 'inceptionv3ens4adv', 'inceptionv3ens3adv', ]
dist_pairs = [('inceptionv3ens3adv', 'inceptionv3ens4adv'), \
        ('inceptionv3ens3adv', 'incresv2ensadv'), \
        ('inceptionv3ens4adv', 'incresv2ensadv'), \
        ('inceptionv3adv', 'incresv2ensadv'), \
        ('inceptionv3ens3adv', 'inceptionv3adv'), \
        ('inceptionv3ens4adv', 'inceptionv3adv'), \
        ]
_ = network_utils.load_defense_model(\
        models = defense_models, \
        make_model = False, \
        dist_pairs = dist_pairs, \
        grad_dict = grad_dict)

######################################
# Code for preparing attack models

# # Phase 1 non-targeted
# is_targeted = False
# file_name = 'ir2ea-i3a-i3-r50-' + ('nt' if not is_targeted else 't')
# source_models = ['incresv2ensadv','resnet50', 'inceptionv3adv', 'inceptionv3']

# # Phase 2 non-targeted
# is_targeted = False
# file_name = 'i3-x-ir2-' + ('nt' if not is_targeted else 't')
# source_models = ['inceptionv3','incresv2','xception']

# # Phase 3 non-targeted
# is_targeted = False
# file_name = 'ir2ea-r101-i3a-' + ('nt' if not is_targeted else 't')
# source_models = ['incresv2ensadv', 'resnet101', 'inceptionv3adv']

# # Targeted
# is_targeted = True
# file_name = 'ir2ea-i3a-i3-r50-' + ('nt' if not is_targeted else 't')
# source_models = ['incresv2ensadv', 'inceptionv3adv', 'inceptionv3', 'resnet50']
# 
# pred_models = source_models
# grad_dict = {}
# _ = network_utils.load_source_model(models=source_models, \
#         pred_models=pred_models, \
#         is_targeted=is_targeted, \
#         grad_dict=grad_dict, \
#         make_model=False, \
#         load_weights=True)

######################################

name_dict = {}
output_op_names = set()
for gname, (input_list, output_list, _) in grad_dict.items():
    input_names = [t.name for t in input_list]
    output_names = [t.name for t in output_list]
    output_op_names = output_op_names.union(set(\
            [t.op.name for t in output_list]))
    name_dict[gname] = (input_names, output_names)
    print('{0}\n  Inputs: {1}\n  Outputs: {2}\n'.format(gname, \
            ' '.join(input_names), ' '.join(output_names)))
 
output_op_names.add(K.learning_phase().op.name)
print('Output set:', output_op_names)

print('Saving metadata')
with open(file_name + '.pickle', 'wb') as f:
    pickle.dump(name_dict, f)

print('Saving graph')
saver = tf.train.Saver()

graph_def = tf.get_default_graph().as_graph_def()
out_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, list(output_op_names))
with open(file_name + '.pb', 'wb') as f:
    f.write(out_graph_def.SerializeToString())

# save_path = saver.save(K.get_session(), file_name + '.ckpt')
# print(save_path)

