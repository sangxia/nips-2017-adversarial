import time
start_time = time.time()

import multiprocessing as mp
mp.set_start_method('fork')

import ctypes as c

import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
import image_generators
import task_utils

task_utils.start_time = start_time


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, type=str, \
            help='Input directory with images.')
    parser.add_argument('--output_file', required=True, type=str, \
            help='Output directory with images.')
    parser.add_argument('--train_batch_size', default=24, type=int, \
            help='How many images process at one time.')
    parser.add_argument('--num_samples', default=-1, type=int, \
            help='Number of samples, -1 for all samples')
    args = parser.parse_args()

    input_dir_abs = os.path.abspath(args.input_dir)

    images_info = image_generators.get_names_and_labels(\
            args.input_dir, mode='test')
    
    print('Total images:', len(images_info))

    eps = 8.

    if args.num_samples < 0:
        num_samples = len(images_info)
    else:
        num_samples = args.num_samples

    use_avg_pred = True
    step_size = eps
    noise_size = eps*0.2
    grad_aug_scale = 0.02

    print('Loading images')
    # each image_data element: filename, img, lbl
    image_data = image_generators.load_images_into_batches(\
            images_info, args.train_batch_size, \
            max_samples=num_samples, to_buffer=True)
    print('Generating placeholders')
    placeholders = {}
    for i, data in enumerate(image_data):
        imgmat_size = (len(data[0]), 299, 299, 3)
        lblmat_size = (len(data[0]), 1000)
        data_min = task_utils.get_raw_array(init=np.clip(task_utils.get_array(data[1], imgmat_size)-eps, 0, None))
        data_max = task_utils.get_raw_array(init=np.clip(task_utils.get_array(data[1], imgmat_size)+eps, None, 255))
        data_actual = data[1]
        grad_mat = task_utils.get_raw_array(dims=imgmat_size)
        lbl_mat = task_utils.get_raw_array(dims=lblmat_size)
        # data, min, max, grad, lbl, pseudo-label matrix
        placeholders[i] = (\
                data[0], \
                data_actual, \
                data_min, \
                data_max, \
                grad_mat, \
                data[2], \
                lbl_mat)
    print('Preparing functions')

    # each tuple is a sequence of pairs followed by two floats
    # the pairs describe which network to use and the rand_repeat for accumulating gradients
    # the two floats are the step size and noise size of that FGSM step

    source_models = ['incresv2ensadv','inceptionv3adv', 'inceptionv3ens3adv', 'inceptionv3ens4adv']
    dist_pairs = [('inceptionv3ens3adv', 'inceptionv3ens4adv'), \
            ('inceptionv3ens3adv', 'incresv2ensadv'), \
            ('inceptionv3ens4adv', 'incresv2ensadv'), \
            ('inceptionv3adv', 'incresv2ensadv'), \
            ('inceptionv3ens3adv', 'inceptionv3adv'), \
            ('inceptionv3ens4adv', 'inceptionv3adv'), \
            ]

    step_size = 1.
    noise_size = 0.4
    grad_aug_scale = 0.01
    pred_aug_scale = 0.01

    plan = [\
            (('RST', 'def'), \
            # ((source_models, dist_pairs), \
                ((('inceptionv3ens3adv','inceptionv3ens4adv'), 1), step_size, noise_size), \
                ((('inceptionv3ens3adv','inceptionv3ens4adv'), 1), step_size, noise_size), \
                ((('inceptionv3ens4adv','incresv2ensadv'), 1), (('inceptionv3ens3adv','incresv2ensadv'), 1), step_size, noise_size), \
                ((('inceptionv3ens3adv','inceptionv3ens4adv'), 1), step_size, noise_size), \
                ((('inceptionv3ens4adv','incresv2ensadv'), 1), (('inceptionv3ens3adv','incresv2ensadv'), 1), step_size, noise_size), \
                ((('inceptionv3ens3adv','inceptionv3ens4adv'), 1), step_size, noise_size), \
                ((('inceptionv3ens4adv','inceptionv3adv'), 1), (('inceptionv3ens3adv','incresv2ensadv'), 1), step_size, noise_size)), \
            ]

    task_list = []
    phase_info = {}
    next_task = {}

    for i_phase, phase in enumerate(plan):
        last_task_in_batch = {}
        if phase[0][0] == 'RST':
            task = ('phase {0} restore from {1}'.format(i_phase, phase[0][1]), \
                    (), 7, (phase[0][1],))
            task_list.append(task)
            phase_info[i_phase] = (len(task_list)-1, set(), set())
        else:
            task = ('phase {0} load {1}'.format(i_phase, phase[0]), (), 5+1*(i_phase==0), \
                    (phase[0][0], phase[0][1]))
            task_list.append(task)
            # id of loading task, all tasks other than load, first tasks after load
            phase_info[i_phase] = (len(task_list)-1, set(), set())
        for phid, (fname_list, imgs, imgs_min, imgs_max, grads, lbls, lbls_mat) in placeholders.items():
            for i_pl, pl in enumerate(phase[1:]):
                if pl[-1]>1e-5:
                    task = ('phase {0} batch {1} step {2}, noise {3:.2f}'.format(i_phase, phid, i_pl, pl[-1]), \
                            (fname_list,imgs,None,grads,imgs_min,imgs_max,lbls_mat),4,\
                            (pl[-1],))
                    task_list.append(task)

                    # COPIED
                    if phid not in last_task_in_batch:
                        phase_info[i_phase][2].add(len(task_list)-1)
                    else:
                        next_task[last_task_in_batch[phid]] = len(task_list)-1
                    last_task_in_batch[phid] = len(task_list)-1
                    phase_info[i_phase][1].add(len(task_list)-1)

                for j_stp, stp in enumerate(pl[:-2]):
                    task = ('phase {0} batch {1} step {2}, grad {3} repeat {4}'.format(i_phase, phid, i_pl, stp[0], stp[1]), \
                            (fname_list,imgs,None,grads,imgs_min,imgs_max,lbls_mat),0,\
                            (None, stp[0], grad_aug_scale, stp[1]))
                    task_list.append(task)

                    # COPIED
                    if phid not in last_task_in_batch:
                        phase_info[i_phase][2].add(len(task_list)-1)
                    else:
                        next_task[last_task_in_batch[phid]] = len(task_list)-1
                    last_task_in_batch[phid] = len(task_list)-1
                    phase_info[i_phase][1].add(len(task_list)-1)

                task = ('phase {0} batch {1} step {2}, FGSM step, step size {3:.2f}'.format(i_phase, phid, i_pl, pl[-2]), \
                        (fname_list,imgs,None,grads,imgs_min,imgs_max,lbls_mat),1,\
                        (pl[-2],))
                task_list.append(task)
                # COPIED
                if phid not in last_task_in_batch:
                    phase_info[i_phase][2].add(len(task_list)-1)
                else:
                    next_task[last_task_in_batch[phid]] = len(task_list)-1
                last_task_in_batch[phid] = len(task_list)-1
                phase_info[i_phase][1].add(len(task_list)-1)

            task = ('phase {0} batch {1}, final pred'.format(i_phase, phid), \
                    (fname_list,imgs,None,grads,imgs_min,imgs_max,lbls_mat),3,(pred_aug_scale,))
            task_list.append(task)
            # COPIED
            if phid not in last_task_in_batch:
                phase_info[i_phase][2].add(len(task_list)-1)
            else:
                next_task[last_task_in_batch[phid]] = len(task_list)-1
            last_task_in_batch[phid] = len(task_list)-1
            phase_info[i_phase][1].add(len(task_list)-1)

    print('Preparation finished', time.time()-start_time)
    print(len(task_list), 'tasks')
    task_utils.run_tasks(task_list, phase_info, next_task, \
            verbose=False, cpu_worker=1, is_defense=True)

    with open(args.output_file, 'w') as f:
        for phid, (fnames, _, _, _, _, _, lbls_mat) in placeholders.items():
            batch_preds = task_utils.get_array(lbls_mat, (len(fnames),1000)).argmax(axis=1)
            for img, pred in zip(fnames, batch_preds):
                _ = f.write('{0},{1}\n'.format(os.path.split(img)[1], pred+1))

    print('Time:', time.time()-start_time)
    print('\n'.join(str(u) for u in plan))
    print()

if __name__ == '__main__':
    main()

