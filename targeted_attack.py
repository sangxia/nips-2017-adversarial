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
    parser.add_argument('--output_dir', required=True, type=str, \
            help='Output directory with images.')
    parser.add_argument('--max_epsilon', default=16.0, type=float, \
            help='Maximum size of adversarial perturbation.')
    parser.add_argument('--input_dir_mode', default='test', type=str, \
            help='Either flat or hierarchy, how the input dir is organised')
    parser.add_argument('--meta', default='/dev/shm/dev_dataset.csv', type=str, \
            help='True labels for dev set')
    parser.add_argument('--num_samples', default=-1, type=int, \
            help='Number of samples, -1 for all samples')
    parser.add_argument('--train_batch_size', default=24, type=int, \
            help='How many images process at one time.')
    args = parser.parse_args()

    input_dir_abs = os.path.abspath(args.input_dir)

    if args.input_dir_mode == 'flat':
        df_meta = pd.read_csv(args.meta)
        df_meta = df_meta[['ImageId', 'TrueLabel']]
        df_meta['ImageId'] = os.path.abspath(args.input_dir) + '/' + df_meta['ImageId'] + '.png'
        df_meta['TrueLabel'] = df_meta['TrueLabel'] - 1
        df_meta = df_meta.set_index('ImageId')
        meta_dict = df_meta.to_dict()['TrueLabel']
        images_info = image_generators.get_names_and_labels(\
                args.input_dir, mode='flat', meta_dict=meta_dict)
    elif args.input_dir_mode == 'flat_targeted':
        df_meta = pd.read_csv(args.meta)
        df_meta = df_meta[['ImageId', 'TrueLabel', 'TargetClass']]
        df_meta['ImageId'] = os.path.abspath(args.input_dir) + '/' + df_meta['ImageId'] + '.png'
        df_meta['TrueLabel'] = df_meta['TrueLabel'] - 1
        df_meta['TargetClass'] = df_meta['TargetClass'] - 1
        df_meta = df_meta.set_index('ImageId')
        meta_dict = df_meta.to_dict()['TrueLabel']
        meta_target_dict = df_meta.to_dict()['TargetClass']
        images_info = image_generators.get_names_and_labels(\
                args.input_dir, mode='flat_targeted', meta_dict=meta_dict, meta_target_dict=meta_target_dict)
    elif args.input_dir_mode == 'test':
        images_info = image_generators.get_names_and_labels(\
                args.input_dir, mode='test')
    elif args.input_dir_mode == 'test_targeted':
        df_meta = pd.read_csv(os.path.join(os.path.abspath(args.input_dir), 'target_class.csv'), \
                header=None, names=['ImageId', 'TargetClass'])
        df_meta['TargetClass'] = df_meta['TargetClass'] - 1
        df_meta = df_meta.set_index('ImageId')
        meta_target_dict = df_meta.to_dict()['TargetClass']
        images_info = image_generators.get_names_and_labels(\
                args.input_dir, mode='test_targeted', meta_target_dict=meta_target_dict)
    else:
        images_info = image_generators.get_names_and_labels(\
                args.input_dir, mode='hierarchy')
        image_generators.initialize_hierarchy(args.output_dir)

    print('Total images:', len(images_info))

    eps = args.max_epsilon

    if args.num_samples < 0:
        num_samples = len(images_info)
    else:
        num_samples = args.num_samples

    is_targeted = ('targeted' in args.input_dir_mode)
    print('Is targeted', is_targeted)
    use_avg_pred = True

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
        lbls_mat = task_utils.get_raw_array(dims=lblmat_size) if not is_targeted else data[3]
        # data, min, max, grad, lbl, pseudo-label matrix
        placeholders[i] = (\
                data[0], \
                data_actual, \
                data_min, \
                data_max, \
                grad_mat, \
                data[2], \
                lbls_mat)

    print('Preparing functions')

    source_models = ['incresv2ensadv','resnet50', 'inceptionv3adv', 'inceptionv3']
    pred_models = ['incresv2ensadv','resnet50', 'inceptionv3adv']

    step_size = 1.8
    noise_size = 0.
    grad_aug_scale = 0.
    plan = [\
            (('RST', 'ir2ea-i3a-i3-r50-t', ('incresv2ensadv', 'inceptionv3adv', 'inceptionv3', 'resnet50')), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), ('inceptionv3', 1), step_size, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), ('inceptionv3', 1), step_size, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), ('inceptionv3', 1), step_size, noise_size), \
                (('incresv2ensadv', 1), step_size, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), ('inceptionv3', 1), step_size, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), step_size, noise_size), \
                (('incresv2ensadv', 1), step_size, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), step_size, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), step_size, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), step_size, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), step_size*0.5, noise_size), \
                (('incresv2ensadv', 1), step_size*0.5, noise_size), \
                (('incresv2ensadv', 1), step_size*0.5, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3', 1), step_size*0.5, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), step_size*0.5, noise_size), \
                (('incresv2ensadv', 1), step_size*0.5, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), ('inceptionv3', 1), step_size*0.25, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), ('inceptionv3', 1), step_size*0.25, noise_size), \
                (('incresv2ensadv', 1), step_size*0.25, noise_size), \
                (('incresv2ensadv', 1), step_size*0.25, noise_size), \
                (('incresv2ensadv', 1), step_size*0.25, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), ('inceptionv3', 1), step_size*0.25, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), ('inceptionv3', 1), step_size*0.25, noise_size), \
                (('incresv2ensadv', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3adv', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3', 1), step_size*0.1, noise_size), \
                (('incresv2ensadv', 1), ('inceptionv3', 1), step_size*0.1, noise_size)), \
            ]

    if not task_utils.verify_plan(source_models, pred_models, plan):
        return

    task_list = []
    phase_info = {}
    next_task = {}

    for i_phase, phase in enumerate(plan):
        last_task_in_batch = {}
        if phase[0][0] == 'RST':
            task = ('phase {0} restore from {1} model {2}'.format(i_phase, phase[0][1], phase[0][2]), \
                    (), 7, (phase[0][1],))
            task_list.append(task)
            phase_info[i_phase] = (len(task_list)-1, set(), set())
            # COPIED
            if (not is_targeted) and (i_phase==0):
                for phid, (fname_list, imgs, imgs_min, imgs_max, grads, lbls, lbls_mat) in placeholders.items():
                    task = ('batch {0}, fname [{1}], pred'.format(phid, ' '.join([os.path.split(f)[1][:-4] for f in fname_list])), \
                            (fname_list,imgs,None,grads,imgs_min,imgs_max,lbls_mat),3,(None,))
                    task_list.append(task)
                    last_task_in_batch[phid] = len(task_list)-1
                    phase_info[i_phase][1].add(len(task_list)-1)
                    phase_info[i_phase][2].add(len(task_list)-1)
        else:
            if i_phase:
                task = ('phase {0} reload {1}'.format(i_phase, phase[0]), (), 5, \
                        (phase[0][0], use_avg_pred, is_targeted))
                task_list.append(task)
                # id of loading task, all tasks other than load, first tasks after load
                phase_info[i_phase] = (len(task_list)-1, set(), set())
            else:
                task = ('phase {0} initial load {1} pred {2}'.format(i_phase, phase[0][0], phase[0][1]), (), 6, \
                        (phase[0][0], phase[0][1], use_avg_pred, is_targeted))
                task_list.append(task)
                phase_info[0] = (len(task_list)-1, set(), set())
                if not is_targeted:
                    for phid, (fname_list, imgs, imgs_min, imgs_max, grads, lbls, lbls_mat) in placeholders.items():
                        task = ('batch {0}, fname [{1}], pred'.format(phid, ' '.join([os.path.split(f)[1][:-4] for f in fname_list])), \
                                (fname_list,imgs,None,grads,imgs_min,imgs_max,lbls_mat),3,(None,))
                        task_list.append(task)
                        last_task_in_batch[phid] = len(task_list)-1
                        phase_info[i_phase][1].add(len(task_list)-1)
                        phase_info[i_phase][2].add(len(task_list)-1)
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

            task = ('phase {0} batch {1}, finish'.format(i_phase, phid), \
                    (fname_list,imgs,None,grads,imgs_min,imgs_max,lbls_mat),2,(len(input_dir_abs), args.output_dir))
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

    task_utils.run_tasks(task_list, phase_info, next_task, verbose=False, cpu_worker=1)

    print('Time:', time.time()-start_time)
    print('Total images:', len(images_info))
    print('Augmentation:', grad_aug_scale)
    print('\n'.join(str(u) for u in plan))
    print()


if __name__ == '__main__':
    main()

