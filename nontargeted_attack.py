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

def get_array(raw_arr, dims):
    return np.frombuffer(raw_arr).reshape(dims)


def get_raw_array(dims=None, init=None):
    if dims is not None:
        return mp.RawArray(c.c_double, int(np.prod(dims)))
    if init is not None:
        arr = mp.RawArray(c.c_double, int(np.prod(init.shape)))
        get_array(arr, init.shape)[...] = init
        return arr
    raise ValueError


def proc_cpu_task(task_list, task_q, fin_q, verbose=False):
    for task_id in iter(task_q.get, -1):
        task = task_list[task_id]
        if verbose:
            print(task[0])
        batch_size = len(task[1][0])
        imgmat_size = (batch_size, 299, 299, 3)
        lblmat_size = (batch_size, 1000)
        if task[2] == 1:
            _ = np.clip(get_array(task[1][1], imgmat_size)[...] - np.sign(get_array(task[1][3], imgmat_size)[...])*task[3][0], \
                    get_array(task[1][4], imgmat_size)[...], get_array(task[1][5], imgmat_size)[...], out=get_array(task[1][1], imgmat_size)[...])
            get_array(task[1][3], imgmat_size)[...].fill(0.)
        elif task[2] == 2: 
            adv_imgs = get_array(task[1][1], imgmat_size)[...].astype(np.uint8)
            image_generators.save_images(adv_imgs, \
                    [fname[task[3][0]+1:] for fname in task[1][0]], \
                    task[3][1])
        elif task[2] == 4:
            _ = np.clip(get_array(task[1][1], imgmat_size)[...]+np.random.normal(scale=task[3][0], size=imgmat_size), \
                    get_array(task[1][4], imgmat_size)[...], get_array(task[1][5], imgmat_size)[...], out=get_array(task[1][1], imgmat_size)[...])
        if verbose:
            print(task[0], ': Finished at', time.time()-start_time)
        fin_q.put(task_id)


def proc_gpu_task(task_list, task_q, fin_q, verbose=False):
    import tensorflow as tf
    tf.set_random_seed(1)
    from keras import backend as K
    K.manual_variable_initialization(True) # NOTE very important
    import network_utils
    pred_f = None
    grad_dict = {}
    for task_id in iter(task_q.get, -1):
        task = task_list[task_id]
        if verbose or task[2] in [5,6,7]:
            print(task[0])
        if task[2] != 5 and task[2] != 6 and task[2] != 7:
            batch_size = len(task[1][0])
            imgmat_size = (batch_size, 299, 299, 3)
            lblmat_size = (batch_size, 1000)
        if task[2] == 0:
            for _ in range(task[3][3]):
                _ = np.add(get_array(task[1][3], imgmat_size)[...],\
                        grad_dict[task[3][1]][2]([\
                            get_array(task[1][1], imgmat_size)[...], \
                            [task[3][2]], \
                            get_array(task[1][6],lblmat_size)[...],0])[0],\
                        out=get_array(task[1][3], imgmat_size)[...])
        elif task[2] == 3:
            results = pred_f[2]([get_array(task[1][1], imgmat_size)[...], [0.], 0])
            tmp = (results[0] == np.max(results[0], axis=1, keepdims=True)).astype(float)
            batch_labels = (tmp / np.sum(tmp, axis=1, keepdims=True))
            get_array(task[1][6], lblmat_size)[...].put(np.arange(0,np.prod(lblmat_size)), batch_labels)
        elif task[2] == 5:
            K.clear_session()
            _ = network_utils.load_source_model(\
                models=task[3][0], \
                pred_models=None, \
                is_targeted=task[3][2], \
                grad_dict=grad_dict)
        elif task[2] == 6:
            grad_dict = {}
            _ = network_utils.load_source_model(\
                    models=task[3][0], \
                    pred_models=task[3][1], \
                    is_targeted=task[3][3], \
                    grad_dict=grad_dict)
            pred_f = grad_dict['PRED']
        elif task[2] == 7:
            # NOTE must reset graph, otherwise old ops will stay around
            # causing problem when restoring from .pb
            # 
            # K.clear_session() is not enough because then learning_phase
            # will be created and that will cause problems if there is
            # another learning_phase tensor in the saved graph.
            # although the current approach could cause problem if one 
            # plans to use K.learning_phase later
            # 
            # however, in the submission code, around here it is clear_session
            # followed by close session, but no reset graph, and for some
            # reason that didn't cause problem with feed_dict
            # but if we do clear, reset and close here, we get
            # tensor not found. 
            #
            # in brief, it is a mess
            K.clear_session()
            tf.reset_default_graph()
            grad_dict = {}
            network_utils.restore_source_model(task[3][0], grad_dict=grad_dict)
            if 'PRED' in grad_dict:
                pred_f = grad_dict['PRED']
        if verbose or task[2] in [5,6,7]:
            print(task[0], ': Finished at', time.time()-start_time)
        fin_q.put(task_id)
    K.get_session().close()


def run_tasks(task_list, phase_info, next_task, verbose=False, cpu_worker=1):
    cpu_task_set = set([1,2,4])
    gpu_task_set = set([0,3,5,6,7])
    cpu_q = mp.Queue()
    gpu_q = mp.Queue()
    finished_q = mp.Queue()
    cpu_procs = [mp.Process(target=proc_cpu_task, \
            args=(task_list, cpu_q, finished_q), \
            kwargs={'verbose': verbose}) for _ in range(cpu_worker)]
    gpu_proc = mp.Process(target=proc_gpu_task, \
            args=(task_list, gpu_q, finished_q), \
            kwargs={'verbose': verbose})
    [p.start() for p in cpu_procs]
    gpu_proc.start()
    for i in range(len(phase_info)):
        print('Phase', i)
        ph = phase_info[i]
        gpu_q.put(ph[0])
        for tid in iter(finished_q.get, ph[0]):
            # NOTE: this part should never be executed
            time.sleep(.1)
            continue
        for tid in ph[2]:
            if task_list[tid][2] in cpu_task_set:
                cpu_q.put(tid)
            else:
                gpu_q.put(tid)
        for tid in iter(finished_q.get, -1):
            ph[1].remove(tid)
            if tid in next_task:
                if task_list[next_task[tid]][2] in cpu_task_set:
                    cpu_q.put(next_task[tid])
                else:
                    gpu_q.put(next_task[tid])
            if len(ph[1]) == 0:
                break
    [cpu_q.put(-1) for _ in range(cpu_worker)]
    gpu_q.put(-1)
    gpu_proc.join()
    [p.join() for p in cpu_procs]


def verify_plan(source_models, pred_models, plan):
    if not all(m in source_models for m in pred_models):
        print('Not all pred_models are in source_models.')
        return False
    for i,pl in enumerate(plan):
        if pl[0][0] == 'RST':
            pl_models = set(pl[0][2])
        elif i==0:
            pl_models = set(pl[0][0])
        else:
            pl_models = set(pl[0][0])
        if pl[0][0] != 'RST' and i==0:
            if set(pl[0][0]).symmetric_difference(source_models):
                print('source_models not equal to plan[0][0]')
                return False
            if set(pl[0][1]).symmetric_difference(pred_models):
                print('pred_models not equal to plan[0][1]')
                return False
        for j,stp in enumerate(pl):
            if j==0:
                continue
            if not all(m in pl_models for m,_ in stp[:-2]):
                print('Model not loaded in phase {0} step {1}'.format(i,j))
                return False
    return True


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
        data_min = get_raw_array(init=np.clip(get_array(data[1], imgmat_size)-eps, 0, None))
        data_max = get_raw_array(init=np.clip(get_array(data[1], imgmat_size)+eps, None, 255))
        data_actual = data[1]
        grad_mat = get_raw_array(dims=imgmat_size)
        lbls_mat = get_raw_array(dims=lblmat_size) if not is_targeted else data[3]
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

    step_size = eps
    noise_size = eps*0.2
    grad_aug_scale = 0.02
    min_step = 0.
    min_noise = 0.

    plan = [\
            (('RST', 'ir2ea-i3a-i3-r50-nt', source_models), \
            # ((source_models, pred_models), \
                (('resnet50', 2), ('resnet50', 2), step_size, noise_size), \
                (('inceptionv3adv', 2), ('incresv2ensadv', 2), step_size, noise_size), \
                (('inceptionv3adv', 2), ('incresv2ensadv', 2), step_size*0.5, noise_size), \
                (('inceptionv3adv', 2), ('incresv2ensadv', 2), step_size*0.5, noise_size), \
                (('inceptionv3adv', 2), ('incresv2ensadv', 2), step_size*0.5, noise_size)), \
            (('RST', 'i3-x-ir2-nt', ['inceptionv3','incresv2','xception']), \
            # ((['inceptionv3','incresv2','xception'], None), \
                (('inceptionv3', 2), ('incresv2', 2), step_size*0.5, noise_size), \
                (('inceptionv3', 2), ('xception', 2), step_size*0.5, noise_size), \
                (('incresv2', 2), ('xception', 2), step_size*0.5, noise_size), \
                (('inceptionv3', 2), ('xception', 2), step_size*0.5, noise_size)), \
            (('RST', 'ir2ea-r101-i3a-nt', ['incresv2ensadv', 'resnet101', 'inceptionv3adv']), \
            # ((['incresv2ensadv', 'resnet101', 'inceptionv3adv'], None), \
                (('incresv2ensadv', 2), ('resnet101', 2), step_size*0.2, noise_size*0.5), \
                (('inceptionv3adv', 2), ('resnet101', 2), step_size*0.2, noise_size*0.5), \
                (('inceptionv3adv', 1), ('incresv2ensadv', 1), max(min_step, step_size*0.1), max(min_noise, noise_size*0.5)), \
                (('inceptionv3adv', 1), ('incresv2ensadv', 1), max(min_step, step_size*0.1), max(min_noise, noise_size*0.5)), \
                (('inceptionv3adv', 1), ('incresv2ensadv', 1), max(min_step, step_size*0.1), max(min_noise, noise_size*0.2)), \
                (('incresv2ensadv', 1), ('incresv2ensadv', 1), max(min_step, step_size*0.1), max(min_noise, noise_size*0.2)), \
                (('incresv2ensadv', 1), ('resnet101', 1),      max(min_step, step_size*0.1), max(min_noise, noise_size*0.2)), \
                (('incresv2ensadv', 1), ('incresv2ensadv', 1), max(min_step, step_size*0.1), max(min_noise, noise_size*0.2)), \
                (('incresv2ensadv', 1), ('incresv2ensadv', 1), max(min_step, step_size*0.1), max(min_noise, noise_size*0.2)), \
                (('inceptionv3adv', 2), ('incresv2ensadv', 2), max(min_step, step_size*0.1), max(min_noise, noise_size*0.2))), \
            ]

    if not verify_plan(source_models, pred_models, plan):
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

    run_tasks(task_list, phase_info, next_task, verbose=False, cpu_worker=1)

    print('Time:', time.time()-start_time)
    print('Total images:', len(images_info))
    print('Augmentation:', grad_aug_scale)
    print('\n'.join(str(u) for u in plan))
    print()


if __name__ == '__main__':
    main()

