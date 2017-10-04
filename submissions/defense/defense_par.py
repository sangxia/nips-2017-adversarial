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
    # very important, not sure why
    # worked for load_ckpt probably because 
    # those tensors are marked as initialized by keras
    # and after that init_fn is used?
    # whereas if backend function is initialized directly
    # there is get_session and this clears up all previously
    # loaded weights
    K.manual_variable_initialization(True)
    import network_utils
    pred_f = None
    grad_dict = {}
    for task_id in iter(task_q.get, -1):
        task = task_list[task_id]
        if verbose:
            print(task[0])
        if task[2] != 5 and task[2] != 6 and task[2] != 7:
            batch_size = len(task[1][0])
            imgmat_size = (batch_size, 299, 299, 3)
            lblmat_size = (batch_size, 1000)
        if task[2] == 0:
            for _ in range(task[3][3]):
                _ = np.add(get_array(task[1][3], imgmat_size)[...],\
                        grad_dict[task[3][1]]([get_array(task[1][1], imgmat_size)[...],[task[3][2]],\
                        get_array(task[1][6],lblmat_size)[...],0])[0],\
                        out=get_array(task[1][3], imgmat_size)[...])
        elif task[2] == 3:
            results = pred_f([get_array(task[1][1], imgmat_size)[...], [0.], 0])
            tmp = (results[0] == np.max(results[0], axis=1, keepdims=True)).astype(float)
            batch_labels = (tmp / np.sum(tmp, axis=1, keepdims=True))
            get_array(task[1][6], lblmat_size)[...].put(np.arange(0,np.prod(lblmat_size)), batch_labels)
        elif task[2] == 5 or task[2] == 6:
            if task[2] == 5:
                K.clear_session()
            else:
                grad_dict = {}
            _ = network_utils.load_defense_model(\
                models=task[3][0], \
                dist_pairs=task[3][1], \
                grad_dict=grad_dict)
            pred_f = grad_dict['PRED']
        elif task[2] == 7:
            # TODO there is a bug in prepare model. many times keras learning phase shouldn't be in the output
            # so if we don't clear session this learning phase variable will not be found
            # one side effect of k.clear session is that this tensor gets created
            K.clear_session()
            K.get_session().close()
            grad_dict = {}
            network_utils.restore_source_model(task[3][0], grad_dict=grad_dict, from_frozen=True)
            pred_f = grad_dict['PRED']
        if verbose:
            print(task[0], ': Finished at', time.time()-start_time)
        fin_q.put(task_id)
    K.get_session().close()

def run_tasks(task_list, phase_info, next_task, verbose=False, cpu_worker=1):
    cpu_task_set = set([1,4])
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
    # TODO adapt this for defense plans
    if not all(m in source_models for m in pred_models):
        print('Not all pred_models are in source_models.')
        return False
    for i,pl in enumerate(plan):
        if pl[0][0] == 'RST':
            pl_models = set(pl[0][2])
        elif i==0:
            pl_models = set(pl[0][0])
        else:
            pl_models = set(pl[0])
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
    parser.add_argument('--output_file', required=True, type=str, \
            help='Output directory with images.')
    parser.add_argument('--input_dir_mode', default='test', type=str, \
            help='Either flat or hierarchy, how the input dir is organised')
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
        df_meta = pd.read_csv(os.path.join(obs.path.abspath(args.input_dir), 'target_class.csv'), \
                header=None, names=['ImageId', 'TargetClass'])
        df_meta = df_meta.set_index('ImageId')
        meta_target_dict = df_meta.to_dict()['TargetClass']
        images_info = image_generators.get_names_and_labels(\
                args.input_dir, mode='test_targeted', meta_target_dict=meta_target_dict)
    else:
        images_info = image_generators.get_names_and_labels(\
                args.input_dir, mode='hierarchy')
        image_generators.initialize_hierarchy(args.output_dir)

    print('Total images:', len(images_info))

    eps = 8.

    num_samples = len(images_info)

    use_avg_pred = True
    step_size = eps
    noise_size = eps*0.2
    grad_aug_scale = 0.02

    print('Loading images')
    # each image_data element: filename, img, lbl
    image_data = image_generators.load_images_into_batches(\
            images_info, args.train_batch_size, args.input_dir, \
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
        lbl_mat = get_raw_array(dims=lblmat_size)
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
            #((source_models, dist_pairs), \
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
    run_tasks(task_list, phase_info, next_task, verbose=False, cpu_worker=1)

    with open(args.output_file, 'w') as f:
        for phid, (fnames, _, _, _, _, _, lbls_mat) in placeholders.items():
            batch_preds = get_array(lbls_mat, (len(fnames),1000)).argmax(axis=1)
            for img, pred in zip(fnames, batch_preds):
                _ = f.write('{0},{1}\n'.format(os.path.split(img)[1], pred+1))

    print('Time:', time.time()-start_time)
    print('\n'.join(str(u) for u in plan))
    print()

if __name__ == '__main__':
    main()

