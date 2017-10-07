import time
import multiprocessing as mp
import ctypes as c
import numpy as np
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
            # NOTE this part is not needed for defense
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

def proc_gpu_defense_task(task_list, task_q, fin_q, verbose=False):
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
                            [task[3][2]], 0])[0],\
                        out=get_array(task[1][3], imgmat_size)[...])
        elif task[2] == 3:
            results = pred_f[2]([get_array(task[1][1], imgmat_size)[...], [0.], 0])
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
            pred_f = grad_dict['PRED']
        if verbose or task[2] in [5,6,7]:
            print(task[0], ': Finished at', time.time()-start_time)
        fin_q.put(task_id)
    K.get_session().close()


def run_tasks(task_list, phase_info, next_task, \
        verbose=False, cpu_worker=1, is_defense=False):
    cpu_task_set = set([1,2,4]) if not is_defense else set([1,4])
    gpu_task_set = set([0,3,5,6,7])
    cpu_q = mp.Queue()
    gpu_q = mp.Queue()
    finished_q = mp.Queue()
    cpu_procs = [mp.Process(target=proc_cpu_task, \
            args=(task_list, cpu_q, finished_q), \
            kwargs={'verbose': verbose}) for _ in range(cpu_worker)]
    gpu_proc = mp.Process(\
            target=proc_gpu_task if not is_defense else proc_gpu_defense_task, \
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


