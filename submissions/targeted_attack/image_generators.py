import os
import multiprocessing as mp
import ctypes as c
import glob
import numpy as np
from PIL import Image
from multiprocessing import Pool

def initialize_hierarchy(tgt_dir):
    tgt_dir_abs = os.path.abspath(tgt_dir)
    if not os.path.exists(tgt_dir_abs):
        os.mkdir(tgt_dir_abs)
    else:
        if not os.path.isdir(tgt_dir_abs):
            raise FileExistsError(\
                    tgt_dir_abs + ' exists but is not a directory')
    for i in range(1000):
        class_dir = os.path.join(tgt_dir_abs, str(i))
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        else:
            if not os.path.isdir(class_dir):
                raise FileExistsError(\
                        class_dir + ' exists but is not a directory')

def save_image_from_uint8_array(img, fname):
    Image.fromarray(img).save(fname)

def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  # TODO decide on pool size
  with Pool(2) as p:
    p.starmap(save_image_from_uint8_array, \
            [(images[i,...], os.path.join(output_dir,filename)) \
            for i, filename in enumerate(filenames)])

#  for i, filename in enumerate(filenames):
#      save_image_from_uint8_array(images[i,...], os.path.join(output_dir,filename))
#      #Image.fromarray(images[i,...]).save(os.path.join(output_dir,filename))

def get_names_and_labels(img_dir, mode='flat', meta_dict=None, meta_target_dict=None):
    """
    Returns a list of tuples (img absolute location, label)

    Args:
      img_dir: directory where the images are
      mode: either 'flat' or 'hierarchy'
        'flat' means the directory is flat and true labels comes
          from a separate dictionary mapping relative file name to
          label
        'hierarchy' means the directory is organized into sub-dirs
          and the names of the sub-dirs are the labels
    """
    img_abs_dir = os.path.abspath(img_dir)
    if mode == 'flat':
        assert(meta_dict is not None)
        img_names = sorted(glob.glob(img_abs_dir + '/*'))
        ret = [(f, meta_dict[f]) for f in img_names]
        return ret
    elif mode == 'flat_targeted':
        assert(meta_dict is not None)
        assert(meta_target_dict is not None)
        img_names = sorted(glob.glob(img_abs_dir + '/*'))
        ret = [(f, meta_dict[f], meta_target_dict[f]) for f in img_names]
        return ret
    elif mode == 'test':
        img_names = sorted(glob.glob(img_abs_dir + '/*'))
        ret = [(f, None) for f in img_names]
        return ret
    elif mode == 'test_targeted':
        assert(meta_target_dict is not None)
        img_names = sorted([nms for nms in glob.glob(img_abs_dir + '/*') if not nms.endswith('.csv')])
        ret = [(f, None, meta_target_dict[os.path.split(f)[1]]) for f in img_names]
        return ret
    elif mode=='hierarchy':
        ret = []
        for root, dirs, files in os.walk(img_abs_dir):
            if root == img_abs_dir:
                continue
            lbl = int(root[len(img_abs_dir)+1:])
            ret += [(os.path.join(root,f), lbl) for f in files]
        return ret
    else:
        raise ValueError('Unknown mode')

def load_image_to_float_array(fname):
    img = Image.open(fname)
    return np.asarray(img, dtype=float)
    #return image.img_to_array(image.load_img(fname))

def nparray_to_rawarray(arr):
    raw_arr = mp.RawArray(c.c_double, int(np.prod(arr.shape)))
    np.frombuffer(raw_arr).reshape(arr.shape)[...] = arr
    return raw_arr

def load_images_into_batches(imgs_info, batch_size, input_dir, max_samples=None, to_buffer=False):
    # NOTE I don't think input_dir is being used, because the paths
    # coming from get_names_and_labels are absolute
    num_imgs = len(imgs_info)
    index_limit = num_imgs
    if max_samples is not None:
        index_limit = min(max_samples, index_limit)
    is_targeted = (len(imgs_info[0])==3)
    img_names = [u[0] for u in imgs_info[:index_limit]]
    img_lbls = np.array([u[1] for u in imgs_info[:index_limit]])
    if is_targeted:
        #tgt_lbls = np.array([u[2] for u in imgs_info[:index_limit]])
        tgt_lbls = np.ones((index_limit, 1000))*(1e-5)
        tgt_lbls[np.arange(index_limit), [u[2] for u in imgs_info[:index_limit]]] = 0.99001
    base_dir_abs = os.path.abspath(input_dir)
    with Pool(4) as p:
        imgs = p.map(load_image_to_float_array, img_names[:index_limit])

    imgs = np.array(imgs, dtype=float)
    img_labels = img_lbls[:index_limit]
    if not to_buffer:
        return [(\
                img_names[batch_start:batch_start+batch_size], \
                imgs[batch_start:batch_start+batch_size,...], \
                img_labels[batch_start:batch_start+batch_size], \
                None if not is_targeted else tgt_lbls[batch_start:batch_start+batch_size, ...]) \
                for batch_start in range(0, index_limit, batch_size)]
    else:
        return [(\
                img_names[batch_start:batch_start+batch_size], \
                nparray_to_rawarray(imgs[batch_start:batch_start+batch_size,...]), \
                img_labels[batch_start:batch_start+batch_size], \
                None if not is_targeted else nparray_to_rawarray(tgt_lbls[batch_start:batch_start+batch_size, ...])) \
                for batch_start in range(0, index_limit, batch_size)]

def image_generator(imgs_info, batch_size, input_dir, adv_dir=None, max_samples=None):
    """
    iterator yields:
      current total
      path to clean image
      clean image
      true labels, followed by targeted labels if provided
      adversarial image if provided adv_dir
    Note that all image arrays are float values in [0,255] from keras

    Args:
      imgs_info: a list of tuples (img abs location, label)
      batch_size: batch size
      input_dir: the base directory of the clean inputs
        this is useful when we need to evaluate on both clean and
        adversarial images
      adv_dir: the base directory of adversarial images
    """
    is_targeted = (len(imgs_info[0])==3)
    num_imgs = len(imgs_info)
    img_names = [u[0] for u in imgs_info]
    img_lbls = np.array([u[1] for u in imgs_info])
    if is_targeted:
        tgt_lbls = np.array([u[2] for u in imgs_info])
    index_limit = num_imgs
    if max_samples is not None:
        index_limit = min(max_samples, index_limit)
    idx = 0
    total = 0
    # calculate base dir for input so we can substitute for adv if needed
    base_dir_abs = os.path.abspath(input_dir)
    if adv_dir is not None:
        adv_dir_abs = os.path.abspath(adv_dir)
    while True:
        imgs = []
        names_list = img_names[idx:min(index_limit,idx+batch_size)]
        with Pool(4) as p:
            imgs = p.map(load_image_to_float_array, names_list)
        imgs = np.array(imgs, dtype=float)
        if adv_dir is not None:
            adv_imgs = []
            with Pool(4) as p:
                adv_imgs = p.map(load_image_to_float_array, \
                        [os.path.join(adv_dir_abs, f[len(base_dir_abs)+1:]) for f in names_list])
            adv_imgs = np.array(adv_imgs, dtype=float)
        total += imgs.shape[0]
        ret = [total, names_list, imgs, img_lbls[idx:min(index_limit,idx+batch_size)]]
        if is_targeted:
            ret.append(tgt_lbls[idx:min(index_limit,idx+batch_size)])
        if adv_dir is not None:
            ret.append(adv_imgs)
        yield ret
        idx += batch_size
        if idx >= index_limit:
            total = 0
            idx = 0

