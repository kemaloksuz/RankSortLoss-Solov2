from functools import partial

import mmcv
import numpy as np
import torch
from six.moves import map, zip
import pdb

def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret

def vectorize_labels(flat_labels, num_classes, label_weights = None):
    prediction_number = flat_labels.shape[0]
    labels = torch.zeros([prediction_number, num_classes], device=flat_labels.device)
    pos_labels = flat_labels > 0
    labels[pos_labels, flat_labels[pos_labels]-1] = 1
    if label_weights is not None:
        ignore_labels = (label_weights == 0)
        labels[ignore_labels, :] = -1
    return labels.reshape(-1)
