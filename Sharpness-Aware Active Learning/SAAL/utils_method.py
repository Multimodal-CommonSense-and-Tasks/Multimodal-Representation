import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.distributions as distr
from utils_data import CustomTensorDataset
import torch.utils.data as data_utils
from torchvision import transforms


def remove_pooled_points(pool_data, pool_target, pool_subset_dropout, pool_data_dropout, pool_target_dropout, pool_index):

    np_data = pool_data.numpy()
    np_target = pool_target.numpy()

    pool_data_dropout = pool_data_dropout.numpy()
    pool_target_dropout = pool_target_dropout.numpy()

    pool_subset_idx = pool_subset_dropout.numpy()
    np_index = pool_index.numpy()

    np_data = np.delete(np_data, pool_subset_idx, axis=0)
    np_target = np.delete(np_target, pool_subset_idx, axis=0)

    pool_data_dropout = np.delete(pool_data_dropout, np_index, axis=0)
    pool_target_dropout = np.delete(pool_target_dropout, np_index, axis=0)

    np_data = np.concatenate((np_data, pool_data_dropout), axis=0)
    np_target = np.concatenate((np_target, pool_target_dropout), axis=0)
    pool_data = torch.from_numpy(np_data)
    pool_target = torch.from_numpy(np_target)

    return pool_data, pool_target

def evaluate_featuremap(featuremap, args):
    args.model.eval()

    predictions = []
    featuremap = featuremap.cuda()
    output = args.model.forward_rest(featuremap, args)
    softmaxed = F.softmax(output, dim=1)
    predictions.append(softmaxed)
    predictions = torch.cat(predictions, dim=0)

    return predictions

def print_total_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    elapsed_hours = int(elapsed_mins / 60)
    elapsed_mins = int(elapsed_mins - (elapsed_hours * 60))

    if elapsed_hours == 0:
        print('...%dmin %dsec'%(elapsed_mins, elapsed_secs))
    if elapsed_hours != 0:
        print('...%dhour %dmin %dsec'%(elapsed_hours, elapsed_mins, elapsed_secs))

def print_memory_usage():
    t = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    c = torch.cuda.memory_cached(0) / (1024 ** 2)
    a = torch.cuda.memory_allocated(0) / (1024 ** 2)
    f = c - a
    print(t, c, a, f)

def print_number_of_data(args):
    print('...Train data : %d' % len(args.train_data))
    print('...Pool data : %d' % len(args.pool_data))
    print('...Test data : %d' % len(args.test_loader.dataset))

