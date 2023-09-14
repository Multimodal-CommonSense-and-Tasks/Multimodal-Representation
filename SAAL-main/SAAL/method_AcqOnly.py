import random
import copy
from utils_method import *

import pdb
from sklearn.metrics import pairwise_distances
from scipy import stats

def init_centers(X, K):
    X_array = np.expand_dims(X, 1)
    ind = np.argmax([np.linalg.norm(s, 2) for s in X_array])    # s should be array-like.
    mu = [X_array[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    # print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X_array, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X_array, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X_array[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X_array[indsAll], X_array[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return np.array(indsAll)

def acquire_AcqOnly(args):
    args.model.eval()

    # 1-1) Sampling
    print('...Acquisition Only')
    pool_subset_dropout = torch.from_numpy(
        np.asarray(random.sample(range(0, args.pool_data.size(0)), args.pool_subset))).long()
    pool_data_dropout = args.pool_data[pool_subset_dropout]
    pool_target_dropout = args.pool_target[pool_subset_dropout]

    predictions = []

    points_of_interest = args.acquisition_function(pool_data_dropout, pool_target_dropout, args)
    points_of_interest = points_of_interest.detach().cpu().numpy()

    ''''''
    if 'Diversity' in args.acqMode:
        pool_index = init_centers(points_of_interest, int(args.numQ))
    else:
        pool_index = np.flip(points_of_interest.argsort()[::-1][:int(args.numQ)], axis=0)
    ''''''

    pool_index = torch.from_numpy(pool_index)
    pooled_data = pool_data_dropout[pool_index]
    pooled_target = pool_target_dropout[pool_index]

    batch_size = pooled_data.shape[0]
    target1 = pooled_target.unsqueeze(1)
    y_onehot1 = torch.FloatTensor(batch_size, args.nb_classes)
    y_onehot1.zero_()
    target1_oh = y_onehot1.scatter_(1, target1, 1)
    pooled_target_oh = target1_oh.float()

    # 1-3) Remove from pool_data
    pool_data, pool_target = remove_pooled_points(args.pool_data, args.pool_target, pool_subset_dropout,
                                                  pool_data_dropout, pool_target_dropout, pool_index)
    args.pool_data = pool_data
    args.pool_target = pool_target
    args.pool_all = np.append(args.pool_all, pool_index)

    return pooled_data, pooled_target, pooled_target_oh