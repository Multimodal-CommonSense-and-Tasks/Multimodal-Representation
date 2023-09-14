import time
import os
from scipy.stats import mode
import copy
import torch.utils.data as data_utils

from utils_data import *
from method_AcqOnly import *


def getAcquisitionFunction(labelMode):
    if labelMode == 'Pseudo':
        return max_sharpness_acquisition_pseudo
    else:
        raise ValueError('Wrong labelMode input!!!')
    # if labelMode == 'True':
    #     return max_sharpness_acquisition_true
    # if labelMode == 'InversePseudo':
    #     return max_sharpness_acquisition_inverse_pseudo

def acquire_points(args):

    args.pool_all = np.zeros(shape=(1))

    args.acquisition_function = getAcquisitionFunction(args.labelMode)

    args.test_acc_hist = []
    args.train_loss_hist = []

    for i in range(args.ai):
        st = time.time()
        print('---------------------------------')
        print("Acquisition Iteration " + str(i+1))
        print_number_of_data(args)

        print('(step1) Choose useful data')

        pooled_data, pooled_target, pooled_target_oh = acquire_AcqOnly(args)

        args.train_data = torch.cat([args.train_data, pooled_data], 0)
        args.train_target = torch.cat([args.train_target, pooled_target], 0)
        if args.data == 'Cifar10' or args.data == 'Cifar100':
            transform_train = transforms.Compose(
                [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        else:
            transform_train = None
        train_dataset = CustomTensorDataset(tensors=(args.train_data, args.train_target), dataname=args.data, transform=transform_train)
        train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        args.train_target_oh = torch.cat([args.train_target_oh, pooled_target_oh], 0)
        args.train_loader = train_loader

        total_train_loss = 0.0
        ####################################################
        """INITIALIZE CURRENT MODEL"""
        if args.isInit == 'True':
            print('...Initialize Classifier')
            init_model(args, first=False)
        ####################################################
        for epoch in range(args.epochs):
            args.layer_mix = None
            train_loss_labeled, num_batch_labeled, num_data_labeled = train(epoch, args)
            total_train_loss += train_loss_labeled/num_data_labeled
        total_train_loss /= args.epochs


        print('(step3) Results')
        args.layer_mix = None
        test_loss, test_acc, best_acc = test(i, args)
        args.test_acc_hist.append(test_acc)
        args.train_loss_hist.append(total_train_loss)
        et = time.time()

        print('...Train loss = %.5f' % total_train_loss)
        print('...Test accuracy = %.2f%%' % test_acc)
        print('...Best accuracy = %.2f%%' % args.best_acc)
        print_total_time(st, et)

    if args.isInit == 'True':
        np.save(args.saveDir + '/test_acc_init.npy', np.asarray(args.test_acc_hist))
        np.save(args.saveDir + '/train_loss_init.npy', np.asarray(args.train_loss_hist))
    else:
        np.save(args.saveDir + '/test_acc.npy', np.asarray(args.test_acc_hist))
        np.save(args.saveDir + '/train_loss.npy', np.asarray(args.train_loss_hist))

    return args.test_acc_hist



def max_sharpness_acquisition_pseudo(pool_data_dropout, pool_target_dropout, args):

    pool_pseudo_target_dropout = torch.tensor(torch.zeros(pool_data_dropout.size(0)))
    original_loss = []
    max_perturbed_loss = []

    data_size = pool_data_dropout.shape[0]
    num_batch = int(data_size / args.pool_batch_size)
    for idx in range(num_batch):
        batch = pool_data_dropout[idx*args.pool_batch_size : (idx+1)*args.pool_batch_size]

        output = args.model(batch.cuda())
        softmaxed = F.softmax(output.cpu(), dim=1)
        pseudo_target = np.argmax(softmaxed.data.numpy(), axis=-1)
        pseudo_target = torch.Tensor(pseudo_target).long()
        pool_pseudo_target_dropout[idx * args.pool_batch_size:(idx + 1) * args.pool_batch_size] = pseudo_target

        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(output, pseudo_target.cuda())
        original_loss.append(loss.cpu().detach().data)

    original_loss = torch.cat(original_loss, dim=0)


    for idx in range(num_batch):
        model_copy = copy.deepcopy(args.model)

        batch = pool_data_dropout[idx*args.pool_batch_size : (idx+1)*args.pool_batch_size]
        pseudo_target = pool_pseudo_target_dropout[idx*args.pool_batch_size : (idx+1)*args.pool_batch_size].long()

        output = model_copy(batch.cuda())
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss1 = criterion(output, pseudo_target.cuda())
        loss1.mean().backward()

        norm = torch.norm(
            torch.stack([(torch.abs(p)*p.grad).norm(p=2) for p in model_copy.parameters()]),
            p=2
        )
        scale = args.rho / (norm + 1e-12)
        with torch.no_grad():
            for p in model_copy.parameters():#named_paraeters()
                e_w = (torch.pow(p, 2)) * p.grad * scale.to(p)
                p.add_(e_w)

        output_updated = model_copy(batch.cuda())
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss2 = criterion(output_updated, pseudo_target.cuda())
        max_perturbed_loss.append(loss2.cpu().detach().data)

    max_perturbed_loss = torch.cat(max_perturbed_loss, dim=0)

    # return max_perturbed_loss, original_loss

    if args.acqMode == 'Max' or args.acqMode == 'Max_Diversity':
        return max_perturbed_loss
    if args.acqMode == 'Diff' or args.acqMode == 'Diff_Diversity':
        return max_perturbed_loss - original_loss
