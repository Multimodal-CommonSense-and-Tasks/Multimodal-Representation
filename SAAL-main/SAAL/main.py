from __future__ import print_function
import argparse
import os

from utils_data import *
from utils_al import *

parser = argparse.ArgumentParser(description='Active Learning with Data Augmentation')

parser.add_argument('--isSimpleTest', default=False, type=bool, help='lenet if True / ResNet if False')
parser.add_argument('--data_scale', default='Small', type=str, help='data scale / small, large')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs before acquiring points')
parser.add_argument('--model', default=None, help='classifier model')
parser.add_argument('--optimizer', default=None, help='optimizer of classifier')
parser.add_argument('--criterion', default=None, help='criterion of classifier')
parser.add_argument('--model_scheduler', default=None, help='model scheduler')
parser.add_argument('--best_acc', default=0., type=float, help='best accuracy')
parser.add_argument('--batch_size', default=10, type=int, help='batch_size')
parser.add_argument('--pool_batch_size', default=1, type=int, help='batch_size for calculating score in pool data')
parser.add_argument('--test_batch_size', default=100, type=int, help='test_batch_size')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1 for learning rate')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for learning rate')
parser.add_argument('--ai', default=100, type=int, help='acquisition_iterations')
parser.add_argument('--numQ', default=10, type=int, help='number of queries')
parser.add_argument('--pool_subset', default=2000, type=int, help='number of pool')

parser.add_argument('--isInit', default='False', type=str, help='True=Initialize / False=NoInitialize')
parser.add_argument('--rs', default=123, type=int, help='random seed / 123, 456, 789, 135, 246')
parser.add_argument('--data', default='Cifar10', type=str, help='data name / Fashion, SVHN, Cifar10, Cifar100')
parser.add_argument('--acqMode', default='Max_Diversity', type=str, help='acquisition mode / Max (max_perturbed_loss), Diff (max_perturbed_loss-original_loss), Max_Diversity, Diff_Diversity')
parser.add_argument('--labelMode', default='Pseudo', type=str, help='label mode / True (sharpness with true label), Pseudo (sharpness with pseudo label), InversePseudo (sharpness with inverse pseudo label')
parser.add_argument('--optimizer_name', default='Adam', type=str, help='Adam / SAM')
parser.add_argument('--rho', default=0.05, type=float, help='sharpness computation parameter')
args = parser.parse_args()

torch.set_printoptions(precision=10)

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

rs = args.rs
seed_torch(rs)

##################################
# Is simple test?
##################################
# args.epochs = 3
# args.ai = 5
# args.numQ = 10

cuda = True

args.init_budget = 20
args.numQ = 10
args.ai = 100
args.batch_size = 10
if args.data == 'Cifar100':
    args.init_budget = 1000
    args.numQ = 100
    args.pool_subset = 10000
    args.ai = 100
    args.batch_size = 100

if args.data == 'Fashion':
    args.input_dim, args.input_height, args.input_width = 1, 28, 28
    args.nb_classes = 10
elif args.data == 'SVHN' or args.data == 'Cifar10':
    args.input_dim, args.input_height, args.input_width = 3, 32, 32
    args.nb_classes = 10
elif args.data == 'Cifar100':
    args.input_dim, args.input_height, args.input_width = 3, 32, 32
    args.nb_classes = 100
    args.pool_subset = 10000
else:
    raise ValueError('Wrong data input!!!')

args.init_num_per_class = int(args.init_budget / args.nb_classes)


kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

##################################
# Loading Data
##################################
train_loader_all, test_loader, trainset_all, testset = make_loader(args, kwargs)
train_data, train_target, pool_data, pool_target, test_data, test_target = prepare_data(train_loader_all, test_loader, trainset_all, testset, args)

args.train_data = train_data
args.pool_data = pool_data
args.pool_target = pool_target

##################################
# Run file
##################################
def main():
    start_time = time.time()

    args.saveDir = os.path.join('Results', args.optimizer_name, args.labelMode, args.acqMode, args.data_scale, args.data, str(args.rs))
    args.modelDir = os.path.join('Models', args.optimizer_name, args.labelMode, args.acqMode, args.data_scale, args.data, str(args.rs))

    if not os.path.exists(args.saveDir):
        print('Result Directory Constructed Successfully!!!')
        os.makedirs(args.saveDir)
    if not os.path.exists(args.modelDir):
        print('Model Directory Constructed Successfully!!!')
        os.makedirs(args.modelDir)

    train_loader, test_loader = initialize_train_set(train_data, train_target, test_data, test_target, args)

    args.train_loader, args.train_data, args.train_target = train_loader, train_data, train_target
    args.test_loader, args.test_data, args.test_target = test_loader, test_data, test_target

    batch_size = args.train_data.shape[0]
    target = train_target.unsqueeze(1)
    y_onehot = torch.FloatTensor(batch_size, args.nb_classes)
    y_onehot.zero_()
    target_oh = y_onehot.scatter_(1, target, 1)
    args.train_target_oh = target_oh.float()

    args.layer_mix = None
    init_model(args, first=True)

    print_number_of_data(args)

    print("<Training without acquisition>")
    for epoch in range(args.epochs):
        loss, num_batch, num_data = train(epoch, args)
        if (epoch+1)%10 == 0:
            print(loss / num_data)
        test_loss, test_acc, best_acc = test(epoch, args)
        if epoch == args.epochs-1:
            print('Test Loss = %f\n' % (test_loss))
            print('Test Accuracy = %.2f%%\n' % (test_acc))
            print('Best Accuracy = %.2f%%\n' % (best_acc))


    print("<Acquiring points>")
    test_acc_hist = acquire_points(args)

    print('=========================')
    end_time = time.time()
    print_total_time(start_time, end_time)
    print('\n')

if __name__ == '__main__':
    main()



