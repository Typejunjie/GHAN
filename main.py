import time
import argparse
import pickle
from model import *
from utils import *
from tensorboardX import SummaryWriter


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/Nowplaying/Tmall')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=12)
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')

opt = parser.parse_args()



def main():

    if opt.dataset == 'diginetica':
        init_seed(2020)
        num_node = 43098
        opt.n_iter = 2
        opt.dropout_gcn = 0.2
        opt.dropout_local = 0.0
    elif opt.dataset == 'Nowplaying':
        init_seed(2020)
        num_node = 60417
        opt.n_iter = 1
        opt.dropout_gcn = 0.0
        opt.dropout_local = 0.0
    elif opt.dataset == 'Tmall':
        init_seed(2021)
        num_node = 40728
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.5
    elif opt.dataset == 'yoochoose1_64':
        init_seed(2020)
        num_node = 37484
        opt.n_iter = 1
        opt.dropout_gcn = 0.5
        opt.dropout_local = 0.5
    elif opt.dataset == 'RetailRocket':
        init_seed(2020)
        num_node = 36969
        opt.n_iter = 1
        opt.dropout_gcn = 0.5
        opt.dropout_local = 0.0
    else:
        raise KeyError

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))

    global_train = pickle.load(open(f'./datasets/{opt.dataset}/global_train.txt', 'rb'))
    global_test = pickle.load(open(f'./datasets/{opt.dataset}/global_test.txt', 'rb'))
    train_data = Data(train_data, global_train)
    test_data = Data(test_data, global_test)

    model = trans_to_cuda(CombineGraph(opt, num_node))

    logdir = f'./log/{opt.dataset}'
    writer = SummaryWriter(logdir + '-' + opt.version)

    print(opt)
    start = time.time()
    topk = [5, 10, 20]
    best_result = [[0, 0] for i in topk]
    best_epoch = [[0, 0] for i in topk]
    bad_counter = [0 for i in topk]

    for epoch in tqdm(range(opt.epoch), colour='#6A5ACD', desc='Training model', leave=False):
        tqdm.write('-------------------------------------------------------')
        tqdm.write(f'epoch: {epoch + 1}')
        hit, mrr = train_test(model, train_data, test_data, writer, epoch + 1, topk)
        for index, i in enumerate(topk):
            flag = 0
            if hit[index] >= best_result[index][0]:
                best_result[index][0] = hit[index]
                best_epoch[index][0] = epoch + 1
                flag = 1
            if mrr[index] >= best_result[index][1]:
                best_result[index][1] = mrr[index]
                best_epoch[index][1] = epoch + 1
                flag = 1
            bad_counter[index] += 1 - flag

        if max(bad_counter) >= opt.patience:
            break
        tqdm.write('Best Result:')
        for index, i in enumerate(topk):
            tqdm.write(f'\tRecall@{i}:\t%.4f\tMMR@{i}:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[index][0], best_result[index][1], best_epoch[index][0], best_epoch[index][1]))

    print('-------------------------------------------------------')
    writer.close()
    end_time = time.time()
    total_seconds = end_time - start
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    print(f"Total duration {str(int(hours)) + 'h' if hours > 0 else ''} {str(int(minutes)) + 'min' if minutes > 0 else ''} {int(seconds)}s")
    print('Done')


if __name__ == '__main__':
    main()
