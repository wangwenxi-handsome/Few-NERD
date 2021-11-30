import argparse


def init_parser():
    # 读取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='inter',
            help='training mode, must be in [inter, intra, supervised]')
    parser.add_argument('--trainN', default=5, type=int,
            help='N in train')
    parser.add_argument('--N', default=5, type=int,
            help='N way')
    parser.add_argument('--K', default=1, type=int,
            help='K shot')
    parser.add_argument('--Q', default=1, type=int,
            help='Num of query per class')
    parser.add_argument('--batch_size', default=1, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=20000, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=500, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=5000, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=1000, type=int,
           help='val after training how many iters')
    parser.add_argument('--lr', default=1e-4, type=float,
           help='learning rate')
    parser.add_argument('--grad_iter', default=1, type=int,
           help='accumulate gradient every x iterations')
    parser.add_argument('--loss', type=str, default="ce",
           help='use released sampled data, the data should be stored at "data/episode-data/" ')

    # for span level
    parser.add_argument('--max_length', default=512, type=int,
           help='max length')
    parser.add_argument('--max_span_length', default=30, type=int,
           help='Max length of span')  
    parser.add_argument('--support_span_num', default=1000, type=int,
           help='support span num')  
    parser.add_argument('--query_span_num', default=1000, type=int,
           help='query span num')  

    # only for bert / roberta
    parser.add_argument('--pretrain_ckpt', default=None,
           help='bert / roberta pre-trained checkpoint')
    # only for prototypical networks
    parser.add_argument('--dot', action='store_true', 
           help='use dot instead of L2 distance for proto')
    # # train related
    parser.add_argument('--use_sgd_for_bert', action='store_true',
           help='use SGD instead of AdamW for BERT.')
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
           help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true',
           help='only test')
    parser.add_argument('--ckpt_name', type=str, default='',
           help='checkpoint name.')
    parser.add_argument('--seed', type=int, default=0,
           help='random seed')
    parser.add_argument('--use_sampled_data', type=bool, default=True,
           help='use released sampled data, the data should be stored at "data/episode-data/" ')
    return parser