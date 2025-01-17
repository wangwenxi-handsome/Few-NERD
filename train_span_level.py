from transformers import BertTokenizer
import torch
import numpy as np
import argparse
import os
import time
import torch
import random
from util.word_encoder import BERTWordEncoder
from span_level.dataloader import get_loader
from span_level.framework import FewShotNERFramework
from span_level.model import SpanNNShot

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # 读取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='inter',
            help='training mode, must be in [inter, intra, supervised]')
    parser.add_argument('--trainN', default=2, type=int,
            help='N in train')
    parser.add_argument('--N', default=2, type=int,
            help='N way')
    parser.add_argument('--K', default=2, type=int,
            help='K shot')
    parser.add_argument('--Q', default=3, type=int,
            help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=600, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=100, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=500, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=20, type=int,
           help='val after training how many iters')
    parser.add_argument('--model', default='proto',
            help='model name, must be basic-bert, proto, nnshot, or structshot')
    parser.add_argument('--max_length', default=100, type=int,
           help='max length')
    parser.add_argument('--lr', default=1e-4, type=float,
           help='learning rate')
    parser.add_argument('--grad_iter', default=1, type=int,
           help='accumulate gradient every x iterations')
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
    parser.add_argument('--ignore_index', type=int, default=-1,
           help='label index to ignore when calculating loss and metrics')
    parser.add_argument('--use_sampled_data', action='store_true',
           help='use released sampled data, the data should be stored at "data/episode-data/" ')

    # only for bert / roberta
    parser.add_argument('--pretrain_ckpt', default=None,
           help='bert / roberta pre-trained checkpoint')

    # only for prototypical networks
    parser.add_argument('--dot', action='store_true', 
           help='use dot instead of L2 distance for proto')

    # only for structshot
    parser.add_argument('--tau', default=0.05, type=float,
           help='StructShot parameter to re-normalizes the transition probabilities')

    # for span level
    parser.add_argument('--max_span_length', default=10, type=int,
           help='Max length of span')  

    # experiment
    parser.add_argument('--use_sgd_for_bert', action='store_true',
           help='use SGD instead of AdamW for BERT.')

    opt = parser.parse_args()
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    max_length = opt.max_length
    max_span_length = opt.max_span_length

    print("{}-way-{}-shot Few-Shot NER".format(N, K))
    print("model: {}".format(model_name))
    print("max_length: {}".format(max_length))
    print("max_span_length: {}".format(max_span_length))
    print('mode: {}'.format(opt.mode))

    set_seed(opt.seed)
    print('loading model and tokenizer...')
    pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
    word_encoder = BERTWordEncoder(
            pretrain_ckpt)
    # 默认使用的即bert使用的分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print('loading data...')
    if not opt.use_sampled_data:
        opt.train = f'data/{opt.mode}/train.txt'
        opt.test = f'data/{opt.mode}/test.txt'
        opt.dev = f'data/{opt.mode}/dev.txt'
        if not (os.path.exists(opt.train) and os.path.exists(opt.dev) and os.path.exists(opt.test)):
            os.system(f'bash data/download.sh {opt.mode}')
    else:
        opt.train = f'data/episode-data/{opt.mode}/train_{opt.N}_{opt.K}.jsonl'
        opt.test = f'data/episode-data/{opt.mode}/test_{opt.N}_{opt.K}.jsonl'
        opt.dev = f'data/episode-data/{opt.mode}/dev_{opt.N}_{opt.K}.jsonl'
        if not (os.path.exists(opt.train) and os.path.exists(opt.dev) and os.path.exists(opt.test)):
            os.system(f'bash data/download.sh episode-data')
            os.system('unzip -d data/ data/episode-data.zip')

    # 获取dataloader
    train_data_loader = get_loader(opt.train, tokenizer, batch_size=batch_size, max_length=max_length, max_span_length=max_span_length)
    val_data_loader = get_loader(opt.dev, tokenizer, batch_size=batch_size, max_length=max_length, max_span_length=max_span_length)
    test_data_loader = get_loader(opt.test, tokenizer, batch_size=batch_size, max_length=max_length, max_span_length=max_span_length)

        
    prefix = '-'.join([model_name, opt.mode, str(N), str(K), 'seed'+str(opt.seed)])
    if opt.dot:
        prefix += '-dot'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name
    
    # 初始化模型和框架
    model = SpanNNShot(word_encoder, dot = opt.dot, max_span_length = max_span_length, ignore_index = opt.ignore_index)
    framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader)

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt
    print('model-save-path:', ckpt)

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if opt.lr == -1:
            opt.lr = 2e-5

        framework.train(
            model, 
            load_ckpt=opt.load_ckpt, 
            save_ckpt=ckpt,
            val_step=opt.val_step, 
            fp16=opt.fp16,
            train_iter=opt.train_iter, 
            warmup_step=int(opt.train_iter * 0.1), 
            val_iter=opt.val_iter, 
            learning_rate=opt.lr, 
            use_sgd_for_bert=opt.use_sgd_for_bert
        )
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

    # test
    precision, recall, f1 = framework.eval(model, opt.test_iter, ckpt=ckpt)
    print("RESULT: precision: %.4f, recall: %.4f, f1:%.4f" % (precision, recall, f1))

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(time.time() - start_time)