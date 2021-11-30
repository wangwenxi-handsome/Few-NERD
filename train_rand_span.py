import os
import time
import parse
import random
from transformers import BertTokenizer
import torch
import numpy as np
from util.word_encoder import BERTWordEncoder
from span_level.rand_dataloader import get_loader
from span_level.rand_framework import RandFewShotNERFramework
from span_level.rand_model import RandSpanNNShot
from util.rand_parse import init_parser

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(opt):
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
    train_data_loader = get_loader(opt.train, tokenizer, batch_size=batch_size, max_length=max_length, max_span_length=max_span_length, support_span_num=opt.support_span_num, query_span_num=opt.query_span_num)
    val_data_loader = get_loader(opt.dev, tokenizer, batch_size=batch_size, max_length=max_length, max_span_length=max_span_length, support_span_num=opt.support_span_num, query_span_num=opt.query_span_num)
    test_data_loader = get_loader(opt.test, tokenizer, batch_size=batch_size, max_length=max_length, max_span_length=max_span_length, support_span_num=opt.support_span_num, query_span_num=opt.query_span_num)

        
    prefix = '-'.join([model_name, opt.mode, str(N), str(K), 'seed'+str(opt.seed)])
    if opt.dot:
        prefix += '-dot'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name
    
    # 初始化模型和框架
    model = RandSpanNNShot(word_encoder, dot = opt.dot, max_span_length = max_span_length, loss = opt.loss)
    framework = RandFewShotNERFramework(train_data_loader, val_data_loader, test_data_loader)

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
    # 初始化parser
    parser = init_parser()
    opt = parser.parse_args()
    
    # experiment1
    start_time = time.time()
    main(opt)
    print(time.time() - start_time)

    # experiment2
    start_time = time.time()
    opt.loss = "focal"
    main(opt)
    print(time.time() - start_time)