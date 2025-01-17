import os
import sys
import torch
# from pytorch_pretrained_bert import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP


class FewShotNERFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def train(
            self,
            model,
            learning_rate = 1e-1,
            train_iter = 30000,
            val_iter = 1000,
            val_step = 2000,
            load_ckpt = None,
            save_ckpt = None,
            warmup_step = 300,
            grad_iter = 1,
            fp16 = False,
            use_sgd_for_bert = False
        ):
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        '''
        print("Start training...")
    
        # Init optimizer
        print('Use bert optim!')
        # 设置优化器和学习率
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if use_sgd_for_bert:
            optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
        else:
            optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        
        # load model
        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)

        # 是否使用半精度
        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model.train()

        # Training
        best_f1 = 0.0
        iter_loss = 0.0
        iter_sample = 0
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0

        it = 0
        while it + 1 < train_iter:
            lack_support_num = 0
            for _, (support, query) in enumerate(self.train_data_loader):
                # support, query = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in support:
                        if k != 'label' and k != 'sentence_num' and k != 'sub_word':
                            support[k] = support[k].cuda()

                # 有的support label被截断了，去除此类数据
                support_label = set()        
                for ls in support["label"]:
                    support_label = support_label | set(ls)
                if len(support_label) != len(query["label2tag"][0]):
                    lack_support_num += 1
                    continue
                
                # only support one batch
                assert len(support["sentence_num"]) == 1
                assert len(query["sentence_num"]) == 1
                tmp_pred_cnt = 0 
                tmp_label_cnt = 0 
                correct = 0
                for i in range(query["sentence_num"][0]):
                    one_query = {}
                    # 取出query中的一句话
                    one_query["sentence_num"] = [1]
                    one_query["label2tag"] = query["label2tag"]
                    one_query["word"] = query["word"][i: i + 1].cuda()
                    one_query["mask"] = query["mask"][i: i + 1].cuda()
                    one_query["text_mask"] = query["text_mask"][i: i + 1].cuda()                   
                    one_label = torch.tensor(query["label"][i]).cuda()
                    
                    # 模型预测
                    logits, pred = model(support, one_query)
                    loss = model.loss(logits, one_label) / float(grad_iter) / query["sentence_num"][0]
                    iter_loss += loss.item()
                    one_pred_cnt, one_label_cnt, one_correct = model.metrics_by_entity(pred, one_label)
                    tmp_pred_cnt += one_pred_cnt
                    tmp_label_cnt += one_label_cnt
                    correct += one_correct
                    if fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    del logits, pred, loss
                
                if it % grad_iter == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                #iter_right += self.item(right.data)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct
                iter_sample += 1
                
                if (it + 1) % 100 == 0 or (it + 1) % val_step == 0:
                    precision = correct_cnt / pred_cnt
                    recall = correct_cnt / label_cnt
                    f1 = 2 * precision * recall / (precision + recall)
                    print('step: {0:4} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                        .format(it + 1, iter_loss/ iter_sample, precision, recall, f1) + '\r')

                if (it + 1) % val_step == 0:
                    _, _, f1 = self.eval(model, val_iter)
                    model.train()
                    if f1 > best_f1:
                        print('Best checkpoint')
                        torch.save({'state_dict': model.state_dict()}, save_ckpt)
                        best_f1 = f1
                    iter_loss = 0.
                    iter_sample = 0.
                    pred_cnt = 0
                    label_cnt = 0
                    correct_cnt = 0

                if (it + 1)  == train_iter:
                    break
                it += 1
        
            print("lack support num", lack_support_num)
        print("\n####################\n")
        print("Finish training ")

    def eval(
        self,
        model,
        eval_iter,
        ckpt=None
    ): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        
        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        pred_cnt = 0 # pred entity cnt
        label_cnt = 0 # true label entity cnt
        correct_cnt = 0 # correct predicted entity cnt
        eval_iter = min(eval_iter, len(eval_dataset))

        lack_support_num = 0
        with torch.no_grad():
            it = 0
            while it + 1 < eval_iter:
                for _, (support, query) in enumerate(eval_dataset):
                    if torch.cuda.is_available():
                        for k in support:
                            if k != 'label' and k != 'sentence_num' and k != 'sub_word':
                                support[k] = support[k].cuda()

                    # 有的support label被截断了，去除此类数据
                    support_label = set()        
                    for ls in support["label"]:
                        support_label = support_label | set(ls)
                    if len(support_label) != len(query["label2tag"][0]):
                        lack_support_num += 1
                        continue
                
                    # only support one batch
                    assert len(support["sentence_num"]) == 1
                    assert len(query["sentence_num"]) == 1
                    
                    for i in range(query["sentence_num"][0]):
                        one_query = {}
                        # 取出query中的一句话
                        one_query["sentence_num"] = [1]
                        one_query["label2tag"] = query["label2tag"]
                        one_query["word"] = query["word"][i: i + 1].cuda()
                        one_query["mask"] = query["mask"][i: i + 1].cuda()
                        one_query["text_mask"] = query["text_mask"][i: i + 1].cuda()                   
                        one_label = torch.tensor(query["label"][i]).cuda()

                        # 模型预测
                        _, pred = model(support, one_query)
                        one_pred_cnt, one_label_cnt, one_correct = model.metrics_by_entity(pred, one_label)
                        pred_cnt += one_pred_cnt
                        label_cnt += one_label_cnt
                        correct_cnt += one_correct
                        del pred

                    if it + 1 == eval_iter:
                        break
                    it += 1
            
            print("val lack label", lack_support_num)
            precision = correct_cnt / pred_cnt
            recall = correct_cnt /label_cnt
            f1 = 2 * precision * recall / (precision + recall)
            print('[EVAL] step: {0:4} | [ENTITY] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}'.format(it + 1, precision, recall, f1) + '\r')
        return precision, recall, f1
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()