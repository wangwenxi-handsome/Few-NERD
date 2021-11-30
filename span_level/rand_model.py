import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class RandSpanNNShot(nn.Module):
    
    def __init__(self, word_encoder, dot = False, max_span_length = 10, loss = "ce"):
        super(RandSpanNNShot, self).__init__()
        self.word_encoder = word_encoder
        self.drop = nn.Dropout()
        if loss == "ce":
            self.cost = nn.CrossEntropyLoss()
        elif loss == "focal":
            self.cost = FocalLoss()
        self.dot = dot
        self.max_span_length = max_span_length

    def __dist__(self, x, y, dim):
        # 这里有broadcast机制
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)

    def __get_nearest_dist__(self, S, S_tag, Q):
        dist = self.__batch_dist__(S, Q)
        nearest_dist = []
        for label in range(torch.max(S_tag) + 1):
            nearest_dist.append(torch.max(dist[:,S_tag==label], 1)[0])
        nearest_dist = torch.stack(nearest_dist, dim=1) # [num_of_query_tokens, class_num]
        return nearest_dist

    def __get_span_tensor__(self, sent_embs, text_mask, span_id):
        span_tensors = []
        for s in span_id:
            sent_emb = sent_embs[s[0]]
            sent_emb = sent_emb[text_mask[s[0]] == 1]
            span_tensors.append(
                torch.cat([sent_emb[s[1]], sent_emb[s[2]]], axis = 0)
            )
        return span_tensors

    def forward(self, support, query):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        # 利用word encoder对句子解码
        support_emb = self.word_encoder(support['word'], support['mask']) # [num_sent, number_of_tokens, 768]
        query_emb = self.word_encoder(query['word'], query['mask']) # [num_sent, number_of_tokens, 768]
        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)

        # 处理成span的形式
        support_spans = self.__get_span_tensor__(support_emb, support["text_mask"], support["span_id"])
        query_spans = self.__get_span_tensor__(query_emb, query["text_mask"], query["span_id"])
        support_spans = torch.stack(support_spans, axis = 0)
        query_spans = torch.stack(query_spans, axis = 0)

        # 计算距离
        logits = self.__get_nearest_dist__(support_spans, support['label'], query_spans)
        _, pred = torch.max(logits, 1)
        return logits, pred

    def loss(self, logits, label):
        return self.cost(logits, label)

    def metrics_by_entity(self, pred, label):
        assert len(pred) == len(label)
        pred_entity = 0
        label_entity = 0
        correct_entity = 0
        for i in range(len(pred)):
            if pred[i] != 0:
                pred_entity += 1
            if label[i] != 0:
                label_entity += 1
            if pred[i] != 0 and label[i] != 0 and pred[i] == label[i]:
                correct_entity += 1
        return pred_entity, label_entity, correct_entity


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None,ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index=ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight,ignore_index=self.ignore_index)
        return loss