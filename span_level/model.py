import torch
import torch.nn as nn


class SpanNNShot(nn.Module):
    
    def __init__(self, word_encoder, dot=False, ignore_index=-1):
        self.ignore_index = ignore_index
        self.word_encoder = nn.DataParallel(word_encoder)
        self.drop = nn.Dropout()
        self.cost = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dot = dot

    def __dist__(self, x, y, dim):
        # 这里有broadcast机制
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        for q in Q:
            
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)

    def __get_nearest_dist__(self, embedding, tag, query):
        # 将support set中的span铺平
        nearest_dist = []
        S = []
        S_tag = []
        for i, e in enumerate(embedding):
            S.extend(e)
            S_tag.extend(tag[i])
        assert len(S) == len(S_tag)

        # 计算query set中的每一个span与support set中的距离
        dist = self.__batch_dist__(S, query)

        for label in range(torch.max(tag)+1):
            nearest_dist.append(torch.max(dist[:,tag==label], 1)[0])
        nearest_dist = torch.stack(nearest_dist, dim=1)
        return nearest_dist

    def __get_span_tensor__(self, sent_embs, text_mask):
        span_tensors = []
        for i, sent in enumerate(sent_embs):
            sent_emb = sent[text_mask[i] == 1]
            span_tensor = []
            for start in range(len(sent_emb)):
                for end in range(start, len(sent_emb)):
                    span_tensor.append(torch.cat([sent_emb[start], sent_emb[end]], axis = 0))
            span_tensors.append(span_tensor)
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
        assert support_emb.size()[:2] == support['mask'].size()
        assert query_emb.size()[:2] == query['mask'].size()

        # 处理成span的形式
        support_spans = self.__get_span_tensor__(support_emb, support["text_mask"])
        query_spans = self.__get_span_tensor__(query_emb, query["text_mask"])

        # 检查一下mask形状对不对，虽然我不知道这样做的意义
        logits = []
        current_support_num = 0
        current_query_num = 0

        # 一次取出一条数据进行操作，一条数据是指 N-way-K-shot-Q-shot
        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            # Calculate nearest distance to single entity in each class in support set
            logits.append(self.__get_nearest_dist__(
                support_spans[current_support_num: current_support_num + sent_support_num], 
                support['label'][current_support_num: current_support_num + sent_support_num], 
                query_spans[current_query_num : current_query_num + sent_query_num],
            ))
            current_query_num += sent_query_num
            current_support_num += sent_support_num

        logits = torch.cat(logits, 0)
        _, pred = torch.max(logits, 1)
        return logits, pred