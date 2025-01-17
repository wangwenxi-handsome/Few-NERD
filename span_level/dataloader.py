import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SpanLevelDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length, max_span_length, ignore_label_id=-1):
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.class2sampleid = {}
        self.tokenizer = tokenizer
        self.samples = self.__load_data_from_file__(filepath)
        self.max_length = max_length
        self.max_span_length = max_span_length
        self.ignore_label_id = ignore_label_id
    
    def __load_data_from_file__(self, filepath):
        with open(filepath)as f:
            lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = json.loads(lines[i].strip())
        return lines
    
    def __additem__(self, d, word, mask, text_mask, label, sub_word):
        d['word'] += word
        d['mask'] += mask
        d['label'] += label
        d['text_mask'] += text_mask
        d['sub_word'] += sub_word
    
    def __get_token_label_list__(self, words, tags):
        tokens = []
        labels = []
        sub_words = []
        for word, tag in zip(words, tags):
            word_tokens = self.tokenizer.tokenize(word)
            if word_tokens:
                # tokens and labels
                tokens.extend(word_tokens)
                word_labels = [self.tag2label[tag]] + [self.ignore_label_id] * (len(word_tokens) - 1)
                labels.extend(word_labels)
                # sub word
                if len(word_tokens) == 1:
                    sub_words.append(0)
                else:
                    sub_words.extend([1] + [2] * (len(word_tokens) - 2) + [3])
        return tokens, labels, sub_words

    def __get_all_span_label__(self, label):
        span_label = []
        now_label = None
        l = 0
        while(l < len(label)):
            if label[l] not in [0, self.ignore_label_id]:
                now_label = label[l]
                start = l
                while(label[l] in [now_label, self.ignore_label_id] and l < len(label)):
                    l += 1
                span_label.append([now_label, start, l - 1])
            else:
                l += 1
        return span_label

    def __getraw__(self, tokens, span_labels, sub_words):
        # 将分词好的句子和label处理成bert的输入形式
        # get tokenized word list, attention mask, text mask (mask [CLS], [SEP] as well), tags   
        # split into chunks of length (max_length-2)
        # 2 is for special tokens [CLS] and [SEP]
        tokens_list = []
        span_labels_list = []
        sub_words_list = []
        label_start_id = 0
        while len(tokens) > self.max_length - 2:
            # 处理labels，保留在句子范围内的label，并且修正id
            tmp_span_label = []
            for l in span_labels:
                tmp_l = [l[0], l[1] - label_start_id, l[2] - label_start_id]
                if tmp_l[2] < self.max_length - 2 and tmp_l[1] >= 0:
                    tmp_span_label.append(tmp_l)
                    span_labels.remove(l)
            span_labels_list.append(tmp_span_label)
            label_start_id += self.max_length - 2
            # 处理tokens
            tokens_list.append(tokens[:self.max_length - 2])
            tokens = tokens[self.max_length - 2:]
            # 处理subwords
            sub_words_list.append(sub_words[:self.max_length - 2])
            sub_words = sub_words[self.max_length - 2]
        # 超过最大长度但不足最大长度的部分也要补齐
        if tokens:
            tokens_list.append(tokens)
            sub_words_list.append(sub_words)
            tmp_span_label = []
            for l in span_labels:
                tmp_l = [l[0], l[1] - label_start_id, l[2] - label_start_id]
                if tmp_l[2] < self.max_length - 2 and tmp_l[1] >= 0:
                    tmp_span_label.append(tmp_l)
                    span_labels.remove(l)
            span_labels_list.append(tmp_span_label)

        # 将span_label转化为一个span对应一个标签的形式
        labels = []
        span_labels_dict = {}

        for i, span_labels in enumerate(span_labels_list):
            # 取出raw labels
            sub_word = sub_words_list[i]
            # 取出句子长度和span labels的字典形式
            sent_len = len(tokens_list[i])
            span_label_dict = {}
            for span_label in span_labels:
                # assert span_label[2] - span_label[1] < self.max_span_length
                span_label_dict[(span_label[1], span_label[2])] = span_label[0]
            
            label = []
            for start in range(sent_len):
                # 如果start位置是一个单独的词或第一个子词才会加入span label
                if sub_word[start] in [0, 1]:
                    end = start
                    span_num = 0
                    while(span_num < self.max_span_length and end < sent_len):
                        # end位置是一个单独的词或最后一个子词才会加入span label
                        if sub_word[end] in [0, 3]:
                            if (start, end) in span_label_dict:
                                label.append(span_label_dict[(start, end)])
                            else:
                                label.append(0)
                            # 下一个位置是新的span
                            span_num += 1
                        end += 1
            labels.append(label)

        # add special tokens and get masks
        indexed_tokens_list = []
        mask_list = []
        text_mask_list = []
        for i, tokens in enumerate(tokens_list):
            # token -> ids
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
            # padding
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens_list.append(indexed_tokens)

            # mask
            mask = np.zeros((self.max_length), dtype=np.int32)
            mask[:len(tokens)] = 1
            mask_list.append(mask)

            # text mask, also mask [CLS] and [SEP]
            text_mask = np.zeros((self.max_length), dtype=np.int32)
            text_mask[1:len(tokens)-1] = 1
            text_mask_list.append(text_mask)
            
        return indexed_tokens_list, mask_list, text_mask_list, labels, sub_words_list

    def __populate__(self, data, savelabeldic=False):
        '''
        populate samples into data dict
        set savelabeldic=True if you want to save label2tag dict
        'word': tokenized word ids
        'mask': attention mask in BERT
        'label': NER labels
        'sentence_num': number of sentences in this set (a batch contains multiple sets)
        'text_mask': 0 for special tokens and paddings, 1 for real text
        '''
        dataset = {'word': [], 'mask': [], 'label':[], 'sentence_num':[], 'text_mask':[], 'sub_word':[]}
        for i in range(len(data['word'])):
            # 传入一句话和这句话对应的label并进行分词
            tokens, raw_labels, sub_words = self.__get_token_label_list__(data['word'][i], data['label'][i])
            # 将label处理成span-level的形式
            span_labels = self.__get_all_span_label__(raw_labels)
            # 转换成bert的输入
            word, mask, text_mask, label, sub_word = self.__getraw__(tokens, span_labels, sub_words)
            word = torch.tensor(word).long()
            mask = torch.tensor(mask).long()
            text_mask = torch.tensor(text_mask).long()
            self.__additem__(dataset, word, mask, text_mask, label, sub_word)
        dataset['sentence_num'] = [len(dataset['word'])]
        if savelabeldic:
            dataset['label2tag'] = [self.label2tag]
        return dataset

    def __getitem__(self, index):
        sample = self.samples[index]
        target_classes = sample['types']
        support = sample['support']
        query = sample['query']
        # add 'O' and make sure 'O' is labeled 0
        distinct_tags = ['O'] + target_classes
        self.tag2label = {tag:idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx:tag for idx, tag in enumerate(distinct_tags)}
        support_set = self.__populate__(support)
        query_set = self.__populate__(query, savelabeldic=True)
        return support_set, query_set

    def __len__(self):
        return len(self.samples)


def collate_fn(data):
    # dataset中的每一条数据的形式是(support_set, query_set)
    # one batch data is [(support_set, query_set), (support_set, query_set)......]
    # support_set is [{'word': [], 'mask': [], 'label':[], 'sentence_num':[], 'text_mask':[]}......]
    # 所以这里做的处理为将每一个键值extend，且concat成tensor
    batch_support = {'word': [], 'mask': [], 'label':[], 'sentence_num':[], 'text_mask':[]}
    batch_query = {'word': [], 'mask': [], 'label':[], 'sentence_num':[], 'label2tag':[], 'text_mask':[]}
    support_sets, query_sets = zip(*data)
    for i in range(len(support_sets)):
        for k in batch_support:
            batch_support[k] += support_sets[i][k]
        for k in batch_query:
            batch_query[k] += query_sets[i][k]

    for k in batch_support:
        if k != 'label' and k != 'sentence_num':
            batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        if k !='label' and k != 'sentence_num' and k!= 'label2tag':
            batch_query[k] = torch.stack(batch_query[k], 0)
    return batch_support, batch_query


def get_loader(
        filepath, 
        tokenizer,
        batch_size, 
        max_length, 
        max_span_length,
        num_workers = 0, 
        collate_fn = collate_fn, 
        ignore_index = -1
    ):
    dataset = SpanLevelDataset(filepath, tokenizer, max_length, max_span_length, ignore_label_id=ignore_index)
    data_loader = DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader