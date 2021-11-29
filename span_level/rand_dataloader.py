import os
import json
import random
from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RandSpanDataset(Dataset):
    def __init__(
        self, 
        filepath, 
        tokenizer, 
        max_length, 
        max_span_length = 30, 
        ignore_label_id=-1
    ):
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
    
    def __additem__(self, d, word, mask, text_mask, label, span_id):
        d['word'] += word
        d['mask'] += mask
        d['text_mask'] += text_mask
        d['label'] += label
        d['span_id'] += span_id
    
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

    def __sample_span__(self, tokens, span_labels, sub_word):
        span_label_dict = {}
        for span_label in span_labels:
            # assert span_label[2] - span_label[1] < self.max_span_length
            span_label_dict[(span_label[1], span_label[2])] = span_label[0]
        
        labels = []
        spans = []
        for start in range(len(tokens)):
            # 如果start位置是一个单独的词或第一个子词才会加入span label
            if sub_word[start] in [0, 1]:
                end = start
                span_num = 0
                while(span_num < self.max_span_length and end < len(tokens)):
                    # end位置是一个单独的词或最后一个子词才会加入span label
                    if sub_word[end] in [0, 3]:
                        spans.append((start, end))
                        if (start, end) in span_label_dict:
                            labels.append(span_label_dict[(start, end)])
                            span_label_dict.pop((start, end))
                        else:
                            labels.append(0)
                        # 下一个位置是新的span
                        span_num += 1
                    end += 1
        for l in span_label_dict:
            labels.append(span_label_dict[l])
            spans.append(l)
        return labels, spans

    def __getraw__(self, tokens, span_labels, sub_words):
        # 将分词好的句子和label处理成bert的输入形式
        # get tokenized word list, attention mask, text mask (mask [CLS], [SEP] as well), tags   
        # split into chunks of length (max_length-2)
        # 2 is for special tokens [CLS] and [SEP]
        # 处理token
        assert len(tokens) <= self.max_length - 2
        special_tokens = ['[CLS]'] + tokens + ['[SEP]']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(special_tokens)
    
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(special_tokens)] = 1

        # text mask, also mask [CLS] and [SEP]
        text_mask = np.zeros((self.max_length), dtype=np.int32)
        text_mask[1:len(special_tokens)-1] = 1

        # sample label
        labels, span_ids = self.__sample_span__(tokens, span_labels, sub_words)
        return [indexed_tokens], [mask], [text_mask], [labels], [span_ids]

    def __populate__(self, data, savelabeldic=False):
        '''
        populate samples into data dict
        set savelabeldic=True if you want to save label2tag dict
        'word': tokenized word ids
        'mask': attention mask in BERT
        'label': NER labels
        'text_mask': 0 for special tokens and paddings, 1 for real text
        '''
        dataset = {"word": [], "mask": [], "text_mask": [], "label": [], "span_id": []}
        for i in range(len(data['word'])):
            # 传入一句话和这句话对应的label并进行分词
            tokens, raw_labels, sub_words = self.__get_token_label_list__(data['word'][i], data['label'][i])
            # 将label处理成span-level的形式
            span_labels = self.__get_all_span_label__(raw_labels)
            # 转换成bert的输入
            word, mask, text_mask, label, span_id = self.__getraw__(tokens, span_labels, sub_words)
            word = torch.tensor(word).long()
            mask = torch.tensor(mask).long()
            text_mask = torch.tensor(text_mask).long()
            self.__additem__(dataset, word, mask, text_mask, label, span_id)
        dataset["word"] = torch.stack(dataset["word"], axis = 0)
        dataset["mask"] = torch.stack(dataset["mask"], axis = 0)
        dataset["text_mask"] = torch.stack(dataset["text_mask"], axis = 0)
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


def random_sample(data, support_span_num = 1000, query_span_num = 1000):
    # only support batch_size = 1
    assert len(data) == 1
    support_set, query_set = data[0][0], data[0][1]

    # for query set
    query_label = []
    query_span_id = []
    for i in range(len(query_set["word"])):
        query_label.extend(query_set["label"][i])
        span_id = query_set["span_id"][i]
        span_id = [(i, s[0], s[1]) for s in span_id]
        query_span_id.extend(span_id)
    query_label = [torch.tensor(query_label[i: i + query_span_num]).long() for i in range(0, len(query_label), query_span_num)]
    query_span_id = [query_span_id[i: i + query_span_num] for i in range(0, len(query_span_id), query_span_num)]

    query_set["label"] = query_label
    query_set["span_id"] = query_span_id
    
    # for support set
    support_label = []
    support_span_id = []
    for i in range(len(support_set["word"])):
        support_label.extend(support_set["label"][i])
        span_id = support_set["span_id"][i]
        span_id = [(i, s[0], s[1]) for s in span_id]
        support_span_id.extend(span_id)

    special_span = []
    normal_span = []
    for i in range(len(support_label)):
        if support_label[i] != 0:
            special_span.append((support_label[i], support_span_id[i]))
        else:
            normal_span.append((support_label[i], support_span_id[i]))
    select_span = special_span + random.sample(
        normal_span, 
        min(
            len(normal_span), 
            support_span_num - len(special_span)
        )
    )
    random.shuffle(select_span)
    
    support_set["label"] = torch.tensor([s[0] for s in select_span])
    support_set["span_id"] = [s[1] for s in select_span]
    # for support set
    return support_set, query_set


def get_loader(
        filepath, 
        tokenizer,
        batch_size, 
        max_length, 
        max_span_length,
        support_span_num = 1000,
        query_span_num = 1000,
        num_workers = 8, 
        ignore_index = -1
    ):
    dataset = RandSpanDataset(filepath, tokenizer, max_length, max_span_length, ignore_label_id=ignore_index)
    collate_fn = partial(
        random_sample, 
        support_span_num = support_span_num, 
        query_span_num = query_span_num
    )
    data_loader = DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader