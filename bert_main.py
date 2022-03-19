#!/usr/bin/env python
# encoding=utf-8
'''
@Time    :   2020/06/14 17:45:13
@Author  :   zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   
1. 随机插入mask,使用bert来生成 mask 的内容,来丰富句子
2. 随机将某些词语mask,使用bert来生成 mask 的内容。
    - 使用贪心算法,每次最优。
    - beam search方法,每次保留最优的前n个,最多num_beams个句子。(注意句子数据大于num_beams个时候,剔除概率最低的,防止内存溢出)。
'''

from paddlenlp.transformers import BertTokenizer,BertForMaskedLM
from collections import defaultdict
import jieba
import numpy as np
import heapq
import paddle.nn
import paddle
# from transformers import pipelines


class BertAugmentor(object):
    def __init__(self,  pre_train_dir: str, beam_size=5):
        self.beam_size = beam_size    # 每个带mask的句子最多生成 beam_size 个。
        self.bert_encoder = BertForMaskedLM.from_pretrained(pre_train_dir)
        self.tokenizer = BertTokenizer.from_pretrained(pre_train_dir)
        self.topk = 2
        # token策略,由于是中文,使用了token分割,同时对于数字和英文使用char分割。
        # self.tokenizer = tokenization.CharTokenizer(vocab_file=self.bert_vocab_file)
        self.mask_token = "[MASK]"
        self.mask_id = self.tokenizer.convert_tokens_to_ids([self.mask_token])[0]
        self.cls_token = "[CLS]"
        self.cls_id = self.tokenizer.convert_tokens_to_ids([self.cls_token])[0]
        self.sep_token = "[SEP]"
        self.sep_id = self.tokenizer.convert_tokens_to_ids([self.sep_token])[0]

    

    def gen_sen(self, word_ids:list, indexes:list):
        """
        输入是一个word id list, 其中包含mask,对mask生产对应的词语。
        因为每个query的mask数量不一致,预测测试不一致,需要单独预测
        """
        outputs = self.bert_encoder(paddle.to_tensor(word_ids).unsqueeze(axis=0)).squeeze(axis=0) # (seq_len,dim)
        outputs = outputs.numpy()
        logits = paddle.to_tensor(outputs[indexes, :]) #(mask_num,vocb_corpus_size)
        probs = paddle.nn.functional.softmax(logits,axis=-1)
        values, predictions = probs.topk(self.beam_size)
        print(len(indexes),values.shape,predictions.shape)
        return self.gen_seq(word_ids,indexes,values,predictions)
        
    
    def gen_seq(self,word_ids:list, indexes:list,values,predictions:paddle.Tensor):
        result = []
        ptr = [0] * len(indexes)
        while len(result) < self.topk:
            seq = np.array(word_ids).copy()
            min_value = 1
            pop_ptr = 0
            v = []
            p = []
            for i,index in enumerate(indexes):
                seq[index] = predictions[i,ptr[i]].item()
                v.append(values[i,ptr[i]].item())
                p.append(predictions[i,ptr[i]].item())
                if values[i,ptr[i]].item() < min_value:
                    min_value = values[i,ptr[i]].item()
                    pop_ptr = i
            seq = seq[np.where(seq != self.tokenizer.pad_token_id)]
            sequence = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(seq,skip_special_tokens=True))
            proposition = {"score": v, "insert_index":indexes,"token": p, "token_str": self.tokenizer.convert_ids_to_tokens(p), "sequence": sequence}
            result.append(proposition)
            ptr[pop_ptr] += 1
        return result

    def word_insert(self, query):
        """随机将某些词语mask,使用bert来生成 mask 的内容。
        max_query: 所有query最多生成的个数。
        """
        result = []
        seg_list = jieba.cut(query, cut_all=False)
        # 随机选择非停用词mask。
        i, index_arr = 1, [1]
        for each in seg_list:
            i += len(each)
            index_arr.append(i)
        # query转id
        query = '[CLS]' + query + '[SEP]'
        word_ids = self.tokenizer(query)["input_ids"]
        word_ids_arr, word_index_arr = [], []
        # 随机insert n 个字符, 1<=n<=3
        for index_ in index_arr:
            insert_num = np.random.randint(1, 4)
            word_ids_ = word_ids.copy()
            word_index = []
            for i in range(insert_num):
                word_ids_.insert(index_, self.mask_id)
                word_index.append(index_ + i)
            word_ids_arr.append(word_ids_)
            word_index_arr.append(word_index)
        for word_ids, word_index in zip(word_ids_arr, word_index_arr):
            result.extend(self.gen_sen(word_ids, indexes=word_index))

        return result

    def word_replace(self,query):
        """随机将某些词语mask，使用bert来生成 mask 的内容。"""
        result = []
        seg_list = jieba.cut(query, cut_all=False)
        # 随机选择非停用词mask。
        i, index_map = 1, {}
        for each in seg_list:
            index_map[i] = len(each)
            i += len(each)
        # query转id
        query = '[CLS]' + query + '[SEP]'
        word_ids = self.tokenizer(query)["input_ids"]
        word_ids_arr, word_index_arr = [], []
        # 依次mask词语，
        for index_, word_len in index_map.items():
            word_ids_ = word_ids.copy()
            word_index = []
            for i in range(word_len):
                word_ids_[index_ + i] = self.mask_id
                word_index.append(index_ + i)
            word_ids_arr.append(word_ids_)
            word_index_arr.append(word_index)

        for word_ids, word_index in zip(word_ids_arr, word_index_arr):
            result.extend(self.gen_sen(word_ids, indexes=word_index))

        return result

    def insert_word2queries(self, queries:list, beam_size=10):
        out_map = {}
        for query in queries:
            out_map[query] = self.word_insert(query)
        return out_map

    def replace_word2queries(self, queries:list, beam_size=10):
        self.beam_size = beam_size
        out_map = defaultdict(list)
        for query in queries:
            out_map[query] = self.word_replace(query)
        return out_map

    def predict(self, query_arr, beam_size=None):
        """
        query_arr: ["w1", "w2", "[MASK]", ...], shape=[word_len]
        每个query_arr, 都会返回beam_size个
        """
        self.beam_size = beam_size if beam_size else self.beam_size
        word_ids, indexes = self.tokenizer.convert_tokens_to_ids(query_arr), [x[0] for x in filter(lambda x: x[1] == self.mask_token, enumerate(query_arr))]
        out_queries = self.gen_sen(word_ids, indexes)
        out_queries = [["".join(x[0]), x[1]] for x in out_queries]
        return out_queries