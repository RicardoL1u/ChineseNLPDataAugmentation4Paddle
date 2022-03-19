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
        # token策略,由于是中文,使用了token分割,同时对于数字和英文使用char分割。
        # self.tokenizer = tokenization.CharTokenizer(vocab_file=self.bert_vocab_file)
        self.mask_token = "[MASK]"
        self.mask_id = self.tokenizer.convert_tokens_to_ids([self.mask_token])[0]
        self.cls_token = "[CLS]"
        self.cls_id = self.tokenizer.convert_tokens_to_ids([self.cls_token])[0]
        self.sep_token = "[SEP]"
        self.sep_id = self.tokenizer.convert_tokens_to_ids([self.sep_token])[0]

    def predict_single_mask(self, word_ids:list, mask_index:int, prob:float=None):
        """输入一个句子token id list,对其中第mask_index个的mask的可能内容,返回 self.beam_size 个候选词语,以及prob"""
        print(word_ids)
        logits = self.bert_encoder(paddle.to_tensor(word_ids).unsqueeze(axis=0))
        print(logits.shape)
        mask_probs = paddle.nn.functional.softmax(logits.squeeze(axis=0)[mask_index],axis=-1)
        word_ids_out = []
        for mask_prob in mask_probs:
            mask_prob = mask_prob.tolist()
            max_num_index_list = map(mask_prob.index, heapq.nlargest(self.beam_size, mask_prob))
            for i in max_num_index_list:
                if prob and mask_prob[i] < prob:
                    continue
                cur_word_ids = word_ids.copy()
                cur_word_ids[mask_index] = i
                word_ids_out.append([cur_word_ids, mask_prob[i]])
        return word_ids_out
    
    # def predict_batch_mask(self, query_ids:list, mask_indexes:int, prob:float=0.5):
    #     """输入多个token id list,对其中第mask_index个的mask的可能内容,返回 self.beam_size 个候选词语,以及prob
    #     word_ids: [word_ids1:list, ], shape=[batch, query_lenght]
    #     mask_indexes: query要预测的mask_id, [[mask_id], ...], shape=[batch, 1, 1]
    #     """
    #     word_ids_out = []
    #     word_mask = [[1] * len(x) for x in query_ids]
    #     word_segment_ids = [[1] * len(x) for x in query_ids]
    #     fd = {self.input_ids: query_ids, self.input_mask: word_mask, self.segment_ids: 
    #           word_segment_ids, self.masked_lm_positions: mask_indexes}
    #     mask_probs = self.sess.run(self.predict_prob, feed_dict=fd)
    #     for mask_prob, word_ids_, mask_index in zip(mask_probs, query_ids, mask_indexes):
    #         # each query of batch
    #         cur_out = []
    #         mask_prob = mask_prob.tolist()
    #         max_num_index_list = map(mask_prob.index, heapq.nlargest(self.n_best, mask_prob))
    #         for i in max_num_index_list:
    #             cur_word_ids = word_ids_.copy()
    #             cur_word_ids[mask_index[0]] = i
    #             cur_out.append([cur_word_ids, mask_prob[i]])
    #         word_ids_out.append(cur_out)
    #     return word_ids_out

    def gen_sen(self, word_ids:list, indexes:list):
        """
        输入是一个word id list, 其中包含mask,对mask生产对应的词语。
        因为每个query的mask数量不一致,预测测试不一致,需要单独预测
        """
        out_arr = []
        for i, index_ in enumerate(indexes):
            if i == 0:
                out_arr = self.predict_single_mask(word_ids, index_)
            else:
                tmp_arr = out_arr.copy()
                out_arr = []
                for word_ids_, prob in tmp_arr:
                    cur_arr = self.predict_single_mask(word_ids_, index_)
                    cur_arr = [[x[0], x[1] * prob] for x in cur_arr]
                    out_arr.extend(cur_arr)
                # 筛选前beam size个
                out_arr = sorted(out_arr, key=lambda x: x[1], reverse=True)[:self.beam_size]
        print(type(out_arr))
        for i, (each, _) in enumerate(out_arr):
            query_ = [self.tokenizer.convert_ids_to_tokens[x] for x in each]
            out_arr[i][0] = query_
        return out_arr

    def word_insert(self, query):
        """随机将某些词语mask,使用bert来生成 mask 的内容。
        
        max_query: 所有query最多生成的个数。
        """
        out_arr = []
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
        print(word_ids_arr)
        print(word_index_arr)
        for word_ids, word_index in zip(word_ids_arr, word_index_arr):
            arr_ = self.gen_sen(word_ids, indexes=word_index)
            out_arr.extend(arr_)
            pass
        # 这个是所有生成的句子中,筛选出前 beam size 个。
        out_arr = sorted(out_arr, key=lambda x: x[1], reverse=True)
        out_arr = ["".join(x[0][1:-1]) for x in out_arr[:self.beam_size]]
        return out_arr

    def insert_word2queries(self, queries:list, beam_size=10):
        self.beam_size = beam_size
        out_map = defaultdict(list)
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