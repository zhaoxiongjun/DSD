import os
import json
import torch
import random
import logging
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict



#固定随机种子
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_test_data(fn):
    """读取用于测试的json数据"""
    with open(fn, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    
    seq_ins, seq_attrs, seq_eids, seq_questions = [], [], [], []
    for k, v in data.items():
        eid = k
        tmp_dialogue = data[eid]
        cur_len = 0
        seq_in, seq_attr= [], []
        for i in range(len(tmp_dialogue)):
            tmp_sent = tmp_dialogue[i]['speaker'] + '：' + tmp_dialogue[i]['sentence']
            tmp_attr = tmp_dialogue[i]['symptom_norm']

            if cur_len + len(tmp_sent) <= 254:
                if cur_len==0:  #首句
                    seq_in = tmp_sent
                    seq_attr = tmp_attr
                else:
                    seq_in = seq_in + ' ' + tmp_sent
                    seq_attr.extend(tmp_attr)
                cur_len = cur_len + len(tmp_sent)
            else:  # 超过max_len，重新开始

                s_attr = set(seq_attr)
                for attr in s_attr:
                    seq_eids.append(eid)
                    seq_ins.append(seq_in)
                    seq_attrs.append(attr)
                    # seq_questions.append(attr)
                    # seq_questions.append("有没有" + attr +"?")
                    seq_questions.append("是否患有" + attr +"?")
                    # seq_questions.append("是否患有" + attr +"症状?")

                seq_in = tmp_sent
                seq_attr = tmp_attr
                cur_len = len(tmp_sent)

        s_attr = set(seq_attr)
        for attr in s_attr:
            seq_eids.append(eid)
            seq_ins.append(seq_in)
            seq_attrs.append(attr)
            # seq_questions.append(attr)
            seq_questions.append("是否患有" + attr +"?")
            # seq_questions.append("是否患有" + attr +"症状?")

    assert len(seq_eids) == len(seq_ins) == len(seq_attrs) == len(seq_questions)
    # print('句子数量为：', len(seq_ins))

    # 数据保存
    name={
        "eids": seq_eids,
        "content": seq_ins,
        "question": seq_questions,
        "attr": seq_attrs
    }
    data=pd.DataFrame(name)
    data_dir = './data'
    csv_path = os.path.join(data_dir, 'test_attr.csv')
    data.to_csv(csv_path, index=False)

    return csv_path


"""
读取训练/ 测试集的 Dataset
每次将csv里的一个成对句子转换成 BERT 的需要的格式，并返回 3 个 tensors：
- tokens_tensor：两个句子合并后的索引序列，包含 [CLS] 与 [SEP]
- segments_tensor：可以用辨别两个句子界限的 binary tensor
- label_tensor：将分类目标转换成类别索引的 tensor, 如果是测试集测传回 None
"""   
class DialogueDataset(Dataset):
    # 读取csv数据文件，并初始化一些参数
    def __init__(self, path, mode, tokenizer):
        assert mode in ["train", "dev", "test"]
        self.path = path
        self.mode = mode
        # 大数据你会需要用 iterator=True
#         self.df = pd.read_csv("./data/" + mode + ".csv", sep=",").fillna("")
        self.df = pd.read_csv(self.path, sep=",").fillna("")
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 使用 BERT tokenizer
    
    # 定义返回一笔训练 / 测试数据的函数
    def __getitem__(self, idx):
        if self.mode == "test":
            eid, text_a, text_b, attr = self.df.iloc[idx].values
            label_tensor = None
        else:
            eid, text_a, text_b, label, attr = self.df.iloc[idx].values
            # 将label 文字转换成索引方便转换成 tensor
#             label_id = self.label_map[label]
            label_id = label
            label_tensor = torch.tensor(label_id)
            
        # 建立第一个句子的 BERT tokens 并加入分隔符号 [SEP]
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)
        
        # 第二个句子的 BERT tokens
        tokens_b = self.tokenizer.tokenize(text_b)
        word_pieces += tokens_b + ["[SEP]"]
        len_b = len(word_pieces) - len_a
        
        # 将整个 token 序列转换成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 将第一句包含 [SEP] 的 token 位置设为0，其他为 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, dtype=torch.long)
        
        return (eid, tokens_tensor, segments_tensor, label_tensor, attr)
    
    def __len__(self):
        return self.len

"""
可以一次回传一个 mini-batch 的 DataLoader
输入我们上面定义的 `DialogueDataset`，
返回训练 BERT 时需要的 4 个 tensors：
- tokens_tensors  : (batch_size, max_seq_len_in_batch)
- segments_tensors: (batch_size, max_seq_len_in_batch)
- masks_tensors   : (batch_size, max_seq_len_in_batch)
- label_ids       : (batch_size)
"""
# 这个函数的输入`samples` 是一个 list，里面的每个 element 都是
# 刚刚定义的 `DialogueDataset` 返回的一个样本，每个样本都包含 3 tensors：
# - tokens_tensor
# - segments_tensor
# - label_tensor
# 它会对前两个tensors 作 zero padding，并生成前面说过的 masks_tensors
def create_mini_batch(samples):
    eids = [s[0] for s in samples]
    tokens_tensors = [s[1] for s in samples]
    segments_tensors = [s[2] for s in samples]
    attrs = [s[4] for s in samples]
    
    # 训练集有 labels
    if samples[0][3] is not None:
        label_ids = torch.stack([s[3] for s in samples])
    else:
        label_ids = None
    
    # zero pad 到同一序列长度
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    
    # attention masks，将tokens_tensors 不为 zero padding
    # 的位置设为 1 让 BERT 只关注这些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    #return for test data
    if label_ids == None:
        return  eids, tokens_tensors, segments_tensors, masks_tensors, attrs
    
    return  eids, tokens_tensors, segments_tensors, masks_tensors, label_ids, attrs

# def create_test_mini_batch(samples):
#     eids = [s[0] for s in samples]
#     tokens_tensors = [s[1] for s in samples]
#     segments_tensors = [s[2] for s in samples]
#     attrs = [s[4] for s in samples]
    
#     # zero pad 到同一序列长度
#     tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
#     segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    
#     # attention masks，将tokens_tensors 不为 zero padding
#     # 的位置设为 1 让 BERT 只关注这些位置的 tokens
#     masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
#     masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    
#     return  eids, tokens_tensors, segments_tensors, masks_tensors, attrs