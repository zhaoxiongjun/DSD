# -*- coding:utf-8 -*-

import pandas as pd
import json
import os


def read_train_data(fn):
    """读取用于训练的json数据"""
    with open(fn, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data

def read_test_data(fn):
    """读取用于测试的json数据"""
    with open(fn, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data

def read_example_ids(fn):
    """读取划分数据集的文件"""
    example_ids = pd.read_csv(fn)
    return example_ids

def save_train_data(data, example_ids, mode):
    """
    训练集和验证集的数据转换 (trick：本次处理时将BIO转化为BIOES标注，以提高多任务训练的整体准确率)
    :param data: 用于训练的json数据
    :param example_ids: 样本id划分数据
    :param mode: train/dev
    :return:
    """
    eids = example_ids[example_ids['split'] == mode]['example_id'].to_list()

    seq_ins, seq_attrs, seq_types, seq_eids, seq_questions = [], [], [], [], []
    for eid in eids:
        tmp_data = data[str(eid)]
        tmp_dialogue = tmp_data['dialogue']
        tmp_types = tmp_data['implicit_info']['Symptom']   #文本级label
        cur_len = 0
        seq_in, seq_attr, seq_type = [], [], []
        for i in range(len(tmp_dialogue)):
            tmp_sent = tmp_dialogue[i]['speaker'] + '：' + tmp_dialogue[i]['sentence']
            tmp_attr = tmp_dialogue[i]['symptom_norm']
            # tmp_attr = "有没有" + tmp_dialogue[i]['symptom_norm'] + "?"

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
                    seq_questions.append("有没有" + attr +"?")
                    seq_types.append(tmp_types[attr])

                seq_in = tmp_sent
                seq_attr = tmp_attr
                cur_len = len(tmp_sent)

        s_attr = set(seq_attr)
        for attr in s_attr:
            seq_eids.append(eid)
            seq_ins.append(seq_in)
            seq_attrs.append(attr)
            seq_questions.append("有没有" + attr +"?")
            seq_types.append(tmp_types[attr])

    assert len(seq_eids) == len(seq_ins) == len(seq_attrs) == len(seq_types) == len(seq_questions)
    print(mode, '句子数量为：', len(seq_ins))
    # 数据保存
    name={
        "eids": seq_eids,
        "content": seq_ins,
        "question": seq_questions,
        "label": seq_types,
        "attr": seq_attrs
    }
    data=pd.DataFrame(name)
    # print(data.head(1))
    data_dir = 'data'
    csv_path = os.path.join(data_dir, mode+'.csv')
    data.to_csv(csv_path, index=False)

def save_test_data(data, example_ids, mode):
    """
    训练集和验证集的数据转换 (trick：本次处理时将BIO转化为BIOES标注，以提高多任务训练的整体准确率)
    :param data: 用于训练的json数据
    :param example_ids: 样本id划分数据
    :param mode: train/dev
    :return:
    """
    eids = example_ids[example_ids['split'] == mode]['example_id'].to_list()
    print(len(eids))
    seq_ins, seq_attrs, seq_eids, seq_questions = [], [], [], []
    for eid in eids:
        tmp_data = data[str(eid)]
        tmp_dialogue = tmp_data['dialogue']
        cur_len = 0
        seq_in, seq_attr= [], []
        for i in range(len(tmp_dialogue)):
            tmp_sent = tmp_dialogue[i]['speaker'] + '：' + tmp_dialogue[i]['sentence']
            tmp_attr = tmp_dialogue[i]['symptom_norm']
            # tmp_attr = "有没有" + tmp_dialogue[i]['symptom_norm'] + "?"

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
                    seq_questions.append("有没有" + attr +"?")

                seq_in = tmp_sent
                seq_attr = tmp_attr
                cur_len = len(tmp_sent)

        s_attr = set(seq_attr)
        for attr in s_attr:
            seq_eids.append(eid)
            seq_ins.append(seq_in)
            seq_attrs.append(attr)
            seq_questions.append("有没有" + attr +"?")

    assert len(seq_eids) == len(seq_ins) == len(seq_attrs) == len(seq_questions)
    print(mode, '句子数量为：', len(seq_ins))
    # 数据保存
    name={
        "eids": seq_eids,
        "content": seq_ins,
        "question": seq_questions,
        "attr": seq_attrs
    }
    data=pd.DataFrame(name)
    print(data.head(1))
    data_dir = 'data'
    csv_path = os.path.join(data_dir, 'attr_dev.csv')
    data.to_csv(csv_path, index=False)



if __name__ == "__main__":

    train_data = read_train_data('../dataset/train.json')
    example_ids = read_example_ids('../dataset/split.csv')

    data_dir = 'cls_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
 
    save_train_data(
        read_train_data('../dataset/train.json'),
        example_ids,
        'train',
    )
    save_train_data(
        read_train_data('../dataset/train.json'),
        example_ids,
        'dev',
    )

    # save_test_data(
    #     read_train_data('../dataset/attr_dev.json'),
    #     example_ids,
    #     'dev',
    # )

