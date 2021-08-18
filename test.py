import os
import argparse
import random
import json
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import *

from utils import seed_everything, DialogueDataset, create_mini_batch, read_test_data


"""加载模型"""
def build_model(model_dir):
    # 使用中文 BERT
    # PRETRAINED_MODEL_NAME = "bert-base-chinese"
    PRETRAINED_MODEL_NAME = 'bert_models/chinese-roberta-wwm-ext-large'
    
    NUM_LABELS = 3

    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
    model.load_state_dict(torch.load(model_dir))  
    """获得设备类型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print("***** Model Loaded *****")
    return model

"""
预测阶段：输入句子对（对话，问题），进行3分类预测
0：没有
1：有
2：不确定
"""
def get_predicts(model, dataloader):
    predictions = None
    total = 0
    eids, attrs = [],[]
    
    with torch.no_grad():
        # 遍历整个数据集
        for data in dataloader:
            # 将所有 tensors 移到 GPU 上
            eid = data[0]
            attr = data[4]
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda") for t in data[1:4] if t is not None]
            
            # 强烈建议在将这些 tensors 丟入 `model` 时指定对应的参数名称
            tokens_tensors, segments_tensors, masks_tensors = data
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            # 将当前 batch 记录下来
            if predictions is None:
                predictions = pred
                eids = eid
                attrs = attr
            else:
                predictions = torch.cat((predictions, pred))
                eids.extend(eid)
                attrs.extend(attr)

    return eids, attrs, predictions

def predict(args):
   
    print("===============Start Prediction==============")
    # model_version = 'bert-base-chinese'
    model_version = 'bert_models/chinese-roberta-wwm-ext-large'
    
    tokenizer = BertTokenizer.from_pretrained(model_version)
    
    testset = DialogueDataset(read_test_data(args.test_input_file), "test", tokenizer=tokenizer)
    # testset = DialogueDataset("../data/cls_data/dev_test2.csv", "test", tokenizer=tokenizer)
    testloader = DataLoader(testset, batch_size=16, shuffle=False, collate_fn=create_mini_batch)

    model = build_model(args.model_dir)
    eids, attrs, predictions = get_predicts(model, testloader)

    """save result data to json"""
    outputs = defaultdict(list)

    for i in range(len(eids)) :
        outputs[str(eids[i])].append([attrs[i], predictions[i].item()])
    cnt = 0
    for eid, pairs in outputs.items():
        tmp_pred_new = {}
        if len(pairs) != 0:
            for pair in pairs:
                if pair[1] != 3:  # 4分类
                    tmp_pred_new[pair[0]] = str(pair[1])
                else:
                    cnt += 1
        outputs[eid]=tmp_pred_new

    # 将那些预测为空的样本id也存入进来，防止输出的样本缺失
    with open(args.test_input_file, 'r', encoding='utf-8') as fr:
        eids_all = json.load(fr)
    
    for eid in eids_all.keys():
        if eid not in outputs:
            outputs[eid] = {}

  
    print("测试样本数量为：", len(outputs))
    print("none数量为：", cnt)
    pred_path = os.path.join(args.test_output_file)

    with open(pred_path, 'w', encoding='utf-8') as json_file:
        json.dump(outputs, json_file, ensure_ascii=False, indent=4)
   
    print("Prediction Done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-dd', type=str, default='data/near_data', help='Train/dev data path')
    parser.add_argument('--model_dir', '-sd', type=str, default='./save_model_qe/net_params.pth', help='Path to save, load model')
    parser.add_argument('--test_input_file', '-tif', type=str, default='./test_attr_pred.json', help='Input file for prediction')
    parser.add_argument('--test_output_file', '-tof', type=str, default='./evaluate/preds_roberta-xlarge-q_e.json', help='Output file for prediction')
   
    args = parser.parse_args()

    seed_everything(66)

    predict(args)

   
# 10033750