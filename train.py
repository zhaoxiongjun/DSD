import logging
import os
import argparse
import random
import json

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.metrics import f1_score
from transformers import *
from utils import seed_everything, DialogueDataset, create_mini_batch

logger = logging.getLogger()

def init_logging(args):
    """logging设置和参数信息打印"""
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    logger.addHandler(chlr)

    logger.info("====== parameters setting =======")
    logger.info("data_dir: " + str(args.data_dir))
    logger.info("save_dir: " + str(args.save_dir))
    logger.info("num_epoch: " + str(args.num_epoch))
    logger.info("batch_size: " + str(args.batch_size))
    logger.info("random_seed: " + str(args.random_seed))
    logger.info("evaluate_steps: " + str(args.evaluate_steps))

"""加载预训练模型"""
def build_model():
    # 使用中文 BERT
    PRETRAINED_MODEL_NAME = "bert-base-chinese"
    NUM_LABELS = 3

    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
    
    """获得设备类型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
  
    logger.info("***** Model Loaded *****")
    return model, tokenizer

# def save_result(eids, attrs, preds):
#     name = {
#         "eids": eids,
#         "attrs": attrs,
#         "preds": preds
#     }
#     ret_df = pd.DataFrame(name)
#     ret_df.to_csv('ret.csv',index=False)

def evaluate(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0
    eids, attrs, y_trues = [], [], []

    model.eval()
    with torch.no_grad():
        # 遍历整个数据集
        for data in dataloader:
            # 将所有 tensors 移到 GPU 上
            eid = data[0]
            y_true = data[4]
            attr = data[5]
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda") for t in data[1:4] if t is not None]
            
            tokens_tensors, segments_tensors, masks_tensors = data
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            # 计算分类准确率
            # if compute_acc:
            #     labels = data[3]
            #     total += labels.size(0)
            #     correct += (pred == labels).sum().item()
                
            # 记录当前batch
            if predictions is None:
                predictions = pred
                eids = eid
                attrs = attr
                y_trues = y_true.tolist()
            else:
                predictions = torch.cat((predictions, pred))
                eids.extend(eid)
                attrs.extend(attr)
                y_trues.extend(y_true.tolist())
    
    # if compute_acc:
    #     acc = correct / total
    #     return eids, attrs, y_trues, predictions, acc
    return eids, attrs, y_trues, predictions

def train(model, trainloader, devloader, args):
    # 训练模式
    model.train()

    # 使用 Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    EPOCHS = args.num_epoch  # 训练轮数
    batchs = 0  # batchs 数
    best_f1 = 0

    for epoch in range(EPOCHS):
        
        running_loss = 0.0
        for data in trainloader:
            batchs = batchs + 1
            tokens_tensors, segments_tensors, \
            masks_tensors, labels = [t.to("cuda") for t in data[1:5]]

            # 梯度置零
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors, 
                            labels=labels)

            loss = outputs[0]
            # backward
            loss.backward()
            optimizer.step()

            # 记录当前 batch loss
            running_loss += loss.item()
            
            if batchs % args.evaluate_steps == 0:
                # 计算dev分类acc和f1
                eids, attrs, y_trues, preds = evaluate(model, devloader)
                f1 = f1_score(y_trues, preds.cpu(),average='micro')
                
                # save model
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(),os.path.join(args.save_dir,'net_params.pth'))
                    # save_result(eids, attrs, preds.cpu())
                    logger.info("best performer hear. Saving model checkpoint to %s", args.save_dir)
                
                logger.info('[epoch %d, batch %d] train loss: %.3f, dev f1: %.3f' %
                    (epoch, batchs, loss.item(), f1))
                
                model.train() # 切换回来

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-dd', type=str, default='data', help='Train/dev data path')
    parser.add_argument('--save_dir', '-sd', type=str, default='./save_model', help='Path to save, load model')
    parser.add_argument('--num_epoch', '-ne', type=int, default=6, help='Total number of training epochs to perform')
    parser.add_argument('--batch_size', '-bs', type=int, default=16, help='Batch size for trainging')
    parser.add_argument('--random_seed', '-rs', type=int, default=66, help='Random seed')
    parser.add_argument('--evaluate_steps', '-ls', type=int, default=200, help='Evaluate every X updates steps')

    args = parser.parse_args()

    seed_everything(args.random_seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    init_logging(args)

    model, tokenizer = build_model()
    trainset = DialogueDataset(os.path.join(args.data_dir,"train.csv"), "train", tokenizer=tokenizer)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=create_mini_batch)

    devset = DialogueDataset(os.path.join(args.data_dir,"dev.csv"), "dev", tokenizer=tokenizer)
    devloader = DataLoader(devset, batch_size=args.batch_size, shuffle=False, collate_fn=create_mini_batch)

    train(model, trainloader, devloader, args)

    