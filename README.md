# MRC-DSD
A General MRC Framework for Dialogue Symptom Diagnosis

# Qiuck Start

## 0. Requirements

- python>=3.7
- torch==1.8.1
- transformers
- pandas
- sklearn
- numpy

## 1. Data Preprocess 

The dataset used in this work can be obtained from [CMDD](http://www.sdspeople.fudan.edu.cn/zywei/data/emnlp2019-cmdd.zip)

预处理训练数据，将在data文件夹下生成processed文件夹
```
cd data
python preprocess.py
```

## 2. Training

```
python train.py
```

## 3. Predicting

```
python test.py
```
将输出预测结果文件，`xxx.json`文件


## 4. P, R and F1 calculate

```
python call_f1.py 
```

根据本地实际文件路径，在代码中修改相关路径
