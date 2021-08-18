import logging
import tensorflow as tf
import os
import argparse
import random
import json


from collections import defaultdict
from model import MyModel
from utils import DataProcessor_MTL_BERT_Test as DataProcessor_Test
from utils import load_vocabulary
from utils import extract_kvpairs_in_bioes_type
from bert import modeling as bert_modeling


logger = logging.getLogger()

def get_vocab(args):
    """获得字典"""
    logger.info("loading vocab...")
    # w2i_char, i2w_char = load_vocabulary("./bert_model/chinese_L-12_H-768_A-12/vocab.txt")  # 单词表
    w2i_char, i2w_char = load_vocabulary("./bert_model/chinese_roberta_wwm_large_ext/vocab.txt")  # 单词表
    w2i_bio, i2w_bio = load_vocabulary(os.path.join(args.data_dir, "vocab_bio.txt"))  # BIO表
    w2i_attr, i2w_attr = load_vocabulary(os.path.join(args.data_dir, "vocab_attr.txt"))  # 实体归一化 [咳嗽 咳嗽 null null null null]
    w2i_type, i2w_type = load_vocabulary(os.path.join(args.data_dir, "vocab_type.txt")) # 实体属性 [1 1 null null null null null]
    vocab_dict = {
        "w2i_char": w2i_char,
        "i2w_char": i2w_char,
        "w2i_bio": w2i_bio,
        "i2w_bio": i2w_bio,
        "w2i_attr": w2i_attr,
        "i2w_attr": i2w_attr,
        "w2i_type": w2i_type,
        "i2w_type": i2w_type
    }
    return vocab_dict

def get_predict_feature_data(args, vocab_dict):
    data_processor_test = DataProcessor_Test(
        os.path.join(args.test_input_file),
        vocab_dict['w2i_char'],
        vocab_dict['w2i_bio'],
        vocab_dict['w2i_attr'],
        vocab_dict['w2i_type'],
        shuffling=False
    )
    return data_processor_test

def build_model(args, vocab_dict):
    """初始化模型"""
    logger.info("building model...")
    # bert_config_path = "./bert_model/chinese_L-12_H-768_A-12/bert_config.json"
    bert_config_path = "./bert_model/chinese_roberta_wwm_large_ext/bert_config.json"
    bert_config = bert_modeling.BertConfig.from_json_file(bert_config_path)

    model = MyModel(bert_config=bert_config,
                    vocab_size_bio=len(vocab_dict['w2i_bio']),
                    vocab_size_attr=len(vocab_dict['w2i_attr']),
                    vocab_size_type=len(vocab_dict['w2i_type']),
                    O_tag_index=vocab_dict['w2i_bio']["O"],
                    use_lstm=False,
                    use_crf=args.use_crf) #改

    logger.info("model params:")
    params_num_all = 0
    for variable in tf.trainable_variables():
        params_num = 1
        for dim in variable.shape:
            params_num *= dim
        params_num_all += params_num
        logger.info("\t {} {} {}".format(variable.name, variable.shape, params_num))
    logger.info("all params num: " + str(params_num_all))
    return model

def predict_evaluate(sess, model, data_processor, vocab_dict, max_batches=None, batch_size=1024):
    chars_seq = []
    preds_kvpair = []
    eids = []
    batches_sample = 0

    while True:
        (inputs_seq_batch,
         inputs_mask_batch,
         inputs_segment_batch,
         eids_batch) = data_processor.get_batch(batch_size)

        feed_dict = {
            model.inputs_seq: inputs_seq_batch,
            model.inputs_mask: inputs_mask_batch,
            model.inputs_segment: inputs_segment_batch
        }

        preds_seq_bio_batch, preds_seq_attr_batch, preds_seq_type_batch = sess.run(model.outputs, feed_dict)

        for pred_seq_bio, pred_seq_attr, pred_seq_type, input_seq, mask, eid in zip(preds_seq_bio_batch, preds_seq_attr_batch, preds_seq_type_batch, inputs_seq_batch,inputs_mask_batch, eids_batch):

            l = sum(mask) - 2

            pred_seq_bio = [vocab_dict['i2w_bio'][i] for i in pred_seq_bio[1:-1][:l]]
            char_seq = [vocab_dict['i2w_char'][i] for i in input_seq[1:-1][:l]]
            pred_seq_attr = [vocab_dict['i2w_attr'][i] for i in pred_seq_attr[1:-1][:l]]
            pred_seq_type = [vocab_dict['i2w_type'][i] for i in pred_seq_type[1:-1][:l]]

            pred_kvpair = extract_kvpairs_in_bioes_type(pred_seq_bio, char_seq, pred_seq_attr,
                                                        pred_seq_type)  # (attr,type,word)
            preds_kvpair.append(pred_kvpair) # {(attrs,types, words)}
            eids.append(eid)

        if data_processor.end_flag:
            data_processor.refresh()
            break

        batches_sample += 1
        if (max_batches is not None) and (batches_sample >= max_batches):
            break
    return (preds_kvpair, eids)

def predict(args, model, data_processor_test, vocab_dict):
    """预测并输出结果"""
    # meta_path = os.path.join(args.save_dir, 'best_model.ckpt.meta')
    ckpt_path = os.path.join(args.save_dir, 'best_model.ckpt')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        # saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, ckpt_path)

        (preds_kvpair, eids) = predict_evaluate(sess, model, data_processor_test, vocab_dict, max_batches=2000, batch_size=32)
       
        # 将相同example id的数据以一定规则进行整合，得到样本级别的症状识别结果，用于评估。
        outputs = defaultdict(list)
        for i in range(len(eids)):
            if len(preds_kvpair[i]) != 0:
                outputs[eids[i]].extend(preds_kvpair[i])
        for eid, pairs in outputs.items():
            tmp_pred = defaultdict(list)
            if len(pairs) != 0:
                for pair in pairs:
                    tmp_pred[pair[0]].append(pair[1])
            for k, v in tmp_pred.items():
                new_v = max(v, key=v.count)
                tmp_pred[k] = new_v
            # 如果key 或 value为 null，则删除
            tmp_pred_new = {}
            for k, v in tmp_pred.items():
                if k != 'null' and v != 'null':
                    tmp_pred_new[k] = v
            outputs[eid] = tmp_pred_new
        # 将那些预测为空的样本id也存入进来，防止输出的样本缺失
        for eid in eids:
            if eid not in outputs:
                outputs[eid] = {}
        print("测试样本数量为：", len(outputs))
        pred_path = os.path.join(args.test_output_file)

        with open(pred_path, 'w', encoding='utf-8') as json_file:
            json.dump(outputs, json_file, ensure_ascii=False, indent=4)
        print('=========end prediction===========')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-dd', type=str, default='data/near_data_256', help='Train/dev data path')
    parser.add_argument('--save_dir', '-sd', type=str, default='save_model', help='Path to save, load model')
    parser.add_argument('--test_input_file', '-tif', type=str, default='dataset/test.json', help='Input file for prediction')
    parser.add_argument('--test_output_file', '-tof', type=str, default='roberta-xlarge-256.json', help='Output file for prediction')
    parser.add_argument('--word_embedding_dim', '-wed', type=int, default=300, help='Word embedding dim')
    parser.add_argument('--encoder_hidden_dim', '-ehd', type=int, default=300, help='LSTM encoder hidden dim')
    parser.add_argument('--use_crf', '-crf', action='store_true', default=True, help='Whether to use CRF')
    args = parser.parse_args()
    vocab_dict = get_vocab(args)
    model = build_model(args, vocab_dict)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    data_processor_test = get_predict_feature_data(args, vocab_dict)
    predict(args, model, data_processor_test, vocab_dict)
