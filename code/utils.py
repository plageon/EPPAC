import json
import random
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, BertModel, \
    RobertaConfig, RobertaTokenizer, RobertaModel, \
    AlbertTokenizer, AlbertConfig, AlbertModel


def get_tokenizer_mlm(model_type, model_name):
    model_classes = {
        'bert': {
            'config': BertConfig,
            'tokenizer': BertTokenizer,
            'model': BertModel,
        },
        'roberta': {
            'config': RobertaConfig,
            'tokenizer': RobertaTokenizer,
            'model': RobertaModel,
        },
        'albert': {
            'config': AlbertConfig,
            'tokenizer': AlbertTokenizer,
            'model': AlbertModel,
        }
    }
    model_config = model_classes[model_type]
    tokenizer = model_config['tokenizer'].from_pretrained(model_name)
    mlm = model_config['model'].from_pretrained(model_name)
    return tokenizer, mlm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_single_temps(data_dir):
    temps = {}
    with open(data_dir + "/temp.txt", "r") as f:
        for i in f.readlines():
            i = i.strip().split("\t")
            info = {}
            info['name'] = i[1].strip()
            info['labels'] = [i[2], i[3]]
            temps[info['name']] = info['labels']
    return temps


def get_raw_temps(data_dir):
    temps = {}
    with open(data_dir + "/temp.txt", "r") as f:
        for i in f.readlines():
            i = i.strip().split("\t")
            info = {}
            info['name'] = i[1].strip()
            info['labels'] = [i[2], i[3], i[4], i[5], i[6]]
            temps[info['name']] = info['labels']
    return temps

#calculate f1 score. If output_raw=True, then return original number of gold guess and correct
def f1_score(output, label, rel_num, na_num, output_raw=False):
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    for i in range(len(output)):
        guess = output[i]
        gold = label[i]

        if guess == na_num:  # 如果预测的是无关系，置为0
            guess = 0
        elif guess < na_num:  # 前面的全部移动一位
            guess += 1

        if gold == na_num:
            gold = 0
        elif gold < na_num:
            gold += 1

        if gold == 0 and guess == 0:
            continue
        if gold == 0 and guess != 0:
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1

    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for i in range(1, rel_num):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]
        if recall + precision > 0:
            f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    micro_f1 = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())
        micro_f1 = 2 * recall * prec / (recall + prec)
    else:
        prec, recall, micro_f1 = -1, -1, -1
    if output_raw:
        return sum(gold_by_relation.values()), sum(guess_by_relation.values()), sum(correct_by_relation.values())
    else:
        return prec, recall, micro_f1, f1_by_relation


def acc(preds, all_labels):
    correct = 0
    for pred, label in zip(preds, all_labels):
        if pred == label:
            correct += 1
    acc = correct / len(preds)
    return acc

#evaluate classifier performance
def evaluate(model, val_dataset, val_dataloader):
    model.eval()
    scores = []
    all_labels = []

    with torch.no_grad():
        for batch in iter(tqdm(val_dataloader, desc="Val Iteration")):
            for key, tensor in batch.items():
                batch[key] = tensor.cuda()
            logits = model(**batch)
            labels = batch['labels'].detach().cpu().tolist()
            all_labels += labels
            scores.append(logits.cpu().detach())
        scores = torch.cat(scores, 0)  # 把不同的batch归到一起
        scores = scores.detach().cpu().numpy()
        all_labels = np.array(all_labels)
        # np.save("scores.npy", scores)
        # np.save("all_labels.npy", all_labels)

        pred = np.argmax(scores, axis=-1)  # 选择最大的作为预测结果
        prec, recall, mi_f1, ma_f1 = f1_score(pred, all_labels, val_dataset.num_class, val_dataset.NA_NUM)
        accuracy = acc(pred, all_labels)
        return accuracy, prec, recall, mi_f1, ma_f1


class RelDict():
    def __init__(self, path):
        with open(path, "r") as f:
            self.rel_dict = json.loads(f.read())
        if not 'NA' in self.rel_dict:
            self.NA_NUM = self.rel_dict['no_relation']
        else:
            self.NA_NUM = self.rel_dict['NA']

    def rel2id(self, name):
        return self.rel_dict[name]

    def id2rel(self, id):
        inverse = {v: k for k, v in self.rel_dict.items()}
        return inverse[id]
