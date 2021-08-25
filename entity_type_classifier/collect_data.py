import json

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()
        assert all(next(iter(tensors.values())).size(0) == tensor.size(0) for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)

    def cuda(self):
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cuda()

    def cpu(self):
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cpu()


class EntityPromptDataset(DictDataset):

    def __init__(self, predict=None, subj_obj_pair=None, path=None, name=None, tokenizer=None, virtual_prompt=None,
                 features=None, rel_length=3, max_seq_length=256):
        self.rel2id = virtual_prompt.rel2id
        if 'no_relation' in self.rel2id:
            self.NA_NUM = self.rel2id['no_relation']
        elif 'NA' in self.rel2id:
            self.NA_NUM = self.rel2id['NA']
        else:
            self.NA_NUM = 0

        self.predict = predict
        if self.predict == "rel":
            self.subj_obj_pair = subj_obj_pair
        self.num_class = len(self.rel2id)
        self.rel_length = rel_length
        self.virtual_prompt = virtual_prompt

        if features is None:
            self.max_seq_length = max_seq_length
            with open(path + "/" + name + '-truncate.json', "r", encoding='utf-8') as f:
                features = json.loads(f.read())
            features = self.list2tensor(features, tokenizer)

        super().__init__(**features)

    def save(self, path=None, name=None):
        path = path + "/" + name + "/"
        np.save(path + "input_ids", self.tensors['input_ids'].numpy())
        np.save(path + "token_type_ids", self.tensors['token_type_ids'].numpy())
        np.save(path + "attention_mask", self.tensors['attention_mask'].numpy())
        np.save(path + "labels", self.tensors['labels'].numpy())
        np.save(path + "mlm_labels", self.tensors['mlm_labels'].numpy())

    @classmethod
    def load(cls, predict=None, subj_obj_pair=None, path=None, name=None, virtual_prompt=None, tokenizer=None):
        path = path + "/" + name + "/"
        features = {}
        features['input_ids'] = torch.Tensor(np.load(path + "input_ids.npy")).long()
        features['token_type_ids'] = torch.Tensor(np.load(path + "token_type_ids.npy")).long()
        features['attention_mask'] = torch.Tensor(np.load(path + "attention_mask.npy")).long()
        features['labels'] = torch.Tensor(np.load(path + "labels.npy")).long()
        features['mlm_labels'] = torch.Tensor(np.load(path + "mlm_labels.npy")).long()
        res = cls(predict=predict, subj_obj_pair=subj_obj_pair, virtual_prompt=virtual_prompt, features=features,
                  tokenizer=tokenizer)
        return res

    def list2tensor(self, data, tokenizer):
        res = {}
        res['input_ids'] = []
        res['token_type_ids'] = []
        res['attention_mask'] = []
        res['mlm_labels'] = []
        res['labels'] = []

        for index, item in enumerate(tqdm(data)):
            subj_obj_pair = (item['subj_type'], item['obj_type'])
            if self.predict == 'rel' and not subj_obj_pair == self.subj_obj_pair:
                continue
            input_ids, token_type_ids = self.tokenize(item, tokenizer)
            attention_mask = [1] * len(input_ids)
            padding_length = self.max_seq_length - len(input_ids)

            if padding_length > 0:
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([0] * padding_length)

            assert len(input_ids) == self.max_seq_length
            assert len(attention_mask) == self.max_seq_length
            assert len(token_type_ids) == self.max_seq_length

            # label = self.rel2id[i['relation']]
            if self.predict == "subj":
                label = self.virtual_prompt.subj2id[item['subj_type']]
            elif self.predict == "obj":
                label = self.virtual_prompt.obj2id[item['obj_type']]
            elif self.predict == "rel":
                if item['relation'] not in self.virtual_prompt.subj_obj_pair2rel2id[self.subj_obj_pair]:
                    label = self.virtual_prompt.subj_obj_pair2rel2id[self.subj_obj_pair]['no_relation']
                else:
                    label = self.virtual_prompt.subj_obj_pair2rel2id[self.subj_obj_pair][item['relation']]
            else:
                raise ValueError("unkown prediting type")
            res['input_ids'].append(np.array(input_ids))
            res['token_type_ids'].append(np.array(token_type_ids))
            res['attention_mask'].append(np.array(attention_mask))
            res['labels'].append(np.array(label))
            mask_pos = np.where(res['input_ids'][-1] == tokenizer.mask_token_id)[0]
            mlm_labels = np.ones(self.max_seq_length) * (-1)
            mlm_labels[mask_pos] = 1
            res['mlm_labels'].append(mlm_labels)  # 把<mask>的位置置为1,其余是-1
        for key in res:  # 转化为tensor
            res[key] = np.array(res[key])
            res[key] = torch.Tensor(res[key]).long()
        return res

    def tokenize(self, item, tokenizer):

        sentence = item['token']
        subj = ' '.join(item['subj'])
        obj = ' '.join(item['obj'])
        rel_name = item['relation']

        sentence = " ".join(sentence)
        sentence = tokenizer.encode(sentence, add_special_tokens=False)

        e1 = tokenizer.encode(" ".join(['was', subj]), add_special_tokens=False)[1:]  # 在was XXX语境下的的tokens

        e2 = tokenizer.encode(" ".join(['was', obj]), add_special_tokens=False)[1:]
        hint_token = tokenizer.encode('is a', add_special_tokens=False)
        and_token = tokenizer.encode('and', add_special_tokens=False)

        # prompt =  [tokenizer.unk_token_id, tokenizer.unk_token_id] + \
        # 这里把template和实体mention串起来,类似于the person <e1> was member of the orgnization <e2>
        if self.predict == "subj":
            prompt = e1 + hint_token + [tokenizer.mask_token_id]
        elif self.predict == "obj":
            prompt = e2 + hint_token + [tokenizer.mask_token_id]
        elif self.predict == 'rel':
            prompt = e1 + [tokenizer.mask_token_id] * self.rel_length + e2
        else:
            raise ValueError("unknown prediction type")
        # prompt = e1+hint_token + [tokenizer.mask_token_id]+and_token+e2+hint_token+[tokenizer.mask_token_id]+and_token+e1+[tokenizer.mask_token_id]*3+e2

        tokens = sentence + prompt
        # 太长了就截短
        tokens = self.truncate(tokens, max_length=self.max_seq_length - tokenizer.num_special_tokens_to_add(False))

        input_ids = tokenizer.build_inputs_with_special_tokens(tokens)  # 这时候再加上special tokens,[cls][sep]
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens)  # token type,区别真正的和padding
        assert len(input_ids) == len(token_type_ids)
        return input_ids, token_type_ids

    def truncate(self, seq, max_length):
        if len(seq) <= max_length:
            return seq
        else:
            # print("=========")
            return seq[len(seq) - max_length:]


def collect_raw_data(data_path):
    with open(data_path, 'r', encoding="utf-8") as f:
        data = json.loads(f.read())
    return data
