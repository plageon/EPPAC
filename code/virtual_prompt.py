import torch
import json


class VirutalPrompt():
    def __init__(self, raw_temp=None, tokenizer=None, mlm=None, datadir=None,subj_length=None,obj_length=None, rel_length=None, random_centers=False,
                 single_level=False):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.subj_length=subj_length
        self.obj_length=obj_length
        self.rel_length = rel_length
        self.temp = raw_temp
        self.random = random_centers

        if single_level:
            with open(datadir + '/rel2id.json', 'r', encoding='utf-8') as f:
                self.rel2id=json.loads(f.read())
            #self.rel2id = self.single_rel2id()
            self.rel2embed = self.single_rel2embed()
        else:
            with open(datadir + '/classes.json', 'r', encoding='utf-8') as f:
                self.classes = json.loads(f.read())

            self.subj2id = self.classes['subj_types']
            self.id2subj = {v: k for k, v in self.subj2id.items()}
            self.obj2id = self.classes['obj_types']
            self.id2obj = {v: k for k, v in self.obj2id.items()}
            self.rel2id = self.classes['relation_classes']
            self.subj_obj_pair_id2rel = self.classes["subj_obj2relation"]
            self.subj_obj_pair2id = {tuple(v): int(k) for k, v in self.classes["subj_obj_pairs"].items()}
            self.id2subj_obj_pair = {v: k for k, v in self.subj_obj_pair2id.items()}
            self.rel2embed = self.initialize_temp()
            self.subj2embed, self.obj2embed = self.subj_obj2embed()
            self.subj_obj_pair2rel2id, self.subj_obj_pair2rel2embed = self.pair2rel2embed()

    def initialize_temp(self):
        class2embed = {}
        for name, id in self.rel2id.items():
            if self.random:
                class2embed[name] = torch.rand(self.rel_length, 768) / 10
            else:
                labels=self.temp[name]
                raw_relation = ' '.join([labels[1], labels[2], labels[3]])
                rel_tokens = self.tokenizer.encode(raw_relation, add_special_tokens=False, return_tensors='pt')
                if rel_tokens.size(1) != self.rel_length:
                    rel_embed = self.standarize_rel(labels[0], labels[4], raw_relation).view(self.rel_length, -1)
                else:
                    rel_embed = self.mlm.embeddings.word_embeddings(rel_tokens).view(self.rel_length, -1)
                class2embed[name] = rel_embed
        return class2embed

    def standarize_rel(self, raw_s, raw_o, raw_rel, ):
        seq = "the " + raw_s + " " + raw_rel + " the " + raw_o + " means the " + raw_s + self.tokenizer.mask_token * self.rel_length + " the " + raw_o
        tokens = self.tokenizer.encode(seq, return_tensors='pt')
        out = self.mlm(input_ids=tokens, ).last_hidden_state
        rel_out = out[tokens == self.tokenizer.mask_token_id]

        return rel_out

    def so_pair2rel(self, raw_temps):
        so2rel = {}
        for name, temp in raw_temps:
            so_pair = (temp[0], temp[4])
            if so_pair not in so2rel:
                so2rel[so_pair] = [name]
            else:
                so2rel[so_pair].append(name)
        return so2rel

    def subj_obj2embed(self):
        subj2embed = {}
        obj2embed = {}
        obj_truncate = {
            'cause_of_death': 'death',
            'criminal_charge': 'conviction',
            'state_or_province': 'state',
        }
        for type, id in self.classes['subj_types'].items():
            if self.random:
                subj2embed[id] = torch.rand(self.subj_length, 768) / 10
            else:
                s_token = self.tokenizer.encode("the " + type, add_special_tokens=False, return_tensors='pt')[0][1:]
                subj2embed[id] = self.mlm.embeddings.word_embeddings(s_token)
        for type, id in self.classes['obj_types'].items():
            if '_' in type:
                type = obj_truncate[type]
            if self.random:
                obj2embed[id] = torch.rand(self.obj_length, 768) / 10
            else:
                o_token = self.tokenizer.encode("the " + type, add_special_tokens=False, return_tensors='pt')[0][1:]
                obj2embed[id] = self.mlm.embeddings.word_embeddings(o_token)
        return subj2embed, obj2embed

    def pair2rel2embed(self):
        rel2id = {}
        rel2embed = {}
        for pair_id, rel_dict in self.subj_obj_pair_id2rel.items():
            pair_id = int(pair_id)
            if len(rel_dict) == 1:
                rel2id[self.id2subj_obj_pair[pair_id]] = None
                rel2embed[self.id2subj_obj_pair[pair_id]] = None
            else:
                pair2embed = []
                for rel, id in rel_dict.items():
                    rel_embed = self.rel2embed[rel]
                    assert rel_embed.size(0) == self.rel_length
                    pair2embed.append(rel_embed)
                rel2id[self.id2subj_obj_pair[pair_id]] = rel_dict
                rel2embed[self.id2subj_obj_pair[pair_id]] = pair2embed
        return rel2id, rel2embed

    def single_rel2id(self):
        rel2id = {}
        for name in self.temp.keys():
            rel2id[name] = len(rel2id)
        return rel2id

    def single_rel2embed(self):
        class2embed = {}
        for name, id in self.rel2id.items():
            if self.random:
                class2embed[id] = torch.rand(self.rel_length, 768) / 10
            else:
                labels = self.temp[name]
                raw_relation = ' '.join(labels)
                rel_tokens = self.tokenizer.encode(raw_relation, add_special_tokens=False, return_tensors='pt')
                if rel_tokens.size(1) != self.rel_length:
                    rel_embed = self.standarize_rel("entity", "entity", raw_relation).view(self.rel_length, -1)
                else:
                    rel_embed = self.mlm.embeddings.word_embeddings(rel_tokens).view(self.rel_length, -1)
                class2embed[id] = rel_embed
        return class2embed
