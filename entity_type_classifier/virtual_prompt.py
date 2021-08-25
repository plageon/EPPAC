import torch
import json


class VirutalPrompt():
    def __init__(self, raw_temp=None, tokenizer=None, mlm=None, datadir=None, rel_length=2):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.rel_length = rel_length
        self.temp = raw_temp

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
        self.class2embed = self.initialize_temp(raw=raw_temp)
        self.subj2embed, self.obj2embed = self.subj_obj2embed()
        self.subj_obj_pair2rel2id, self.subj_obj_pair2rel2embed = self.rel2embed()

    def initialize_temp(self, raw):
        class2embed = {}
        for name, labels in raw.items():
            raw_relation = ' '.join([labels[1], labels[2], labels[3]])
            obj = labels[4]
            subj = labels[0]
            obj = 'website' if labels[4] == 'url' else obj
            s_tokens = self.tokenizer.encode("the " + subj, add_special_tokens=False, return_tensors='pt')[0][1:]
            s_embed = self.mlm.embeddings.word_embeddings(s_tokens)
            embed_dim = s_embed.size(-1)
            s_embed = s_embed.view(embed_dim)
            o_tokens = self.tokenizer.encode("the " + obj, add_special_tokens=False, return_tensors='pt')[0][1:]
            o_embed = self.mlm.embeddings.word_embeddings(o_tokens).view(embed_dim)
            rel_tokens = self.tokenizer.encode(raw_relation, add_special_tokens=False, return_tensors='pt')
            if rel_tokens.size(1) != self.rel_length:
                rel_embed = self.standarize_rel(labels[0], obj, raw_relation).view(self.rel_length, embed_dim)
            else:
                rel_embed = self.mlm.embeddings.word_embeddings(rel_tokens).view(self.rel_length, embed_dim)
            temp_embed = torch.stack([s_embed] + [w for w in rel_embed] + [o_embed])
            assert temp_embed.size(0) == self.rel_length + 2, print(labels)
            # print(temp_embed.size())
            class2embed[name] = temp_embed
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
            s_token = self.tokenizer.encode("the " + type, add_special_tokens=False, return_tensors='pt')[0][1:]
            subj2embed[id] = self.mlm.embeddings.word_embeddings(s_token)
        for type, id in self.classes['obj_types'].items():
            if '_' in type:
                type = obj_truncate[type]
            o_token = self.tokenizer.encode("the " + type, add_special_tokens=False, return_tensors='pt')[0][1:]
            obj2embed[id] = self.mlm.embeddings.word_embeddings(o_token)
        return subj2embed, obj2embed

    def rel2embed(self):
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
                    rel_embed = self.class2embed[rel][1:-1]
                    assert rel_embed.size(0) == self.rel_length
                    pair2embed.append(rel_embed)
                rel2id[self.id2subj_obj_pair[pair_id]] = rel_dict
                rel2embed[self.id2subj_obj_pair[pair_id]] = pair2embed
        return rel2id, rel2embed
