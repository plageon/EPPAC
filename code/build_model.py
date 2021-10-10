import torch
from torch.nn import functional as F


class ETypePromptModel(torch.nn.Module):

    def __init__(self, mlm=None, virtualprompt=None, predict=None, subj_obj_pair=None, similarity=None, ):
        super().__init__()
        self.predict = predict
        self.mlm = mlm
        self.similarity = similarity
        if self.predict == "subj":
            self.num_classes = len(virtualprompt.subj2embed)
            self.label2embed = torch.nn.Parameter(
                torch.stack([virtualprompt.subj2embed[i] for i in range(self.num_classes)]))
            self.len_label = self.label2embed.size(1)
        elif self.predict == "obj":
            self.num_classes = len(virtualprompt.obj2embed)
            self.label2embed = torch.nn.Parameter(
                torch.stack([virtualprompt.obj2embed[i] for i in range(self.num_classes)]))
            self.len_label = self.label2embed.size(1)
        elif self.predict == "rel":
            if subj_obj_pair == "None":
                self.num_classes = len(virtualprompt.rel2id)
                self.label2embed = torch.nn.Parameter(
                    torch.stack(
                        [virtualprompt.rel2embed[i] for i in range(self.num_classes)]))
                self.len_label = self.label2embed.size(1)
            else:
                self.num_classes = len(virtualprompt.subj_obj_pair2rel2embed[subj_obj_pair])
                self.label2embed = torch.nn.Parameter(
                    torch.stack(
                        [virtualprompt.subj_obj_pair2rel2embed[subj_obj_pair][i] for i in range(self.num_classes)]))
                self.len_label = self.label2embed.size(1)
        else:
            raise ValueError("unknown prediction type")
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(self.len_label * 768, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.num_classes)
        )
        # self.softmax = torch.nn.Softmax(dim=1)
        self.norm = torch.nn.LayerNorm(self.num_classes)
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, input_ids, attention_mask, token_type_ids, mlm_labels, labels):
        embeddings = self.mlm.embeddings.word_embeddings(input_ids)  # 预训练模型的embedding
        hidden_states = self.mlm(inputs_embeds=embeddings,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids).last_hidden_state
        # print(hidden_states)

        if self.similarity == "dense": #use dense neural network as a classifier
            hidden_states_mask = hidden_states[mlm_labels >= 0].view(hidden_states.size(0), -1)
            logits = self.linear(hidden_states_mask)
        elif self.similarity == "cdist1": # Manhattan distance as similarity measure
            hidden_states_mask = hidden_states[mlm_labels >= 0].view(hidden_states.size(0), -1)
            embedmatrix = self.label2embed.view(self.num_classes, -1)
            out = torch.cdist(hidden_states_mask, embedmatrix, p=1)
            normalized_out = self.norm(out)
            logits = -normalized_out
        elif self.similarity == "cdist2": #Euclidean distance as similarity measure
            hidden_states_mask = hidden_states[mlm_labels >= 0].view(hidden_states.size(0), -1)
            embedmatrix = self.label2embed.view(self.num_classes, -1)
            out = torch.cdist(hidden_states_mask, embedmatrix, p=2)
            normalized_out = self.norm(out)
            logits = -normalized_out
        else:
            hidden_states_mask = hidden_states[mlm_labels >= 0].view(hidden_states.size(0), self.len_label, -1)
            logits = self.vector_similarity(hidden_states_mask)
        return logits  #

    def vector_similarity(self, batchs):
        logits = []
        for batch in batchs:
            if self.similarity == "mm": #dot product as similarity measure
                out = [
                    torch.sum(torch.mm(batch, self.label2embed[i].transpose(1, 0)).diag(), dim=(0,), keepdim=True)
                    for i in range(self.num_classes)]
            elif self.similarity == "cos": #cosine similarity as similarity measure
                out = [torch.sum(self.cos(batch, self.label2embed[i]), dim=(0,), keepdim=True) for i
                       in range(self.num_classes)]
            else:
                raise ValueError("unknown similiarty")
            logits.append(torch.cat(out, 0))
        logits = torch.stack(logits, 0)
        logits=self.norm(logits) if self.similarity=="cos" else logits
        return logits


class FullModel(torch.nn.Module):
    def __init__(self, subj_model_path, obj_model_path, rel_models_path, mlm=None, virtualprompt=None,
                 similarity=None, ):
        super().__init__()
        self.subj_model = ETypePromptModel(mlm=mlm, virtualprompt=virtualprompt, similarity=similarity, predict="subj")
        self.subj_model.load_state_dict(torch.load(subj_model_path))
        self.obj_model = ETypePromptModel(mlm=mlm, virtualprompt=virtualprompt, similarity=similarity, predict="obj")
        self.obj_model.load_state_dict(torch.load(obj_model_path))
        self.rel_models = {}
        for pair, path in rel_models_path:
            rel_model = ETypePromptModel(mlm=mlm, virtualprompt=virtualprompt, similarity=similarity, predict="rel")
            rel_model.load_state_dict(torch.load(path))
            self.rel_models[pair] = rel_model

    def forward(self):
        pass
