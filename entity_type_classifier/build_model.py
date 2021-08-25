import torch


class ETypePromptModel(torch.nn.Module):

    def __init__(self, mlm=None, virtualprompt=None, predict=None, subj_obj_pair=None, similarity=None, ):
        super().__init__()
        self.predict = predict
        self.mlm = mlm
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
            self.num_classes = len(virtualprompt.subj_obj_pair2rel2embed[subj_obj_pair])
            self.label2embed = torch.nn.Parameter(
                torch.stack([virtualprompt.subj_obj_pair2rel2embed[subj_obj_pair][i] for i in range(self.num_classes)]))
            self.len_label = self.label2embed.size(1)
        else:
            raise ValueError("unknown prediction type")

        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        self.similarity = similarity

    def forward(self, input_ids, attention_mask, token_type_ids, mlm_labels, labels):
        embeddings = self.mlm.embeddings.word_embeddings(input_ids)  # 预训练模型的embedding
        hidden_states = self.mlm(inputs_embeds=embeddings,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids).last_hidden_state
        # print(hidden_states)
        hidden_states_mask = hidden_states[mlm_labels >= 0].view(hidden_states.size(0), self.len_label, -1)
        logits = self.vector_similarity(hidden_states_mask)
        return logits  # 每个可能出现mask的位置的list

    def vector_similarity(self, batchs):
        logits = []
        if True:
            for batch in batchs:
                if self.similarity == "mm":
                    out = [
                        torch.sum(torch.mm(batch, self.label2embed[i].transpose(1, 0)).diag(), dim=(0,), keepdim=True)
                        for i in range(self.num_classes)]
                elif self.similarity == "cos":
                    out = [torch.sum(self.cos(batch, self.label2embed[i]), dim=(0,), keepdim=True) for i
                           in range(self.num_classes)]
                else:
                    raise ValueError("unknown similiarty")
                out = torch.cat(out, 0)
                logits.append(out)
            logits = torch.stack(logits, 0)
        else:
            hidden_states_mask = batchs.view(batchs.size(0), -1)
            if self.similarity == "mm":
                logits = torch.mm(hidden_states_mask, self.subj2embed.transpose(1, 0))
            elif self.similarity == "cos":
                for batch in hidden_states_mask:
                    out = [self.cos(batch, self.label2embed[i]) for i in range(self.num_classes)]
                    out = torch.cat(out, 0)
                    logits.append(out)
                logits = torch.stack(logits, 0)
            else:
                raise ValueError("unknown similiarty")

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
