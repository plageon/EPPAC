import datetime

import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm, trange
import numpy as np
import json

from entity_type_classifier.build_model import ETypePromptModel
from entity_type_classifier.collect_data import EntityPromptDataset, collect_raw_data
from entity_type_classifier.optimizing import get_optimizer, get_optimizer4temp
from entity_type_classifier.utils import get_tokenizer_mlm, set_seed, get_raw_temps, evaluate, f1_score
from entity_type_classifier.virtual_prompt import VirutalPrompt


def train_sub_model(dataset_name=None, predict=None, subj_obj_pair_id=None, reload_data=True, train=True,
                    eval_kpl=False, evaluate_sub_model=True):
    model_type = 'roberta'
    model_name = 'roberta-base'
    tokenizer, mlm = get_tokenizer_mlm(model_type, model_name)
    n_gpu = torch.cuda.device_count()

    data_dir = "./datasets/" + dataset_name
    output_dir = "./results/" + dataset_name
    per_gpu_train_batch_size = 8
    gradient_accumulation_steps = 1
    max_seq_length = 256
    warmup_steps = 500
    learning_rate = 3e-5
    learning_rate_for_new_token = 1e-5
    num_train_epochs = 5
    weight_decay = 1e-2
    adam_epsilon = 1e-6
    lr_temp = 1e-5
    max_grad_norm = 1.0
    vec_sim = "mm"
    rel_length = 3
    set_seed(123)

    raw_temp = get_raw_temps(data_dir)
    virtual_prompt = VirutalPrompt(raw_temp=raw_temp, tokenizer=tokenizer, mlm=mlm, rel_length=rel_length,
                                   datadir=data_dir)
    subj_obj_pair = None
    if not subj_obj_pair_id == None:
        subj_obj_pair = virtual_prompt.id2subj_obj_pair[subj_obj_pair_id]
    if predict == "rel":
        if virtual_prompt.subj_obj_pair2id[subj_obj_pair] == None:
            return None
    # """
    if reload_data:
        dataset = EntityPromptDataset(
            predict=predict,
            subj_obj_pair=subj_obj_pair,
            path=data_dir,
            name='train',
            max_seq_length=max_seq_length,
            rel_length=rel_length,
            virtual_prompt=virtual_prompt,
            tokenizer=tokenizer, )
        dataset.save(path=output_dir, name="train")
        # If the dataset has been saved,
        # the code ''dataset = REPromptDataset(...)'' is not necessary.
        dataset = EntityPromptDataset(
            predict=predict,
            subj_obj_pair=subj_obj_pair,
            path=data_dir,
            name='dev',
            max_seq_length=max_seq_length,
            rel_length=rel_length,
            virtual_prompt=virtual_prompt,
            tokenizer=tokenizer)
        dataset.save(path=output_dir, name="dev")

        # If the dataset has been saved,
        # the code ''dataset = REPromptDataset(...)'' is not necessary.

        dataset = EntityPromptDataset(
            predict=predict,
            subj_obj_pair=subj_obj_pair,
            path=data_dir,
            name='test',
            max_seq_length=max_seq_length,
            rel_length=rel_length,
            virtual_prompt=virtual_prompt,
            tokenizer=tokenizer)
        dataset.save(path=output_dir, name="test")

    train_dataset = EntityPromptDataset.load(
        predict=predict,
        subj_obj_pair=subj_obj_pair,
        path=output_dir,
        name="train",
        virtual_prompt=virtual_prompt,
        tokenizer=tokenizer, )

    val_dataset = EntityPromptDataset.load(
        predict=predict,
        subj_obj_pair=subj_obj_pair,
        path=output_dir,
        name="dev",
        virtual_prompt=virtual_prompt,
        tokenizer=tokenizer, )

    test_dataset = EntityPromptDataset.load(
        predict=predict,
        subj_obj_pair=subj_obj_pair,
        path=output_dir,
        name="test",
        virtual_prompt=virtual_prompt,
        tokenizer=tokenizer, )

    train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
    # train_dataset.cuda()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    # val_dataset.cuda()
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=train_batch_size // 2)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=train_batch_size // 2)

    # label2temp = torch.nn.Parameter(torch.stack([virtual_prompt.class2temp[rel_dict.id2rel(i)] for i in range(len(virtual_prompt.class2temp))]))
    model = ETypePromptModel(mlm=mlm, virtualprompt=virtual_prompt, similarity=vec_sim, predict=predict,
                             subj_obj_pair=subj_obj_pair)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        model.cuda()
    optimizer, scheduler = get_optimizer(model, train_dataloader, gradient_accumulation_steps, num_train_epochs,
                                         learning_rate, adam_epsilon, warmup_steps, weight_decay)  # 将lm做fine tune
    optimizer4temp, scheduler4temp = get_optimizer4temp(model, lr_temp)
    criterion = nn.CrossEntropyLoss()
    mx_res = 0.0
    best_model_path = ''
    best_model = dict()
    best_epoch = 0
    his = {
        'acc': [],
        'prec': [],
        'recall': [],
        'mi_f1': [],
        'ma_f1': [],
    }

    if eval_kpl:
        model.load_state_dict(torch.load(output_dir + "/" + 'parameter' + '4' + ".pkl"))
    if train:
        print(" time=", datetime.datetime.now(),
              " dataset=", dataset_name,
              " model_type=", model_type,
              " model_name=", model_name,
              " per_gpu_train_batch_size=", per_gpu_train_batch_size,
              " gradient_accumulation_steps=", gradient_accumulation_steps,
              " max_seq_length=", max_seq_length,
              ' warmup_steps=', warmup_steps,
              ' learning_rate=', learning_rate,
              ' learning_rate_for_new_token=', learning_rate_for_new_token,
              ' num_train_epochs=', num_train_epochs,
              ' weight_decay=', weight_decay,
              ' adam_epsilon=', adam_epsilon,
              ' lr_temp=', lr_temp,
              ' max_grad_norm=', max_grad_norm,
              ' vector_similarity=', vec_sim,
              ' relation length=', rel_length,
              ' predicting=', predict,
              ' subj_obj_pair=', subj_obj_pair, )

        for epoch in trange(int(num_train_epochs), desc="Epoch"):
            # '''
            model.train()  # 启用drop out和batch normalization
            model.zero_grad()  # 把模型的梯度置零
            tr_loss = 0.0
            global_step = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                for key, tensor in batch.items():
                    batch[key] = tensor.cuda()
                logits = model(**batch)  # 正向传播

                loss = criterion(logits, batch['labels'])  # 分类的loss

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()  # 反向传播更新梯度
                tr_loss += loss.item()

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # 防止梯度爆炸
                    optimizer.step()
                    scheduler.step()
                    optimizer4temp.step()
                    scheduler4temp.step()
                    model.zero_grad()  # 重置权重为零
                    global_step += 1
                    # print(tr_loss / global_step, mx_res)
            # '''
            acc, prec, recall, mi_f1, ma_f1 = evaluate(model, val_dataset, val_dataloader)
            his['acc'].append(acc)
            his['prec'].append(prec)
            his['recall'].append(recall)
            his['mi_f1'].append(mi_f1)
            his['ma_f1'].append(ma_f1)
            # print("dev mi f1 ",mi_f1,"\ndev ma f1 ",ma_f1)
            if mi_f1 > mx_res:
                mx_res = mi_f1
                # best_model_path = output_dir + "/" + predict+str(subj_obj_pair)+str(datetime.datetime.now()) + 'parameter' + str(epoch) + ".pkl"
                best_model = model.state_dict()
                best_epoch = epoch
                # torch.save(best_model, best_model_path)
            # break
        best_model_path = output_dir + "/best_models/" + predict + str(subj_obj_pair) + ".pkl"
        torch.save(best_model, best_model_path)
        for k, v in his.items():
            print(k, v)

    if evaluate_sub_model:
        # test_acc, test_prec, test_recall, test_mi_f1, test_ma_f1 = evaluate(model, test_dataset, test_dataloader)
        print("best acc", his['acc'][best_epoch], "best prec", his['prec'][best_epoch], "best recall",
              his['recall'][best_epoch], "best mi f1 ", his['mi_f1'][best_epoch],
              "\nbest ma f1 ", his['ma_f1'][best_epoch])
        print("completed at", datetime.datetime.now())
        print("======================")
        return best_model_path


def evaluate_sub_model(dataset_name, subj_model_path, obj_model_path, rel_model_path, eval_set):
    model_type = 'roberta'
    model_name = 'roberta-base'
    tokenizer, mlm = get_tokenizer_mlm(model_type, model_name)

    data_dir = "./datasets/" + dataset_name
    output_dir = "./results/" + dataset_name
    temp_dir = output_dir + "/data_by_pairs"
    per_gpu_train_batch_size = 8
    n_gpu = torch.cuda.device_count()
    train_batch_size = per_gpu_train_batch_size * n_gpu
    vec_sim = "mm"
    rel_length = 3
    max_seq_length = 256
    set_seed(123)

    raw_temp = get_raw_temps(data_dir)
    virtual_prompt = VirutalPrompt(raw_temp=raw_temp, tokenizer=tokenizer, mlm=mlm, rel_length=rel_length,
                                   datadir=data_dir)
    subj_dataset = EntityPromptDataset(
        predict='subj',
        subj_obj_pair=None,
        path=data_dir,
        name=eval_set,
        max_seq_length=max_seq_length,
        rel_length=rel_length,
        virtual_prompt=virtual_prompt,
        tokenizer=tokenizer)
    subj_sampler = SequentialSampler(subj_dataset)
    subj_dataloader = DataLoader(subj_dataset, sampler=subj_sampler, batch_size=train_batch_size // 2)
    subj_model = ETypePromptModel(mlm=mlm, virtualprompt=virtual_prompt, similarity=vec_sim, predict="subj",
                                  subj_obj_pair=None)
    if torch.cuda.device_count() > 1:
        subj_model = torch.nn.DataParallel(subj_model)
        subj_model.cuda()
    subj_model.load_state_dict(torch.load(subj_model_path), strict=False)
    subj_model.eval()
    subj_scores = []

    with torch.no_grad():
        for batch in iter(tqdm(subj_dataloader, desc="Val Subj Iteration")):
            for key, tensor in batch.items():
                batch[key] = tensor.cuda()
            logits = subj_model(**batch)
            # labels = batch['labels'].detach().cpu().tolist()
            subj_scores.append(logits.cpu().detach())
        subj_scores = torch.cat(subj_scores, 0)  # 把不同的batch归到一起
        subj_scores = subj_scores.detach().cpu().numpy()
        # np.save("scores.npy", scores)
        # np.save("all_labels.npy", all_labels)

        subj_pred = np.argmax(subj_scores, axis=-1)

    obj_dataset = EntityPromptDataset(
        predict="obj",
        subj_obj_pair=None,
        path=data_dir,
        name=eval_set,
        max_seq_length=max_seq_length,
        rel_length=rel_length,
        virtual_prompt=virtual_prompt,
        tokenizer=tokenizer)
    obj_sampler = SequentialSampler(obj_dataset)
    obj_dataloader = DataLoader(obj_dataset, sampler=obj_sampler, batch_size=train_batch_size // 2)
    obj_model = ETypePromptModel(mlm=mlm, virtualprompt=virtual_prompt, similarity=vec_sim, predict="obj",
                                 subj_obj_pair=None)
    if torch.cuda.device_count() > 1:
        obj_model = torch.nn.DataParallel(obj_model)
        obj_model.cuda()
    obj_model.load_state_dict(torch.load(obj_model_path), strict=False)
    obj_model.eval()
    obj_scores = []

    with torch.no_grad():
        for batch in iter(tqdm(obj_dataloader, desc="Val obj Iteration")):
            for key, tensor in batch.items():
                batch[key] = tensor.cuda()
            logits = obj_model(**batch)
            # labels = batch['labels'].detach().cpu().tolist()
            obj_scores.append(logits.cpu().detach())
        obj_scores = torch.cat(obj_scores, 0)  # 把不同的batch归到一起
        obj_scores = obj_scores.detach().cpu().numpy()
        # np.save("scores.npy", scores)
        # np.save("all_labels.npy", all_labels)

        obj_pred = np.argmax(obj_scores, axis=-1)

    subj_obj_pair2data = {}
    for pair in virtual_prompt.subj_obj_pair2id.keys():
        subj_obj_pair2data[pair] = []

    total_gold, total_guess, total_correct = 0, 0, 0

    raw_data = collect_raw_data(data_dir + "/" + eval_set + "-truncate.json")
    for index, item in enumerate(raw_data):
        item_subj_pred = subj_pred[index]
        item_obj_pred = obj_pred[index]
        subj_obj_pair = (virtual_prompt.id2subj[item_subj_pred], virtual_prompt.id2obj[item_obj_pred])
        if subj_obj_pair not in virtual_prompt.subj_obj_pair2id.keys() or item['relation'] not in \
                virtual_prompt.subj_obj_pair2rel2id[subj_obj_pair].keys():
            # at this point, our guess is no_relation
            if item['relation'] == 'no_relation':
                continue
            else:
                total_gold += 1
        else:
            subj_obj_pair2data[subj_obj_pair].append(item)

    for pair in virtual_prompt.subj_obj_pair2id.keys():
        subj_obj_pair_file = temp_dir + "/" + str(pair) + "-truncate.json"
        with open(subj_obj_pair_file, 'w', encoding='utf-8') as f:
            json.dump(subj_obj_pair2data[pair], f)
    print(total_gold, total_guess, total_correct)
    for index, pair in virtual_prompt.id2subj_obj_pair.items():
        rel_dataset = EntityPromptDataset(
            predict="rel",
            subj_obj_pair=pair,
            path=output_dir,
            name='data_by_pairs/' + str(pair),
            max_seq_length=max_seq_length,
            rel_length=rel_length,
            virtual_prompt=virtual_prompt,
            tokenizer=tokenizer)
        rel_sampler = SequentialSampler(rel_dataset)
        rel_dataloader = DataLoader(rel_dataset, sampler=rel_sampler, batch_size=train_batch_size // 2)
        rel_model = ETypePromptModel(mlm=mlm, virtualprompt=virtual_prompt, similarity=vec_sim, predict="rel",
                                     subj_obj_pair=pair)
        if torch.cuda.device_count() > 1:
            rel_model = torch.nn.DataParallel(rel_model)
            rel_model.cuda()
        rel_model.load_state_dict(torch.load(rel_model_path[index]), strict=False)
        rel_model.eval()
        rel_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in iter(tqdm(rel_dataloader, desc="Val " + str(pair) + " Iteration")):
                for key, tensor in batch.items():
                    batch[key] = tensor.cuda()
                logits = rel_model(**batch)
                labels = batch['labels'].detach().cpu().tolist()
                all_labels += labels
                rel_scores.append(logits.cpu().detach())
            rel_scores = torch.cat(rel_scores, 0)  # 把不同的batch归到一起
            rel_scores = rel_scores.detach().cpu().numpy()
            all_labels = np.array(all_labels)
            # np.save("scores.npy", scores)
            # np.save("all_labels.npy", all_labels)

            rel_pred = np.argmax(rel_scores, axis=-1)
            gold, guess, correct = f1_score(rel_pred, all_labels, rel_dataset.num_class, rel_dataset.NA_NUM,
                                            output_raw=True)
            total_gold += gold
            total_guess += guess
            total_correct += correct
            print(total_gold, total_guess, total_correct)

    print(total_gold, total_guess, total_correct)
    total_recall = total_correct / total_gold
    total_prec = total_correct / total_guess
    total_f1 = 2 * total_recall * total_prec / (total_recall + total_prec)

    print('total recall=', total_recall,
          'total precision=', total_prec,
          'total f1 score=', total_f1, )


if __name__ == "__main__":
    SUBJ = False
    OBJ = False
    REL = True
    EVAL = True
    dataset_name = "retacred"
    eval_set = "test"
    model_path = "best-models.txt"
    if SUBJ:
        subj_model = train_sub_model(dataset_name=dataset_name, predict='subj')
        with open(model_path, 'a', encoding='utf-8') as f:
            f.write(subj_model + "\n")
    if OBJ:
        obj_model = train_sub_model(dataset_name=dataset_name, predict='obj')
        with open(model_path, 'a', encoding='utf-8') as f:
            f.write(obj_model + "\n")
    if REL:
        rel_models = []
        for subj_obj_pair_id in range(5, 27):
            rel_model = train_sub_model(dataset_name=dataset_name, predict='rel', subj_obj_pair_id=subj_obj_pair_id)
            rel_models.append(rel_model)
            with open(model_path, 'a', encoding='utf-8') as f:
                f.write(rel_model + "\n")
    if EVAL:
        models = []
        with open(model_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) > 0:
                    models.append(line)
        evaluate_sub_model(dataset_name, models[0], models[1], models[2:], eval_set)
