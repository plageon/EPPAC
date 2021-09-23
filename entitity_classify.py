import datetime

import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm, trange
import numpy as np
import json

from entity_type_classifier.build_model import ETypePromptModel
from entity_type_classifier.collect_data import EntityPromptDataset, collect_raw_data
from entity_type_classifier.optimizing import get_optimizer, get_optimizer4temp, get_optimizer4dense
from entity_type_classifier.utils import get_tokenizer_mlm, set_seed, get_raw_temps, evaluate, f1_score
from entity_type_classifier.virtual_prompt import VirutalPrompt


def train_sub_model(model_scale=None, dataset_name=None, predict=None, subj_obj_pair_id=None, reload_data=True,
                    train=True, vec_sim=None, rel_length=None, random=False,
                    load_kpl=False, evaluate_sub_model=True, learning_rate=None, lr_temp=None, num_train_epochs=None):
    model_type = 'roberta'
    model_name = 'roberta-' + model_scale
    tokenizer, mlm = get_tokenizer_mlm(model_type, model_name)
    n_gpu = torch.cuda.device_count()

    data_dir = "./datasets/" + dataset_name
    output_dir = "./results/" + dataset_name
    per_gpu_train_batch_size = 8 if model_scale == "base" else 4
    gradient_accumulation_steps = 1
    max_seq_length = 512
    warmup_steps = 500

    num_train_epochs = num_train_epochs
    weight_decay = 1e-2
    adam_epsilon = 1e-6

    max_grad_norm = 1.0
    set_seed(123)

    raw_temp = get_raw_temps(data_dir)
    virtual_prompt = VirutalPrompt(raw_temp=raw_temp, tokenizer=tokenizer, mlm=mlm, rel_length=rel_length,
                                   datadir=data_dir, random=random)
    if predict == 'subj' and len(virtual_prompt.subj2id) == 1:
        return 'None'
    subj_obj_pair = None
    if not subj_obj_pair_id == None:
        subj_obj_pair = virtual_prompt.id2subj_obj_pair[subj_obj_pair_id]
    if predict == "rel":
        if virtual_prompt.subj_obj_pair2rel2id[subj_obj_pair] == None:
            return "None"

    print(" time=", datetime.datetime.now(),
          " dataset=", dataset_name,
          " model_type=", model_type,
          " model_name=", model_name,
          " per_gpu_train_batch_size=", per_gpu_train_batch_size,
          " gradient_accumulation_steps=", gradient_accumulation_steps,
          " max_seq_length=", max_seq_length,
          ' warmup_steps=', warmup_steps,
          ' learning_rate=', learning_rate,
          ' num_train_epochs=', num_train_epochs,
          ' weight_decay=', weight_decay,
          ' adam_epsilon=', adam_epsilon,
          ' lr_temp=', lr_temp,
          ' max_grad_norm=', max_grad_norm,
          ' vector_similarity=', vec_sim,
          ' relation length=', rel_length,
          ' predicting=', predict,
          ' subj_obj_pair=', subj_obj_pair, )
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
    optimizer4temp, scheduler4temp = get_optimizer4dense(model, lr_temp) if vec_sim == "dense" else get_optimizer4temp(
        model, lr_temp)
    criterion = nn.CrossEntropyLoss()
    mx_res = 0.0
    best_model_path = output_dir + "/best_models/" + predict + str(subj_obj_pair) + ".pkl"
    best_epoch = 0
    his = {
        'acc': [],
        'prec': [],
        'recall': [],
        'mi_f1': [],
        'ma_f1': [],
    }

    if load_kpl:
        model.load_state_dict(torch.load(best_model_path))
    if train:
        for epoch in trange(int(num_train_epochs), desc="Epoch"):
            # '''
            model.train()  # 启用drop out和batch normalization
            model.zero_grad()  # 把模型的梯度置零
            tr_loss = 0.0
            global_step = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                for key, tensor in batch.items():
                    if torch.cuda.device_count() > 1:
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
            if mi_f1 > mx_res or (epoch + 1 == num_train_epochs and mx_res == 0):
                mx_res = mi_f1
                # best_model_path = output_dir + "/" + predict+str(subj_obj_pair)+str(datetime.datetime.now()) + 'parameter' + str(epoch) + ".pkl"
                best_model = model.state_dict()
                best_epoch = epoch
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


def evaluate_sub_model(vec_sim, rel_length, model_scale, dataset_name, subj_model_path, obj_model_path, rel_model_path,
                       eval_set):
    model_type = 'roberta'
    model_name = 'roberta-' + model_scale
    tokenizer, mlm = get_tokenizer_mlm(model_type, model_name)

    data_dir = "./datasets/" + dataset_name
    output_dir = "./results/" + dataset_name
    temp_dir = output_dir + "/data_by_pairs"
    per_gpu_train_batch_size = 8 if model_scale == "base" else 4
    n_gpu = torch.cuda.device_count()
    train_batch_size = per_gpu_train_batch_size * n_gpu
    max_seq_length = 512
    set_seed(123)

    raw_temp = get_raw_temps(data_dir)
    virtual_prompt = VirutalPrompt(raw_temp=raw_temp, tokenizer=tokenizer, mlm=mlm, rel_length=rel_length,
                                   datadir=data_dir)

    # predict subject types
    if subj_model_path == 'None':
        pass
    else:
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

    # predict object types
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

    # predict relation type according to subj-obj pairs
    total, total_neg, total_gold, total_guess, total_correct = 0, 0, 0, 0, 0
    # put data into separate files first
    raw_data = collect_raw_data(data_dir + "/" + eval_set + "-truncate.json")
    for index, item in enumerate(raw_data):
        item_subj_pred = 0 if subj_model_path == 'None' else subj_pred[index]
        item_obj_pred = obj_pred[index]
        subj_obj_pair = (virtual_prompt.id2subj[item_subj_pred], virtual_prompt.id2obj[item_obj_pred])
        if subj_obj_pair not in virtual_prompt.subj_obj_pair2id.keys() or virtual_prompt.subj_obj_pair2rel2id[
            subj_obj_pair] == None or item['relation'] not in virtual_prompt.subj_obj_pair2rel2id[subj_obj_pair].keys():
            # at this point, our guess is no_relation
            total += 1
            if item['relation'] == 'no_relation':
                total_neg+=1
                continue
            else:
                total_gold += 1
        else:
            subj_obj_pair2data[subj_obj_pair].append(item)
    # load data from file
    for pair in virtual_prompt.subj_obj_pair2id.keys():
        #print(pair, len(subj_obj_pair2data[pair]))
        subj_obj_pair_file = temp_dir + "/" + str(pair) + "-truncate.json"
        with open(subj_obj_pair_file, 'w', encoding='utf-8') as f:
            json.dump(subj_obj_pair2data[pair], f)

    print(total, total_neg, total_gold, total_guess, total_correct)
    for index, pair in virtual_prompt.id2subj_obj_pair.items():
        if rel_model_path[index] == "None":
            # the prediction and gold-std is both no_relation for each item
            continue
        rel_dataset = EntityPromptDataset(
            predict="rel",
            subj_obj_pair=pair,
            path=output_dir,
            name='data_by_pairs/' + str(pair),
            max_seq_length=max_seq_length,
            rel_length=rel_length,
            virtual_prompt=virtual_prompt,
            tokenizer=tokenizer)
        if len(rel_dataset) == 0:
            continue
        rel_sampler = SequentialSampler(rel_dataset)
        rel_dataloader = DataLoader(rel_dataset, sampler=rel_sampler, batch_size=train_batch_size // 2)
        rel_model = ETypePromptModel(mlm=mlm, virtualprompt=virtual_prompt, similarity=vec_sim, predict="rel",
                                     subj_obj_pair=pair)
        if torch.cuda.device_count() > 1:
            rel_model = torch.nn.DataParallel(rel_model)
            rel_model.cuda()
        rel_model.load_state_dict(torch.load(rel_model_path[index]), strict=True)
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
            total+= len(all_labels)
            neg=all_labels.tolist().count(rel_dataset.NA_NUM)
            total_neg+=neg
            total_gold += gold
            total_guess += guess
            total_correct += correct
            print(pair,len(all_labels), neg, gold, guess, correct)
            #print(pair,total, total_neg, total_gold, total_guess, total_correct)

    print(total, total_neg, total_gold, total_guess, total_correct)
    total_recall = total_correct / total_gold
    total_prec = total_correct / total_guess
    total_f1 = 2 * total_recall * total_prec / (total_recall + total_prec)

    print('total recall=', total_recall,
          'total precision=', total_prec,
          'total f1 score=', total_f1, )


if __name__ == "__main__":
    LEARNING_RATE = {
        "subj": [1e-6, 1e-6],
        "obj": [1e-5, 1e-5],
        "rel": {
            0: [1e-5, 1e-5],  # ('organization', 'person') 7070
            1: [1e-5, 1e-5],  # ('person', 'person') 13677
            2: [3e-5, 1e-5],  # ('organization', 'organization') 8372
            3: [3e-5, 1e-5],  # ('organization', 'number') 3458
            4: [3e-5, 1e-5],  # ('organization', 'date') 4419
            5: [3e-5, 1e-5],  # ('person', 'organization') 5001
            6: [3e-5, 1e-5],  # ('person', 'nationality') 944
            7: [3e-5, 1e-5],  # ('person', 'location') 1297
            8: [3e-5, 1e-5],  # ('person', 'title') 4620
            9: [3e-5, 1e-5],  # ('person', 'date') 3588
            10: [3e-5, 1e-5],  # ('person', 'city') 1268
            11: [3e-5, 1e-5],  # ('organization', 'misc') 1046
            12: [3e-5, 1e-5],  # ('person', 'country') 1480
            13: [3e-5, 1e-5],  # ('person', 'misc') 870
            14: [3e-5, 1e-5],  # ('person', 'criminal_charge') 204
            15: [3e-5, 1e-5],  # ('organization', 'city') 1358
            16: [3e-5, 1e-5],  # ('organization', 'location') 1356
            17: [3e-5, 1e-5],  # ('person', 'religion') 151
            18: [3e-5, 1e-5],  # ('person', 'number') 2183
            19: [3e-5, 1e-5],  # ('person', 'duration') 833
            20: [3e-5, 1e-5],  # ('organization', 'religion') 211
            21: [3e-5, 1e-5],  # ('organization', 'url') 194
            22: [3e-5, 1e-5],  # ('person', 'state_or_province') 927
            23: [3e-5, 1e-5],  # ('organization', 'country') 2197
            24: [3e-5, 1e-5],  # ('organization', 'state_or_province') 672
            25: [3e-5, 1e-5],  # ('organization', 'ideology') 205
            26: [3e-5, 1e-5],  # ('person', 'cause_of_death') 496
        },
    }
    SUBJ = False
    OBJ = False
    REL = True
    EVAL = True
    model_scale = "base"
    dataset_name = "tacred"
    vec_sim = "mm"
    rel_length = 3
    random = False
    num_epoch = 1
    eval_set = "test"
    model_path = "./results/" + dataset_name + "/best-models.txt"
    # learning_rate = 1e-6 if model_scale == "base" else 1e-4
    # lr_temp = 1e-6 if model_scale == "base" else 3e-5
    if SUBJ:
        learning_rate = LEARNING_RATE["subj"][0] if model_scale == "base" else 1e-4
        lr_temp = LEARNING_RATE["subj"][1] if model_scale == "base" else 3e-5
        subj_model = train_sub_model(vec_sim=vec_sim, rel_length=rel_length, model_scale=model_scale,
                                     dataset_name=dataset_name, predict='subj',
                                     learning_rate=learning_rate, lr_temp=lr_temp, num_train_epochs=num_epoch,
                                     random=random, )
        with open(model_path, 'a', encoding='utf-8') as f:
            f.write(subj_model + "\n")
    if OBJ:
        learning_rate = LEARNING_RATE["obj"][0] if model_scale == "base" else 1e-4
        lr_temp = LEARNING_RATE["obj"][1] if model_scale == "base" else 3e-5
        obj_model = train_sub_model(vec_sim=vec_sim, rel_length=rel_length, model_scale=model_scale,
                                    dataset_name=dataset_name, predict='obj',
                                    learning_rate=learning_rate, lr_temp=lr_temp, num_train_epochs=num_epoch * 6,
                                    random=random, )
        with open(model_path, 'a', encoding='utf-8') as f:
            f.write(obj_model + "\n")
    if REL:
        rel_models = []
        for subj_obj_pair_id in range(2, 27):
            learning_rate = LEARNING_RATE["rel"][subj_obj_pair_id][0] if model_scale == "base" else 1e-4
            lr_temp = LEARNING_RATE["rel"][subj_obj_pair_id][1] if model_scale == "base" else 3e-5
            rel_model = train_sub_model(vec_sim=vec_sim, rel_length=rel_length, model_scale=model_scale,
                                        dataset_name=dataset_name, predict='rel',
                                        subj_obj_pair_id=subj_obj_pair_id, learning_rate=learning_rate, lr_temp=lr_temp,
                                        num_train_epochs=num_epoch * 15, random=random, )
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
        evaluate_sub_model(vec_sim, rel_length, model_scale, dataset_name, models[0], models[1], models[2:], eval_set)
