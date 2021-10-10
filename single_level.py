import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm, trange
import os

from code.build_model import ETypePromptModel
from code.collect_data import EntityPromptDataset
from code.optimizing import get_optimizer, get_optimizer4temp, get_optimizer4dense
from code.utils import get_tokenizer_mlm, set_seed, get_single_temps, get_raw_temps, RelDict, evaluate
from code.virtual_prompt import VirutalPrompt
import datetime

if __name__ == '__main__':
    RELOAD_DATA = True
    LOAD_KPL = False
    TRAIN = True
    EVAL = True

    model_type = 'roberta'
    model_name = 'roberta-base'
    tokenizer, mlm = get_tokenizer_mlm(model_type, model_name)
    n_gpu = torch.cuda.device_count()

    dataset_name = "dialogre"
    data_dir = "./datasets/" + dataset_name
    output_dir = "./results/" + dataset_name
    per_gpu_train_batch_size = 4
    gradient_accumulation_steps = 1
    max_seq_length = 512
    warmup_steps = 500
    learning_rate = 2e-5
    learning_rate_for_new_token = 1e-5
    num_train_epochs = 5
    weight_decay = 1e-2
    adam_epsilon = 1e-6
    lr_temp = 1e-5
    max_grad_norm = 1.0
    vec_sim = "mm"
    rel_length = 5
    set_seed(123)

    predict = "rel"
    subj_obj_pair = "None"
    random_centers = True
    fixed_centers = False
    single_level = True
    raw_temp = get_single_temps(data_dir) if os.path.exists(data_dir + "/temp.txt") else None
    virtual_prompt = VirutalPrompt(raw_temp=raw_temp, tokenizer=tokenizer, mlm=mlm, rel_length=rel_length,
                                   datadir=data_dir, random_centers=random_centers, single_level=single_level)

    # rel_dict = RelDict(path=data_dir + "/" + "rel2id.json")
    # temps = get_temps(tokenizer, data_dir)
    # raw_temp = get_raw_temps(data_dir)
    # '''
    if RELOAD_DATA:
        dataset = EntityPromptDataset(
            train=True,
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
            train=True,
            predict=predict,
            subj_obj_pair=subj_obj_pair,
            path=data_dir,
            name="test" if dataset_name == "semeval" else 'dev',
            max_seq_length=max_seq_length,
            rel_length=rel_length,
            virtual_prompt=virtual_prompt,
            tokenizer=tokenizer)
        dataset.save(path=output_dir, name="dev")

        # If the dataset has been saved,
        # the code ''dataset = REPromptDataset(...)'' is not necessary.

        dataset = EntityPromptDataset(
            train=True,
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
        train=True,
        predict=predict,
        subj_obj_pair=subj_obj_pair,
        path=output_dir,
        name="train",
        virtual_prompt=virtual_prompt,
        tokenizer=tokenizer, )

    val_dataset = EntityPromptDataset.load(
        train=True,
        predict=predict,
        subj_obj_pair=subj_obj_pair,
        path=output_dir,
        name="test" if dataset_name == "semeval" else 'dev',
        virtual_prompt=virtual_prompt,
        tokenizer=tokenizer, )

    test_dataset = EntityPromptDataset.load(
        train=True,
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

    if LOAD_KPL:
        model.load_state_dict(torch.load(best_model_path))
    if TRAIN:
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
                    if not fixed_centers:
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

    # model.load_state_dict(torch.load(output_dir+"/"+'parameter'+'4'+".pkl"))
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
          ' relation length=', rel_length)

    if EVAL:
        # test_acc, test_prec, test_recall, test_mi_f1, test_ma_f1 = evaluate(model, test_dataset, test_dataloader)
        print("best acc", his['acc'][best_epoch], "best prec", his['prec'][best_epoch], "best recall",
              his['recall'][best_epoch], "best mi f1 ", his['mi_f1'][best_epoch],
              "\nbest ma f1 ", his['ma_f1'][best_epoch])
        print("completed at", datetime.datetime.now())
        print("======================")
