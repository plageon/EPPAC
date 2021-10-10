import torch.nn as nn
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from code.optimizing import get_optimizer, get_optimizer4temp
from code.build_model import ETypePromptModel
from code.collect_data import EntityPromptDataset
from code.utils import get_tokenizer_mlm, set_seed, get_single_temps, get_raw_temps, RelDict
from code.virtual_prompt import VirutalPrompt

if __name__ == '__main__':
    model_type = 'roberta'
    model_name = 'roberta-base'
    tokenizer, mlm = get_tokenizer_mlm(model_type, model_name)

    dataset = 'tacred'
    data_dir = "../datasets/" + dataset
    output_dir = "../results/" + dataset
    per_gpu_train_batch_size = 8
    gradient_accumulation_steps = 1
    max_seq_length = 256
    warmup_steps = 500
    learning_rate = 3e-5
    learning_rate_for_new_token = 1e-5
    num_train_epochs = 5
    weight_decay = 1e-2
    adam_epsilon = 1e-6
    lr_temp = 1e-2
    max_grad_norm = 1.0
    vec_sim = 'mm'
    predict = 'rel'
    subj_obj_pair = ("person", "location")
    rel_length = 3
    set_seed(123)

    raw_temp = get_raw_temps(data_dir)
    virtual_prompt = VirutalPrompt(raw_temp=raw_temp, tokenizer=tokenizer, mlm=mlm, rel_length=rel_length,
                                   datadir=data_dir)
    # """
    dataset = EntityPromptDataset(
        predict=predict,
        subj_obj_pair=subj_obj_pair,
        path=data_dir,
        name='train',
        max_seq_length=max_seq_length,
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
        virtual_prompt=virtual_prompt,
        tokenizer=tokenizer)
    dataset.save(path=output_dir, name="test")
    # """
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
        name="test",
        virtual_prompt=virtual_prompt,
        tokenizer=tokenizer, )

    train_batch_size = 2
    # train_dataset.cuda()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    # val_dataset.cuda()
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=train_batch_size // 2)

    # test_sampler = SequentialSampler(test_dataset)
    # test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=train_batch_size // 2)

    # label2temp = torch.nn.Parameter(torch.stack([virtual_prompt.class2temp[rel_dict.id2rel(i)] for i in range(len(virtual_prompt.class2temp))]))
    model = ETypePromptModel(mlm=mlm, virtualprompt=virtual_prompt, similarity=vec_sim, predict=predict,
                             subj_obj_pair=subj_obj_pair)
    criterion = nn.CrossEntropyLoss()

    iterator = iter(train_dataloader)
    x = next(iterator)
    # for key, tensor in x.items():
    #    x[key] = tensor.cuda()
    # print(x['mlm_labels'])
    logits = model(**x)
    print(x)
    print(logits)
    optimizer, scheduler = get_optimizer(model, train_dataloader, gradient_accumulation_steps, num_train_epochs,
                                         learning_rate, adam_epsilon, warmup_steps, weight_decay)  # 将lm做fine tune
    optimizer4temp, scheduler4temp = get_optimizer4temp(model, lr_temp)
    loss = criterion(logits, x['labels'])
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer4temp.step()
    scheduler4temp.step()
