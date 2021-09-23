import torch
from transformers import AdamW, get_linear_schedule_with_warmup


def get_optimizer(model, train_dataloader, gradient_accumulation_steps, num_train_epochs, learning_rate, adam_epsilon,
                  warmup_steps, weight_decay):
    t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    cur_model = model.module if hasattr(model, 'module') else model

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in cur_model.mlm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in cur_model.mlm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=t_total)

    return optimizer, scheduler


def get_optimizer4temp(model, lr_temp):
    cur_model = model.module if hasattr(model, 'module') else model
    optimizer = torch.optim.SGD([{'params': cur_model.label2embed}], lr=lr_temp, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    return optimizer, scheduler

def get_optimizer4dense(model,lr_temp):
    cur_model = model.module if hasattr(model, 'module') else model
    optimizer=torch.optim.SGD([{'params':cur_model.linear.parameters()}],lr=lr_temp,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    return optimizer, scheduler