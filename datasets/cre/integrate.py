import json


def integrate():
    with open('./train.json', 'r', encoding='utf-8') as f:
        ori_train = json.loads(f.read())
    with open('./challenge_set.json', 'r', encoding='utf-8') as f:
        cre = json.loads(f.read())
    train = cre + ori_train
    with open('./train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f)


integrate()
