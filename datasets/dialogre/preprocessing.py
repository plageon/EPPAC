import json
from transformers import AutoTokenizer
segments=["train","dev","test"]
tokenizer=AutoTokenizer.from_pretrained("roberta-base")
def truncate_feature():
    full_data={}
    for segment in segments:
        full_data[segment]=[]
        with open(segment+".json","r",encoding="utf-8") as f:
            data=json.loads(f.read())
            for item in data:
                dialog=item[0]
                token=[]
                for sentence in dialog:
                    token+=tokenizer.tokenize(sentence)
                relation_details=item[1]
                for detail in relation_details:
                    sample={}
                    sample["token"]=token
                    sample["subj"]=detail["x"]
                    sample["obj"]=detail["y"]
                    sample["subj_type"]=detail["x_type"]
                    sample["obj_type"]=detail["y_type"]
                    sample["relation"]=detail["r"][0]
                    full_data[segment].append(sample)
        with open(segment+"-truncate.json","w",encoding="utf-8") as f:
            json.dump(full_data[segment],f)

def configure_classes():
    with open("train-truncate.json","r",encoding="utf-8") as f:
        samples=json.loads(f.read())
    subj_classes = {}
    obj_classes = {}
    relation_classes = {}
    subj_obj_pairs = {}
    subj_obj2realtion = {}
    for sample in samples:
        if sample["subj_type"] not in subj_classes:
            subj_classes[sample["subj_type"]] = len(subj_classes)
        if sample["obj_type"] not in obj_classes:
            obj_classes[sample["obj_type"]] = len(obj_classes)
        pair = (sample["subj_type"], sample["obj_type"])
        if pair not in subj_obj_pairs:
            subj_obj_pairs[pair] = len(subj_obj_pairs)
            subj_obj2realtion[pair] = {}
        if sample["relation"] not in relation_classes:
            relation_classes[sample["relation"]] = len(relation_classes)
        if sample["relation"] not in subj_obj2realtion[pair]:
            subj_obj2realtion[pair][sample["relation"]] = len(subj_obj2realtion[pair])

    subj_obj2realtion = {subj_obj_pairs[k]: v for k, v in subj_obj2realtion.items()}
    subj_obj_pairs = {v: k for k, v in subj_obj_pairs.items()}

    classes = {
        "subj_types": subj_classes,
        "obj_types": obj_classes,
        "relation_classes": relation_classes,
        "subj_obj_pairs": subj_obj_pairs,
        "subj_obj2relation": subj_obj2realtion,
    }
    for pair in classes['subj_obj_pairs'].items():
        print(pair)
    with open("classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f)
    print("classes figured")

truncate_feature()
configure_classes()