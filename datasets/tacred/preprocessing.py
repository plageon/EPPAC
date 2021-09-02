import json
import os

segments = ["dev", "train", "test"]


def truncate_feature():
    for segment in segments:
        json_file = "./" + segment + ".json"
        target_file = "./" + segment + "-truncate.json"
        with open(json_file, "r", encoding="utf-8") as f:
            samples = json.loads(f.read())
        features = []
        for sample in samples:
            feature = {}
            token = sample["token"]
            feature["token"] = token
            feature["subj"] = token[sample["subj_start"]:sample["subj_end"] + 1]
            feature["obj"] = token[sample["obj_start"]:sample["obj_end"] + 1]
            feature["subj_type"] = sample["subj_type"].lower()
            feature["obj_type"] = sample["obj_type"].lower()
            feature["relation"] = sample["relation"]
            features.append(feature)
        with open(target_file, "w", encoding="utf-8") as target_f:
            json.dump(features, target_f)
        print(segment, "completed")


def figure_classes():
    with open("train-truncate.json", "r", encoding="utf-8") as f:
        samples = json.loads(f.read())
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
        print(pair[1])
    with open("classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f)
    print("classes figured")


#truncate_feature()
figure_classes()
