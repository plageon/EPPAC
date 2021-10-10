import json

def truncate_feature():
    segments = ["train", "val"]
    full_data = {}
    for segment in segments:
        full_data[segment] = []
        with open("wiki80_" + segment + ".txt", "r", encoding="utf-8") as f:
            for sample in f.readlines():
                data = json.loads(sample)
                truncate_data = {}
                truncate_data["token"] = data["token"]
                truncate_data["subj"] = data["h"]["name"].split(" ")
                truncate_data["obj"] = data["t"]["name"].split(" ")
                truncate_data["relation"] = data["relation"]
                truncate_data["subj_type"] = ""
                truncate_data["obj_type"] = ""
                full_data[segment].append(truncate_data)
    full_data["test"] = full_data["val"]
    for segment, data in full_data.items():
        with open(segment + "-truncate.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

truncate_feature()
