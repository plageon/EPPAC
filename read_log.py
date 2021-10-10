original_log = "cdist1-random555len1009.log"
with open(original_log, 'r', encoding='utf-8') as f:
    lines = f.readlines()
useful = []
for line in lines:
    if "%" not in line and len(line) > 1:
        useful.append(line)
del lines
with open(original_log + ".truncate", 'w', encoding='utf-8') as f:
    f.writelines(useful)
