import numpy as np
import json
path = "/home/laetitia/work/OpenELM/tests/subset2label_lae.json"
with open(path, 'r') as f:
    data2label = json.load(f)

for idx in range(0,len(data2label)):
    if not ("GT_emb_lae" in data2label[idx].keys()):
        print(" \n ============")
        print("====== idx", idx)
        print(data2label[idx]["program_str"])
        emb_chat = [i for i, en in enumerate(data2label[idx]["emb"]) if en == 1]
        
        print("Take a deep breath and label this problem step-by-step\n")
        inp = input()
        true_label = inp.split(",")
        true_label_parsed = [int(idx) for idx in true_label]
        #convert to multi-hot
        true_label = np.zeros(10,dtype=int)
        true_label[true_label_parsed] = 1
        data2label[idx]["GT_emb_lae"] = true_label.tolist()
        with open(path, 'w') as f:
            json.dump(data2label, f, indent=4)
        print("label chatgp", emb_chat)