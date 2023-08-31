#
# test model need to ADD TIMEOUT for testing
import os
path_save="/projets/flowers/julien/OpenELM/test_llama.json"
os.environ['HF_DATASETS_CACHE'] = "/projets/flowers/julien/hf/datasets"
os.environ['TRANSFORMERS_CACHE'] = "/projets/flowers/julien/models/"
import torch

import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,TrainingArguments
from utils_test import pass_at_k,prompt_solve_puzzle,test_puzzle
# from peft import prepare_model_for_kbit_training
# from peft import LoraConfig, get_peft_model
# from trl import SFTTrainer
# from datasets import load_dataset
import numpy as np

model_id="codellama/CodeLlama-7b-Python-hf"#"TheBloke/CodeLlama-7B-Python-fp16"#"codellama/CodeLlama-7b-Python-hf"
# del model
quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,

    # quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer.padding_side='left'
tokenizer.pad_token = tokenizer.eos_token
model.eval()
# model.config.use_cache = True
model=torch.compile(model)


torch._dynamo.config.suppress_errors = True

path_trainset="/projets/flowers/julien/OpenELM/src/openelm/utils/preprocess_p3_emb.json"
with open(path_trainset, 'r') as f:
    trainset = json.load(f)
    
list_trainset= [x["program_str"] for x in trainset]
list_puzzle_correct=[]
correct_puzz=0
curr_idx=0
num_return_sequences=10 #n_try
list_passk=[]
list_puzzle=[]
with torch.no_grad():
    
    for idx in tqdm(range(curr_idx,len(list_trainset))): #len(dataset["test"])
        curr_idx=idx
        # idx=0
        print(f"\n\n============ idx {idx} ==================\n")
        flag=True
        attempt=0
        list_puzzle_idx=[]
        while flag and attempt<5:
            attempt+=1
            try:
                puzzle= list_trainset[idx]
                prompt_f = puzzle.split("def g(")[0]
                prompt = prompt_solve_puzzle.format(pb=prompt_f)
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), attention_mask=inputs["attention_mask"], max_new_tokens=256,do_sample=True, temperature=0.8,num_return_sequences=num_return_sequences)
                len_prompt=inputs["input_ids"][0].shape[0]
                flag=False
                
            except Exception as e:
                print("pb when generating puzzles")
                print(str(e))
                print()
                pass_k=0.
                continue
            cor_puz=0
            for i in range(len(outputs)):
                
                # out ="def g("
                out = tokenizer.decode(outputs[i][len_prompt:], skip_special_tokens=True)
                # print("output gene[rated:",out)
                
                extract_g=out.split("```")[0].split("assert")[0]
                extract_g = extract_g+"\nassert f(g()) == True\n"
                test_fg= prompt_f+extract_g 
                list_puzzle.append(test_fg)
                if i<1:
                    print("\n-------------------\n")
                    print(test_fg)
                if test_puzzle(test_fg):
                    if cor_puz==0:
                        print("\n-------------------\n")
                        print(test_fg)
                    cor_puz+=1

                    list_puzzle_idx.append(test_fg)
                n_sample, n_correct=10,cor_puz
                pass_k = pass_at_k(n_sample, n_correct, k=10)



        list_puzzle_correct.append(list_puzzle_idx)   

        if cor_puz>=1:
            print("\n=================\n")
            print("correct puzzle",correct_puzz)
            correct_puzz+=1
        list_passk.append(pass_k)

        print(f"correct puzzles: {correct_puzz}/{idx+1}")
        with open("sample.json", "w") as outfile:
            json.dump(list_passk,outfile)