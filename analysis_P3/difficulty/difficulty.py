import sys
sys.path.append('/home/flowers/work/OpenELM')
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import json
import pickle
from utils_test import pass_at_k,judge_parallel,return_full_prompt,Prompt_Intstruction
import copy
import numpy as np
from tqdm import tqdm
import os
os.environ['TOKENIZERS_PARALLELISM'] = "True"

parser = argparse.ArgumentParser(description="argument parsing")
parser.add_argument("-p", "--path_maps", type=str, help="path to maps",default="/home/flowers/work/OpenELM/analysis_P3/quality/to_analyse/maps_1_rd_gen.json")
parser.add_argument("-l", "--path_hf_model_repo", type=str, help="path to where hf model are stored",default="")
parser.add_argument("-k", "--arg_k", type=int, help="k in pass@k",default=5)
parser.add_argument("-b", "--arg_bs_test", type=int, help=" bs test",default=16)
parser.add_argument("-m", "--arg_model_idx", type=int, help="model idx",default=0)
parser.add_argument("-f", "--arg_flash", type=str, help="activate flash",default="flash2")
parser.add_argument("-c", "--arg_compile", help="use torch compile ",default=True,type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("-i", "--arg_backend_inference", type=str, help="inference backend [hf,openai,mistral, exllama, vllm]  ",default="vllm")
parser.add_argument("-g", "--arg_gpus", type=int, help="number of  gpu  ",default=1)


args = parser.parse_args()
path_model_base = args.path_hf_model_repo
num_return_sequences = args.arg_k #n_try
snapshot_path = args.path_maps
bs = args.arg_bs_test

mode = args.arg_backend_inference #["hf","openai","mistral", "exllama", "vllm"] hf -> hf model

list_model=[
"deepseek-ai/deepseek-coder-1.3b-instruct",
# "WizardCoder-33B-V1.1-6.0bpw-h6-exl2",
# "CodeLlama-70b-Instruct-hf-6.0bpw-h6-exl2"
]

# Init model

match mode:
    case "hf"|"vllm":
# if mode =="hf":
        model_idx = args.arg_model_idx

        #todo 9,11,14,15,16
        model_id = list_model[model_idx]
        path_model = path_model_base+model_id
        args_model = {}

        tokenizer = AutoTokenizer.from_pretrained(path_model,local_files_only=True)#LlamaTokenizer.from_pretrained(model_id,local_files_only=True)
        tokenizer.padding_side='left'
        tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16
        args_model["torch_dtype"]=dtype
            # remove flash attention for v100 not compatible
        
        if mode == "vllm":
            from vllm import LLM,SamplingParams
            llm = LLM(path_model,max_model_len=1024)

            sampling_params = SamplingParams(
                        temperature=0.7,
                        top_p=1,
                        max_tokens=512,
                        presence_penalty=1.15,
                    )
        else:
            if args.arg_flash == "flash2":
                print("use flash attention 2")
                args_model["attn_implementation"]="flash_attention_2"
            
            model = AutoModelForCausalLM.from_pretrained(
                path_model,

                device_map="auto",
                local_files_only=True,
                **args_model
            )

            model.eval()
            model.config.use_cache = True
            compile=args.arg_compile
            print(f"compile {compile}")
            assert isinstance(compile, bool)
            if compile:
                print("compiling model")
                model = torch.compile(model)

    case "exllama":
        
        from exllamav2 import(
            ExLlamaV2,
            ExLlamaV2Config,
            ExLlamaV2Cache,
            ExLlamaV2Tokenizer,
        )
        from exllamav2.generator import (
            ExLlamaV2BaseGenerator,
            ExLlamaV2Sampler
        )
        # path model
        model_idx = args.arg_model_idx
        model_id = list_model[model_idx]
        path_model=path_model_base+model_id
        tokenizer = AutoTokenizer.from_pretrained(path_model,local_files_only=True)
        # load model+config
        config = ExLlamaV2Config()
        config.model_dir = path_model
        config.prepare()
        batch_size = args.arg_bs_test 
        config.max_batch_size = batch_size

        model = ExLlamaV2(config)
        print("Loading model: " + path_model)
        cache = ExLlamaV2Cache(model, lazy = True, batch_size = batch_size)  # Cache needs to accommodate the batch size
        model.load_autosplit(cache)

        tokenizer_exllama = ExLlamaV2Tokenizer(config)

        # Initialize generator

        generator = ExLlamaV2BaseGenerator(model, cache, tokenizer_exllama)

        # Sampling settings

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 0.7
        settings.top_k = 50
        settings.top_p = 1.
        settings.token_repetition_penalty = 1.05

        max_new_tokens = 512     
        # generator.warmup()  # Only needed to fully initialize CUDA, for correct benchmarking
    case "openai":
        from key import key

        model_id="gpt-3.5-turbo-0125"#"gpt-3.5-turbo-1106"
        from openai import OpenAI
        from utils_test import get_multiple_completions
        client = OpenAI(max_retries=100,timeout=100,api_key=key)
        cfg_generation: dict = {
                "temperature": 0.7,
                "model": model_id,
            }
    case "mistral":
        from mistralai.client import MistralClient
        from key import mistral_key
        from utils_test import get_multiple_completions
        model_id = "mistral-medium"
        api_key=mistral_key
        client = MistralClient(api_key=api_key, max_retries = 5)
        cfg_generation: dict = {
                "temperature": 0.7,
                "model": model_id,
            }
    case _:
        raise ValueError("mode not supported")    # from key import key
 

print(f" ==================  model_id {model_id} ==================")

curr_idx=0
num_return_sequences = args.arg_k #n_try
list_all_passk=[[] for i in range(num_return_sequences)]
list_passk=[]

list_puzzle=[]
list_all_puzzle=[]

# load archive
with open(snapshot_path, "r") as f:
    puzzles = json.load(f)

for idx in range(len(puzzles)):
    if f"pass_{num_return_sequences}" in puzzles[idx]:
        list_passk.append(puzzles[idx][f"pass_{num_return_sequences}"])
        curr_idx = idx

# curr_idx=0


# function to generate response
def generate_response(list_prompt,model_id=model_id):
    match mode:
        case "hf" | "vllm" | "exllama":
            args_generate={}
            # if model_id in ["deepseek-coder-1.3b-instruct","deepseek-coder-6.7b-instruct","deepseek-coder-33B-instruct-GPTQ"] :
            #     args_generate["eos_token_id"]=32021
            # else:
            #     args_generate["pad_token_id"]=tokenizer.eos_token_id
            with torch.no_grad():
                if mode=="hf":
                    inputs = tokenizer(list_prompt, return_tensors="pt",padding=True).to("cuda")
                    # for idx_inp in range(len(inputs)):
                    len_prompt = inputs["input_ids"].shape[1]
                list_puzzle_gen=[[] for _ in range(len(list_prompt))]
                start_time = time.time()  
                num_tokens = 0
                for i_seq in range(num_return_sequences):
                    match mode:
                        case "vllm":
                            result = llm.generate(list_prompt, sampling_params,use_tqdm=False)
                            generated_texts = []
                            for output in result:
                                num_tokens += len(output.outputs[0].token_ids)
                                generated_texts.append(output.outputs[0].text)

                        case "exllama":
                            outputs = generator.generate_simple(list_prompt, settings, max_new_tokens, seed = i_seq)
                            trimmed_outputs = [o[len(p):] for p, o in zip(list_prompt, outputs)]
                            generated_texts = trimmed_outputs
                        case "hf":
                            args_generate["pad_token_id"] = tokenizer.eos_token_id
                            outputs = model.generate(**inputs,max_new_tokens=512,do_sample=True, temperature=0.7,**args_generate)
                            generated_texts = tokenizer.batch_decode(outputs[:,len_prompt:], skip_special_tokens=True)
                    for idx_out_gen in range(len(generated_texts)): #len output -> bs
                        list_puzzle_gen[idx_out_gen].append(generated_texts[idx_out_gen])
                end_time = time.time()  
                total_time = end_time - start_time
                if mode =="vllm":
                    total_tokens = num_tokens
                elif mode=="exllama":
                    total_tokens = max_new_tokens * len(outputs)
                else:
                    total_tokens = outputs.shape[1] * len(outputs)
                tokens_per_second = total_tokens / total_time
                dic_speed={"total_time":total_time,"total_tokens":total_tokens,"tokens_per_second":tokens_per_second}
                # try:
                #     with open(name_json_speed, "r") as outfile:
                #         json_content=json.load(outfile)
                #     json_content.append(dic_speed)
                #     with open(name_json_speed, "w") as outfile:
                #         json.dump(json_content,outfile,indent=4)
                # except:
                #     print("error write speed in  json")
                return list_puzzle_gen
        case "openai": #openai api
            list_puzzle_gen=[[] for _ in range(len(list_prompt))]
            batch_prompt = list_prompt
            for id_num in range(num_return_sequences):
                print(f"num_return_sequences {id_num} / {num_return_sequences}")
                generated_texts=get_multiple_completions(client, batch_prompt, cfg_generation,max_workers=50)
                for idx_out_gen in range(len(generated_texts)):
                    list_puzzle_gen[idx_out_gen].append(generated_texts[idx_out_gen])
            return list_puzzle_gen
        case "mistral": #openai api # could merge case mistral and openai
            list_puzzle_gen=[[] for _ in range(len(list_prompt))]
            batch_prompt = list_prompt
            for id_num in range(num_return_sequences):
                print(f"num_return_sequences {id_num} / {num_return_sequences}")
                generated_texts=get_multiple_completions(client, batch_prompt, cfg_generation,max_workers=5,mode="mistral")
                for idx_out_gen in range(len(generated_texts)):
                    list_puzzle_gen[idx_out_gen].append(generated_texts[idx_out_gen])
            return list_puzzle_gen
        case _:
            raise ValueError("mode not supported")



for idx in tqdm(range(curr_idx,len(puzzles),bs)): #  #len(dataset["test"])
    # curr_idx=idx
    # idx=0
    print(f"\n\n============ idx {idx}/{len(puzzles)} ==================\n")
    attempt=0
    list_puzzle_idx=[]
    list_prompt=[]
    list_prompt_f=[]
    subset_test = puzzles[idx:idx+bs]
    subset_test = [sub_puz["program_str"] for sub_puz in subset_test]
    # subset_emb_test= list_emb_test[idx:idx+bs]
    for idx_puz in range(len(subset_test)):
        
        prompt_f = subset_test[idx_puz].split("def g(")[0]
        list_prompt_f.append(prompt_f)
        # Retrieval augmented generation 
        # Finding the k-nearest neighbors for the first program_str
        # distances, indices = KNN.kneighbors([subset_emb_test[idx_puz]])

        # Retrieving the program strings for the k-nearest neighbors
        # nearest_neighbors_puzzles = [list_puzzle_archive[index] for index in indices[0][:]]
        #list_problem_str= KNN
        if model_id in ["FrankenBeagle14-11B","Nous-Hermes-2-SOLAR-10.7B"]:
            message_chat = Prompt_Intstruction.format(pb=prompt_f)
            message_chat = [
            {"role": "user", "content": message_chat}]

            prompt = tokenizer.apply_chat_template(message_chat, tokenize=False, add_generation_prompt=True)
        else:
            prompt = return_full_prompt(model_id=model_id,pb=prompt_f) # todo
        list_prompt.append(prompt)

    # generate response
    list_puzzle_gen = generate_response(list_prompt)
    list_generated_text = copy.deepcopy(list_puzzle_gen)

    for i in range(len(list_puzzle_gen)): # along the bs
        dic_save={}
        list_raw_puzzle = []
        list_proc_puzzle =[]
        for j in range(len(list_puzzle_gen[i])):
            prompt_f =list_prompt_f[i]
            try:
                #check if "```" is in list_puzzle_gen[i][j]
                list_puzzle_gen[i][j] = list_puzzle_gen[i][j].replace("```python","```")
                list_puzzle_gen[i][j] = list_puzzle_gen[i][j].replace("```Python","```")

                if "```" in list_puzzle_gen[i][j]:
                    extract_g=list_puzzle_gen[i][j].split("```")[1].split("assert")[0]
                else:
                    if "assert" in list_puzzle_gen[i][j]:
                        extract_g=list_puzzle_gen[i][j].split("assert")[0]
                    else:    
                        extract_g=list_puzzle_gen[i][j]
            except:
                print("error extract g")
                print(list_puzzle_gen[i][j])
            extract_g = extract_g+"\nassert f(g()) == True\n"
            test_fg= prompt_f+extract_g 
            list_puzzle_gen[i][j] = test_fg
            list_puzzle.append(test_fg)
            list_proc_puzzle.append(test_fg)
            list_raw_puzzle.append(prompt_f+list_puzzle_gen[i][j])
        dic_save["raw_puzzle"]=list_raw_puzzle
        dic_save["process_puzzle"]=list_proc_puzzle
        
            # if j<1:
            #     print("\n-------------------\n")
            #     print(test_fg)
            
        
        list_valid_puzzles = judge_parallel(list_puzzle_gen[i])
        dic_save["list_valid"]=list_valid_puzzles                 
        list_all_puzzle.append(dic_save)    

        cor_puz= np.sum(list_valid_puzzles)

        n_sample, n_correct=num_return_sequences,cor_puz
        pass_k = pass_at_k(n_sample, n_correct, k=num_return_sequences)
        list_passk.append(pass_k)

        #compute passk for k=[1,...,num_return_sequences]
        for idx_passk in range(num_return_sequences):
            pass2add=pass_at_k(n_sample, n_correct, k=idx_passk+1)
            list_all_passk[idx_passk].append(pass2add)

        puzzles[idx + i][f'pass_{num_return_sequences}'] = pass2add
        puzzles[idx + i]['n_sample'] = int(n_sample)
        puzzles[idx + i]['n_correct'] = int(n_correct)

        proba_solved = n_correct / n_sample
        # testset[idx + i]['proba_solved'] = float(proba_solved)
        # testset[idx + i]['n_sample'] = int(n_sample)
        # testset[idx + i]['n_correct'] = int(n_correct)
        # testset[idx + i]['generated_text'] = list_generated_text[i]
        # testset[idx + i]['parsed_puzzles'] = list_puzzle_gen[i]
        # testset[idx + i]['prompt'] = list_prompt[i]
        
    print(f"correct puzzles: {int(np.sum(list_passk))}/{len(list_passk)}")
    with open(snapshot_path, "w") as f:
        json.dump(puzzles, f, indent=4)