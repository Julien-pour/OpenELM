from concurrent.futures import ThreadPoolExecutor
def get_completion(client, prompt : str, cfg_generation,temperature=None)->str:
    """Get completion from OpenAI API"""
    kwargs={}
    kwargs.update(cfg_generation)
    if temperature is not None:
        kwargs["temperature"]= temperature
    try :
        completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI programming assistant"},#You are a coding assistant, skilled in writting code with creative flair."},
            {"role": "user", "content": prompt}
        ],**kwargs
        )
    except Exception as e:
        print("completion problem: ",e)
        return None 
    # completion_token = completion.usage.completion_tokens
    # prompt_token = completion.usage.prompt_tokens
            
    out = completion.choices[0].message.content
    return out




def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_multiple_completions(client, batch_prompt: list[str], cfg_generation: dict, batch_tools: list[list[dict]]=None,max_workers=20,temperature=None)->list[str]:
    """Get multiple completions from OpenAI API
    batch_tools =[[tools]] tools is the function, toll_name is the name of the tool

                    /!\ need to integrate batch tools in the loop /!\
    """
    # check that batch_prompt is list[str]
    if isinstance(batch_prompt, str):
        batch_prompt = [batch_prompt]
    completions = []
    if max_workers>1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for sub_batch in chunks(batch_prompt, max_workers):
                for idx,message_list in enumerate(sub_batch):
                    # kwargs_modified = args.copy()
                    # kwargs_modified["messages"] = message_list
                    kwargs = {"client":client, "prompt":message_list}
                    kwargs["cfg_generation"]=cfg_generation
                    if temperature is not None:
                        kwargs["temperature"]= temperature
                    # if "kwargs" in kwargs_modified:
                    #     original_kwargs = kwargs_modified.pop("kwargs")
                    future = executor.submit(
                        get_completion,**kwargs
                    )
                    completions.append(future)
        # Retrieve the results from the futures
        results = [future.result() for future in completions]
    else:
        for idx,message_list in enumerate(batch_prompt):
            # kwargs_modified = args.copy()
            # kwargs_modified["messages"] = message_list
            kwargs = {"client":client, "prompt":message_list}
            kwargs["cfg_generation"]=cfg_generation
            if temperature is not None:
                kwargs["temperature"]= temperature
            # if "kwargs" in kwargs_modified:
            #     original_kwargs = kwargs_modified.pop("kwargs")
            result = get_completion(**kwargs)
            completions.append(result)
            results = completions

    return results
