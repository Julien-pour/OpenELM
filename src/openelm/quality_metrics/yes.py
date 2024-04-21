# YES quality metrics
import torch
soft = torch.nn.Softmax(dim=-1)

def return_yes_prompt(yes_mode="skills_improvement"):
    match yes_mode:
        case "skills_improvement":
            return yes_skills_improvement
        case "yes_gpt4":
            return yes_gpt4
        case "yes_gpt4_2":
            return yes_gpt4_2
        case "yes_mixtral":
            return yes_mixtral
        case "yes_opus":
            return yes_opus
        case "yes_opus_2":
            return yes_opus_2
        case "pretraining":
            return yes_pretraining
        case "finetuning":
            return yes_finetuning
        case "education":
            return yes_education
        case _:
            return yes_skills_improvement


def return_proba_yes(values,list_words):
    """
    return the probability of the token "Yes" if it's in the list of words,
    or return the one minus the probability of the token "No" if there si only No in the list of words
    list_words: 1d list of words
    values: 1d tensor of probabilities tensors
    """
    flag_no = False
    if "Yes" in list_words:
            idx = list_words.index("Yes")
            proba_yes = values[idx] 
    elif "No" in list_words:
        idx = list_words.index("No")
        flag_no = True
        proba_No = values[idx] 

    else:
        print("No yes or no token found")
        return -1
    proba_yes=values[idx] 
    if flag_no: # if the token "no" is selected, we need to invert the probability
        proba_yes = 1-proba_No

    proba_yes=values[idx] 
    return proba_yes

prompt_deepseek_coder="""You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science.
### Instruction:
{instruction}
### Response:
"""
prompt_openchat= """GPT4 Correct User: {instruction}<|end_of_turn|>GPT4 Correct Assistant:"""

def return_prompt_format(model_id, text):
    """
    return the prompt with the correct format for the given model_id

    list model_compatible:['deepseek-coder', 'openchat']
    """
    
    if "deepseek-coder" in model_id:
        prompt_model=prompt_deepseek_coder
    elif "openchat" in model_id:
        prompt_model = prompt_openchat
    else: 
        raise ValueError(f"Model {model_id} not supported")
    
    return prompt_model.format(instruction=text)





#prompt yes
 
yes_skills_improvement="""###
Python Programming Puzzles (P3) consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in solving a challenging SAT problem `f` with a function `g` such that `f(g())` evaluates to `True`.
Here is a Python Programming Puzzle:
{datapoint}
assert f(g()) == True
###
Is the previous paragraph demarcated within ### and ###
contain informative signals for you to improve your skills in solving Python Programming Puzzles?
OPTIONS:
- Yes
- No
"""
yes_gpt4="""Consider the following Python Programming Puzzle (P3). This puzzle consists of a problem function `f` and its corresponding solution `g`. The function `g` is designed to solve the SAT problem posed by `f`. The challenge lies in solving a challenging the SAT problem `f` with a function `g` such that `f(g())` evaluates to `True`.
Given the nature of this puzzle, do you find this specific Python Programming Puzzle (P3) helpful for learning new skills or improving your ability to solve other programming puzzles of similar complexity? Answer with 'Yes' or 'No'.

# P3:
```python
{datapoint}
```
"""
yes_gpt4_2="""Please evaluate the following Python Programming Puzzle:
```python
{datapoint}
```
The puzzle consists of two functions: a problem function f and its corresponding solution g. The primary goal is to solve a challenging SAT problem f using a function g such that the expression f(g()) evaluates to True.

Details of the Python Programming Puzzle (P3):

Problem Function (f): [Provide the specific details or description of the problem function here]
Solution Function (g): [Provide the specific details or description of the solution function here]
Based on the complexity and the skills required to solve this puzzle, do you think this P3 is helpful for the LLM to learn new skills or help it improve solving other similar P3 problems? Please answer 'Yes' or 'No'.
"""
yes_mixtral="""Below is a Python Programming Puzzle (P3) that involves solving a challenging SAT problem using a function. Determine whether this P3 is helpful for a large language model (LLM) with 2 billion parameters to learn new skills or improve its ability to solve other P3 problems.

P3 Format:

Function `f` is a SAT problem that takes a boolean expression as input and returns `True` if the expression is satisfiable, and `False` otherwise.

Function `g` is a function that takes no input and returns a boolean expression as output.

The challenge is to solve the SAT problem `f` using the function `g` such that `f(g())` evaluates to `True`.

P3:
```python
{datapoint}
```

Based on the provided P3, answer the following question with only "Yes" or "No":
"""

yes_opus="""Analyze the following Python Programming Puzzle (P3):

{datapoint}

Based on the complexity, novelty, and potential for skill acquisition or improvement, is this P3 helpful for an LLM to learn new skills or enhance its ability to solve other P3 problems? Answer with a simple "Yes" or "No".
"""
yes_opus_2="""Given the following Python Programming Puzzle (P3):

{datapoint}

A P3 consists of a problem function `f` and a corresponding solution function `g`, where the challenge is to solve a difficult SAT problem `f` with a function `g` such that `f(g())` evaluates to `True`.

Based on the provided P3, answer the following question with only "Yes" or "No":

Will this P3 help the AI learn new skills or improve its ability to solve other Python Programming Puzzles?
"""
yes_pretraining="""###
Python Programming Puzzles (P3) consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in solving a challenging SAT problem `f` with a function `g` such that `f(g())` evaluates to `True`.
Here is a Python Programming Puzzle:
{datapoint}
assert f(g()) == True
###
Does the previous paragraph demarcated within ### and ###
contain informative signal for pre-training a large-language model?
OPTIONS:
- yes
- no"""


yes_finetuning="""###
{datapoint}
###
Does the previous paragraph demarcated within ### and ###
contain informative signal for fine-tuning a large-language model?
An informative datapoint should be well-formatted, contain some
usable knowledge about advanced programming skills.
OPTIONS:
- Yes
- No
"""

## TODO: change prompt_education to be more specific to the task
yes_education="""This is a educational datapoint to give to students during their exams:
###
{datapoint}
###
Does the previous paragraph demarcated within ### and ###
contain informative signal to a large-language model?
An informative datapoint should be well-formatted, contain some
usable knowledge of the world.
OPTIONS:
- Yes
- No
"""