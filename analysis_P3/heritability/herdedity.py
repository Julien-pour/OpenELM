import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
# top_logprobs
import sys 
sys.path.append("/gpfsdswork/projects/rech/imi/uqv82bm/OpenELM/")
from heritability import main

prompt_elm_to_test = """You are a helpful assistant to a Professor teaching a programming course in Python. The Professor wants to give some puzzles to his master's students to teach them Python.
I already have a series of Python Programming Puzzles (P3). Each puzzle consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in constructing a SAT problem `f` and a function `g` such that `f(g())` evaluates to `True`.
I will provide two existing puzzles for reference, and I need you to create five new distinct puzzles (Puzzle 3 to Puzzle 7), each being a slight variation derived from Puzzle {N}.

Rules:
- Each puzzle includes two functions: `def f(...)` and `def g(...)`.
- The first argument of `f` is always the output from `g()`.
- Ensure `f` and `g` have matching argument signatures (e.g., `def f(arg0, arg1=value1, arg2=value2, ...)` and `def g(arg1=value1, arg2=value2, ...)`).
- Avoid using `f` inside `g`, and `g` inside `f`.
- Include any necessary imports so your code runs smoothly.
- Give a clear Puzzle description that must be brief and diverse compared to the other puzzles.
- Make sure the puzzle is self-contained within these two functions.

Puzzle Format:
Puzzle description: A brief, one to two sentence summary of the puzzle's content.
```python
def f(solution, args=...) -> bool:
    # Python code to test the solution returned by g.
    # This function is a test unit and must return True if the solution is correct, and False otherwise.

def g(args=...) -> solution:
    # Python code to generate a solution for the problem.
    # The solution should generalize to all possible args.
    return solution

assert f(g()) == True
```
{examples}
Your Task:
Create five new Python Programming Puzzles (Puzzle 3 to Puzzle 7).
Ensure that each new puzzle is inspired by Puzzle {N}.
"""

prompt_aces="""You are a helpful assistant to a Professor teaching a programming course in Python. The Professor wants to give some puzzles to his master's students to teach them Python.
I already have a series of Python Programming Puzzles (P3). Each puzzle consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in constructing a SAT problem `f` and a function `g` such that `f(g())` evaluates to `True`.
I will provide two existing puzzles for reference, and I need you to create five new distinct puzzles (Puzzle 3 to Puzzle 7)

Rules:
- Each puzzle includes two functions: `def f(...)` and `def g(...)`.
- The first argument of `f` is always the output from `g()`.
- Ensure `f` and `g` have matching argument signatures (e.g., `def f(arg0, arg1=value1, arg2=value2, ...)` and `def g(arg1=value1, arg2=value2, ...)`).
- Avoid using `f` inside `g`, and `g` inside `f`.
- Include any necessary imports so your code runs smoothly.
- Give a clear Puzzle description that must be brief and diverse compared to the other puzzles.
- Make sure the puzzle is self-contained within these two functions.

Puzzle Format:
Puzzle description: A brief, one to two sentence summary of the puzzle's content.
```python
def f(solution, args=...) -> bool:
    # Python code to test the solution returned by g.
    # This function is a test unit and must return True if the solution is correct, and False otherwise.

def g(args=...) -> solution:
    # Python code to generate a solution for the problem.
    # The solution should generalize to all possible args.
    return solution

assert f(g()) == True
```
{examples}
Your Task:
Create five new Python Programming Puzzles (Puzzle 3 to Puzzle 7) inspired by the previous puzzles I gave you. 
Please ensure that the each new puzzles test **all** of the following skills: 
{skills}

Make sure that the new problems are more complicated by at least one of those elements:
- for Puzzle 3, choose a puzzle and make it harder.
- for Puzzle 4, you can add a new constraint to a puzzle that I have given to you.
- for Puzzle 5, you can make some reasoning steps harder for LLM such as ChatGPT.
- for Puzzle 6, you can add more reasoning steps to the puzzle.
- for Puzzle 7, you can create a hybrid puzzle by combining the puzzles that I have given.
"""

all_wizard_prompt=["Add new constraints and requirements to the original problem, adding approximately 10 additional words.",
"Replace a commonly used requirement in the programming task with a less common and more specific one.",
"If the original problem can be solved with only a few logical steps, please add more reasoning steps.",
"Provide a piece of erroneous code as a reference to increase misdirection.",
"Propose higher time or space complexity requirements, but please refrain from doing so frequently."]
# constr_aces=
new_constr="Add new constraints and requirements to the original problem, adding approximately 20 additional words."
constraint="Make sure the new problems are no easier than the given problem and require more steps to solve."

last_constraint="""Make sure that the new problems are more complicated by one of those elements:
- for Puzzle 4, you can add a new constraint to the original problem
- for Puzzle 5, you can increase the complexity of the problem."
- for Puzzle 6, you can add more reasoning steps to the problem 
- for Puzzle 7, you can make some reasoning steps harder for LLM such as ChatGPT. 
"""
prompt_elm_to_test= prompt_elm_to_test+last_constraint

metric_dict = main(prompt_to_test=prompt_aces, num_puz=64,config_name="aces")
print(metric_dict)