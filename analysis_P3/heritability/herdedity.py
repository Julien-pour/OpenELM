import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
# top_logprobs
import sys 
sys.path.append("/gpfsdswork/projects/rech/imi/uqv82bm/OpenELM/")
from heritability import main

prompt_rd_gen = """Consider Python Programming Puzzles (P3). P3 consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in constructing a SAT problem `f` and a function `g` such that `f(g())` evaluates to `True`

## Main Rules:
- Each puzzle includes two functions: `def f(...)` and `def g(...)`.
- The first argument of `f` is always the output from `g()`.
- Ensure `f` and `g` have matching argument signatures (e.g., `def f(arg0, arg1=value1, arg2=value2, ...)` and `def g(arg1=value1, arg2=value2, ...)`). You also need to set the value of argument of f (arg1,arg2,...) and g when you define them.
- Avoid using `f` inside `g`, and `g` inside `f`.
- Include any necessary imports so your code runs smoothly.
- Give a clear Puzzle description that must be brief and diverse compared to the other puzzles.
- Make sure the puzzle is self-contained within these two functions.

## P3 Format:
Puzzle description: A two to four sentence summary of the puzzle's content. To explain what is the problem `f`, and how you can solve it with `g`. 
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

## Examples:
{examples}

Generate 5 different P3 similar to previous Examples.

## New 5 problems:
"""
prompt_elm= """Consider Python Programming Puzzles (P3). P3 consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in constructing a SAT problem `f` and a function `g` such that `f(g())` evaluates to `True`

## Main Rules:
- Each puzzle includes two functions: `def f(...)` and `def g(...)`.
- The first argument of `f` is always the output from `g()`.
- Ensure `f` and `g` have matching argument signatures (e.g., `def f(solution, arg1=value1, arg2=value2, ...)` and `def g(arg1=value1, arg2=value2, ...)`). You also need to set the value of argument of f (arg1,arg2,...) and g when you define them.
- Avoid using `f` inside `g`, and `g` inside `f`.
- Include any necessary imports so your code runs smoothly.
- Give a clear Puzzle description that must be brief and diverse compared to the other puzzles.
- Make sure the puzzle is self-contained within these two functions.

## P3 Format:
Puzzle description: A two to four sentence summary of the puzzle's content. To explain what is the problem `f`, and how you can solve it with `g`. 
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

## Examples:
{examples}

Generate 5 P3 similar to the last Examples (Puzzle 2). Ensure that all new puzzles are more challenging than Puzzle 2.

## New 5 problems inspired by Puzzle 2:
"""

prompt_aces= """Consider Python Programming Puzzles (P3). P3 consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in constructing a SAT problem `f` and a function `g` such that `f(g())` evaluates to `True`

## Main Rules:
- Each puzzle includes two functions: `def f(...)` and `def g(...)`.
- The first argument of `f` is always the output from `g()`.
- Ensure `f` and `g` have matching argument signatures (e.g., `def f(solution, arg1=value1, arg2=value2, ...)` and `def g(arg1=value1, arg2=value2, ...)`). You also need to set the value of argument of f (arg1,arg2,...) and g when you define them.
- Avoid using `f` inside `g`, and `g` inside `f`.
- Include any necessary imports so your code runs smoothly.
- Give a clear Puzzle description that must be brief and diverse compared to the other puzzles.
- Make sure the puzzle is self-contained within these two functions.

## P3 Format:
Puzzle description: A two to four sentence summary of the puzzle's content. To explain what is the problem `f`, and how you can solve it with `g`. 
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

## Examples:
{examples}

Generate 5 P3 similar to previous Examples. Ensure that all new puzzles are more challenging than Puzzle from previous examples.
Please make sure that new puzzles test the all the following skills:{skill_target}
## New 5 problems inspired by Puzzle 2:
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
metric_dict = main(num_puz=2,config_name="aces")

print(metric_dict)
