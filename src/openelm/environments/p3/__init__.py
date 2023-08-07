from typing import Optional, Union, List
from openelm.utils.code_eval import get_inputs
P3_PROBLEM_MED_SEED = '''from typing import List

def f1(s: str):
    return "Hello " + s == "Hello world"

def g1():
    return "world"

assert f1(g1())

def f2(s: str):
    return "Hello " + s[::-1] == "Hello world"

def g2():
    return "world"[::-1]

assert f2(g2())

def f3(x: List[int]):
    return len(x) == 2 and sum(x) == 3

def g3():
    return [1, 2]

assert f3(g3())

def f4(s: List[str]):
    return len(set(s)) == 1000 and all(
        (x.count("a") > x.count("b")) and ('b' in x) for x in s)

def g4():
    return ["a"*(i+2)+"b" for i in range(1000)]

assert f4(g4())

def f5(n: int):
    return str(n * n).startswith("123456789")

def g5():
    return int(int("123456789" + "0"*9) ** 0.5) + 1

assert f5(g5())'''

P3_PROBLEM_LONG_SEED = '''from typing import List

def f1(s: str):
    return "Hello " + s == "Hello world"

def g1():
    """Find a string that when concatenated onto 'Hello ' gives 'Hello world'."""
    return "world"

assert f1(g1())

def f2(s: str):
    return "Hello " + s[::-1] == "Hello world"

def g2():
    """Find a string that when reversed and concatenated onto 'Hello ' gives 'Hello world'."""
    return "world"[::-1]

assert f2(g2())

def f3(x: List[int]):
    return len(x) == 2 and sum(x) == 3

def g3():
    """Find a list of two integers whose sum is 3."""
    return [1, 2]

assert f3(g3())

def f4(s: List[str]):
    return len(set(s)) == 1000 and all(
        (x.count("a") > x.count("b")) and ('b' in x) for x in s)

def g4():
    """Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b'."""
    return ["a"*(i+2)+"b" for i in range(1000)]

assert f4(g4())

def f5(n: int):
    return str(n * n).startswith("123456789")

def g5():
    """Find an integer whose perfect square begins with 123456789 in its decimal representation."""
    return int(int("123456789" + "0"*9) ** 0.5) + 1

assert f5(g5())'''

P3_PROBSOL_MED_SEED = '''from typing import List

def f1_1(s: str):
    return "Hello " + s == "Hello world"

def g1_1():
    return "world"

assert f1_1(g1_1())

def f1_2(s: str):
    """Changes the requirements of f1_1"""
    return "Hello " + s == "Hello to the world"

def g1_2():
    return "to the world"

assert f1_2(g1_2())

def f2_1(s: str):
    return "Hello " + s[::-1] == "Hello world"

def g2_1():
    return "world"[::-1]

assert f2_1(g2_1())

def f2_2(s: str):
    """Changes the requirements of f2_1"""
    return s[::-1].swapcase() + " world" == "Hello world"

def g2_2():
    return "Hello"[::-1].swapcase()

assert f2_2(g2_2())

def f3_1(x: List[int]):
    return len(x) == 2 and sum(x) == 3

def g3_1():
    return [1, 2]

assert f3_1(g3_1())

def f3_2(x: List[int]):
    """Changes the requirements of f3_1"""
    return len(x) == 2 and and x[0]+x[1] == 8 and x[0]*x[1] == 12

def g3_2():
    return [2, 6]

assert f3_2(g3_2())

def f4_1(s: List[str]):
    return len(set(s)) == 1000 and all(
        (x.count("a") > x.count("b")) and ('b' in x) for x in s)

def g4_1():
    return ["a"*(i+2)+"b" for i in range(1000)]

assert f4_1(g4_1())

def f4_2(s: List[str]):
    """Changes the requirements of f4_1"""
    return len(set(s)) == 1000 and all(
        (x.count("a") > x.count("b")) and ('b' in x) and (x.count('c')==2) and (x.startswith('cc')) for x in s)

def g4_2():
    return ["cc"+"a"*(i+2)+"b" for i in range(1000)]

assert f4_2(g4_2())

def f5_1(n: int):
    return str(n * n).startswith("123456789")

def g5_1():
    return int(int("123456789" + "0"*9) ** 0.5) + 1

assert f5_1(g5_1())

def f5_2(n: int):
    """Changes the requirements of f5_1"""
    return str((n-10) * (n-10)).startswith("123456789")

def g5_2():
    return (int(int("123456789" + "0"*9) ** 0.5) + 1) + 10

assert f5_2(g5_2())'''

P3_PROBSOL_LONG_SEED = '''from typing import List

def f1_1(s: str):
    return "Hello " + s == "Hello world"

def g1_1():
    """Find a string that when concatenated onto 'Hello ' gives 'Hello world'."""
    return "world"

assert f1_1(g1_1())

def f1_2(s: str):
    """Changes from f1_1: 'to the world' instead of 'world'."""
    return "Hello " + s == "Hello to the world"

def g1_2():
    """Find a string that when concatenated onto 'Hello ' gives 'Hello to the world'."""
    return "to the world"

assert f1_2(g1_2())

def f2_1(s: str):
    return "Hello " + s[::-1] == "Hello world"

def g2_1():
    """Find a string that when reversed and concatenated onto 'Hello ' gives 'Hello world'."""
    return "world"[::-1]

assert f2_1(g2_1())

def f2_2(s: str):
    """Changes from f2_1: swapcase and preprend the given string instead of concatenate to create the phrase"""
    return s[::-1].swapcase() + " world" == "Hello world"

def g2_2():
    """Find a string that when reversed, swapcased, and prepended onto ' world' gives 'Hello world'."""
    return "Hello"[::-1].swapcase()

assert f2_2(g2_2())

def f3_1(x: List[int]):
    return len(x) == 2 and sum(x) == 3

def g3_1():
    """Find a list of two integers whose sum is 3."""
    return [1, 2]

assert f3_1(g3_1())

def f3_2(x: List[int]):
    """Changes from f3_1: change sum to 8 and add requirement for product to equal 12"""
    return len(x) == 2 and and x[0]+x[1] == 8 and x[0]*x[1] == 12

def g3_2():
    """Find a list of two integers whose sum is 8 and product is 12."""
    return [2, 6]

assert f3_2(g3_2())

def f4_1(s: List[str]):
    return len(set(s)) == 1000 and all(
        (x.count("a") > x.count("b")) and ('b' in x) for x in s)

def g4_1():
    """Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b'."""
    return ["a"*(i+2)+"b" for i in range(1000)]

assert f4_1(g4_1())

def f4_2(s: List[str]):
    """Changes from f4_1: add requirement for exactly two 'c's in the front"""
    return len(set(s)) == 1000 and all(
        (x.count("a") > x.count("b")) and ('b' in x) and (x.count('c')==2) and (x.startswith('cc')) for x in s)

def g4_2():
    """Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b' and exactly two 'c's which are in the front."""
    return ["cc"+"a"*(i+2)+"b" for i in range(1000)]

assert f4_2(g4_2())

def f5_1(n: int):
    return str(n * n).startswith("123456789")

def g5_1():
    """Find an integer whose perfect square begins with 123456789 in its decimal representation."""
    return int(int("123456789" + "0"*9) ** 0.5) + 1

assert f5_1(g5_1())

def f5_2(n: int):
    """Changes from f5_1: 10 must be subtracted from the integer first"""
    return str((n-10) * (n-10)).startswith("123456789")

def g5_2():
    """Find an integer for which the output of subtracting 10 and squaring the result begins with 123456789 in its decimal representation."""
    return (int(int("123456789" + "0"*9) ** 0.5) + 1) + 10

assert f5_2(g5_2())'''


def P3_probsol_chat_med_seed(list_few_shot_example :Optional[List[str]] = [], code_batch: Optional[List[str]] = [],new_puzzles = 3) -> str: 
    """
    prompt for mutation
    new_puzzles: how many puzzles to generate should pass it as parameters
    elm_mode
    TODO: add mutation if code_batch is given
    """
    N_python_problem = len(list_few_shot_example)
    if isinstance(code_batch, str):
        code_batch = [code_batch]
    # elm_mode is activated if code_batch is not empty
    # elm_mode: few shot example + prompt mutation + code_batch
    elm_mode = False
    mutate_pb = ""
    if len(code_batch)>0:
        elm_mode = True
        mutate_pb = code_batch[0]
        "Here is a new puzzle:"
    # prompt_elm = "Please mutate this new puzzle into a related but different puzzle that requires similar"
    # for puzz in range(len(list_few_shot_example)):
        
    #     few_shot_examples+=f"Puzzle {N_python_problem+puzz}:\n```\n{list_few_shot_example[puzz]}\n```\n"
    # N_python_problem= +len(code_batch)
    # puzzles for fixed prompt not used anymore
    puzz_0=f'''Puzzle 0:
```
def f(start: int, k=2, upper=-172, seq=[79, 18, -98, -13, 88, -93, -77, -95, 40, -3, -22]) -> bool:
    """Find a sequence of k consecutive indices whose sum is minimal"""
    return 0 <= start <= len(seq) - k and sum(seq[start:start + k]) <= upper

def g(k = 2, upper = -172, seq = [79, 18, -98, -13, 88, -93, -77, -95, 40, -3, -22]):
    return min(range(len(seq) - k + 1), key=lambda start: sum(seq[start:start + k])) 
assert f(g()) == True
```
---'''
    puzz_1 = '''Puzzle 1:
```
def f(i: int, li=[-60, 9, 1, -42, 31, 70, 5, 1, 42, -90, -20], target=-42) -> bool:
    """Find the index of an item in a list using negative indexing."""
    return li[i] == target and i < 0

def g(li = [-60, 9, 1, -42, 31, 70, 5, 1, 42, -90, -20], target = -42):
    return li.index(target) - len(li) 
assert f(g()) == True
```
---'''
    puzz_2 = '''Puzzle 2:
```
from typing import*
def f(ans: List[List[int]], target=2) -> bool:
    """
    Find a list of pairs of integers where the number of pairs in which the second number is more than
    two greater than the first number is a given constant
    """
    for i in range(len(ans)):
        a, b = ans[i]
        if b - a >= 2:
            target -= 1
    return target == 0

def g(target = 2):
    return [[0, 2]] * target 
assert f(g()) == True
```
---''' 
    all_puzzle_str = ""
    for idx_puzz in range(len(list_few_shot_example)):
        all_puzzle_str += f"Puzzle {idx_puzz}:\n```\n{list_few_shot_example[idx_puzz]}\n```\n---\n"
        
    instruction_p3_puzzle= "Note that the first argument of f is the output g(), so you must not give the first argument of f to g. Make sure to define and set values for all arguments of function 'f' (excluding the first argument, which will be provided by function 'g'). Both functions, 'f' and 'g' should have matching argument signatures: def f(arg0, arg1=value1, arg2=value2, ...) and def g(arg1=value1, arg2=value2, ...). Failing to set and provide values for all arguments may result in an incorrect solution. Additionally, make sure to import any necessary libraries to ensure your code runs smoothly."
    prompt = f'''You will be given {N_python_problem+elm_mode} (Puzzle 0 to Puzzle {N_python_problem-1+elm_mode}) Python Programming Puzzle (P3). A P3 consists of a problem f and its corresponding solution g. The puzzle is solved if f(g()) == True. Your role is to generate {new_puzzles} new puzzles (Puzzle {N_python_problem+elm_mode} to Puzzle {N_python_problem+new_puzzles+elm_mode-1}). {instruction_p3_puzzle}
----
{all_puzzle_str}'''
    if elm_mode:
        prompt += "Here is the puzzle to mutate:\n"
        prompt += f"Puzzle {len(list_few_shot_example)}:\n"
        prompt += f"```\n{mutate_pb}\n```\n---\n"
        # prompt += f"Could you please create {new_puzzles} new, interesting, diverse, and valid Python programming puzzles by mutating the Puzzle {len(list_few_shot_example)}? Make sure that the new puzzles are distinctly different from each other."
        # prompt += f"Could you please create {new_puzzles} new, interesting, diverse, and valid Python programming puzzles by mutating the original Puzzle {len(list_few_shot_example)}? Ensure the mutated puzzles are meaningfully different from each other."
        # prompt += f"Could you please mutate this new puzzle into another interesting correct Python Programming Puzzle? Ensure the mutated puzzle is meaningfully different from the existing {len(list_few_shot_example)+elm_mode} puzzles."
        prompt += f"Could you please mutate the Puzzle {len(list_few_shot_example)} into {new_puzzles} new interesting correct Python Programming Puzzles? Please, ensure the mutated puzzles are meaningfully different from the existing puzzles."

    return prompt

def prompt_solve_puzzle_given_f(problem_str: str): 
    """
    prompt to solve a puzzle (generate g) given f
    """
    arg_sol= "..."#get_inputs(problem)
    f = problem_str.split("def g")[0]
    few_shot_ex = 3
    PY_SIMPLE_CHAT_INSTRUCTION_V2 = "You are a world-class mathematician and a world-class Python developer with an eagle eye for unintended bugs and edge cases, that only responds with only python code. You will be given a function and its docstring. Respond only in code with a correct, efficient implementation of the function."
    prompt_base = "Now, you need to generate the correct solutions (g), for the Problem "+str(few_shot_ex)+" that satisfies the condition f(g) == True."
    prompt_base += "\nYou will give the solution (def g("+arg_sol+")) to the last problem f. Don't forget that the first argument of f is the value returned by g(), so it is not given to g."
    fewshot_problems = f'''----
Problem 0:
```
def f(stamps: List[int], target=80, max_stamps=4, options=[10, 32, 8]) -> bool:
    """Find a selection of at most max_stamps stamps whose total worth is the target value."""
    for s in stamps:
        assert s in options
    return len(stamps) <= max_stamps and sum(stamps) == target
```
Solution 0:
```

def g(target = 80, max_stamps = 4, options = [10, 32, 8]):
    from itertools import combinations_with_replacement
    for n in range(max_stamps + 1):
        for c in combinations_with_replacement(options, n):
            if sum(c) == target:
                return list(c)
assert f(g())
```
---
Problem 1:
```
from typing import*
def f(ans: List[List[int]], target=2) -> bool:
    """
    Find a list of pairs of integers where the number of pairs in which the second number is more than
    two greater than the first number is a given constant
    """
    for i in range(len(ans)):
        a, b = ans[i]
        if b - a >= 2:
            target -= 1
    return target == 0
```
Solution 1:
```
def g(target = 2):
    return [[0, 2]] * target 
assert f(g()) == True
```
---
Problem 2:
```
def f(n: int, v=313946483, w=806690290) -> bool:
    """Find the smallest n such that if v is tripled n times and w is doubled n times, v exceeds w."""
    for i in range(n):
        assert v <= w
        v *= 3
        w *= 2
    return v > w
```
Solution 2:
```
def g(v = 313946483, w = 806690290):
    i = 0
    while v <= w:
        v *= 3
        w *= 2
        i += 1
    return i 
assert f(g()) == True
```
---
----
Now, you need to generate the correct solutions (g), for the following problem 3 that satisfies the condition f(g()) == True.
You will give the solution (def g...) to the last problem f. Don't forget that the first argument of f is the value returned by g(), so it is not given to g.
----
Problem 3:
```
{f}
```'''
    full_prompt = PY_SIMPLE_CHAT_INSTRUCTION_V2 + "\n" + prompt_base + "\n" + fewshot_problems 
    return full_prompt

P3_IMPORTS = "from typing import*\n"#"from typing import List\n" # The only import that's necessary as of P3 v0.2

__all__ = [
    "P3_PROBLEM_MED_SEED",
    "P3_PROBLEM_LONG_SEED",
    "P3_PROBSOL_LONG_SEED",
    "P3_IMPORTS",
]