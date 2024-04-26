# should probably move that to smthing like prompt.py
from typing import Optional, Union, List
import json

import numpy as np
import textwrap
from pydantic import BaseModel,Field
from openelm.utils.code_eval import find_first_argument_of_first_function

from openelm.environments.p3.prompt_code import base_persona_code, prompt_gen_description


skill_list = [
    "String Manipulation",
    "Mathematical Operations",
    "Conditional Logic",
    "Recursion",
    "Brute Force Search",
    "Dynamic Programming",
    "Greedy Algorithms",
    "Backtracking",
    "Set Operations",
    "Permutations and Combinations",
    "Probability and Statistics",
    "Pattern Recognition", 
    "Sorting and Ordering",
    "Binary Operations (bitwise shifting, AND, OR)",
    "Geometry and Coordinate Manipulation",
    "Algorithm Optimization",
    "Number Theory (factors, primes, etc.)",
    "Graph Theory (paths, edges, vertices)",
    "Array Indexing",
    "Hashing"
]
# add Set Operations and Hashing




# class for instructor skill labelling prompt for P3
def get_class_PuzzleCheck(mode):
    match mode:
        case "description":
            class PuzzleCheck(BaseModel):
                """Puzzle description and if it should be given to the student or not."""
                puzzle_description: str = Field(description="Provide a brief, one to two sentence summary of the puzzle's content.")

        case "description+is_valid":
            class PuzzleCheck(BaseModel):
                """Puzzle description and if it should be given to the student or not."""
                puzzle_description: str = Field(description="Provide a brief, one to two sentence summary of the puzzle's content.")
                explanations: str = Field(decription="Short explanation of whether the puzzle should be given to the student or not.")
                give_puzzle_to_student: bool = Field(description="Whether the puzzle should be given to student or not based on the previous explanations")
    return PuzzleCheck

class Topics_evaluation(BaseModel):
    """List of topics that are used in the problem and solution."""
    explanations_index_topics: str = Field(decription="Short explanation of the specific topics employed in the puzzle.")
    index_topics: List[int] = Field(description="list of at most 5 index correponding to topics that are actually used in the problem `f` or the solution `g`")


# class Puzzle_Quality_Diversity(BaseModel):
#     """Evaluate the quality of a given pair of programming problem and solution."""
#     interestingness: Puzzle_Interestingness = Field(description="evaluate the interestingness of the problem-solution pair")
#     puzzle_check: PuzzleCheck = Field(description="check the validity of the problem-solution pair")
#     topics: Topics_evaluation = Field(description="list of topics that are used in the problem and solution")


# class Puzzle_Interestingness(BaseModel):
#     """Evaluate the interestingness of a pair of programming problem and solution. The problem should be complexe and original, it should be relevant as a leetcode problem. And it should serve as good learning tools and have high educational value for **master's student**."""
#     puzzle_description: str = Field(description="Provide a brief, one to two sentence summary of the puzzle's content.")
#     interestingness_score_f: int = Field(description="Assess the level of interest in the problem (function f) on a scale of 0 to 10, where the rating must be an integer.")
#     interestingness_score_g: int = Field(description="Assess the level of interest in the solution (function g) on a scale of 0 to 10, where the rating must be an integer.")
# # maybe interestingne in term of pedagogy?




def create_prompt_label(puzzle : str, mode="give_skills"):
    """
    create prompt for label_puzzle goes with Topics_evaluation class with give_skills=True
    mode = "give_skills", "is_valid", "description", "description+is_valid", "general"
    is_valid -> filtering 
    description use to give a description of the puzzle
    """

    level = "master's student in CS"#"master's student"
    # skills format
    format_skills=""
    for idx,skill in enumerate(skill_list):
        format_skills+=f"{idx}. {skill}\n"
    skills = f"\n{format_skills}"
    
    base_persona = base_persona_code.format(level=level)
    match mode:
        case "is_valid": # WIP should also use a persona to label the puzzle
            prompt=base_persona
            prompt += "Your role is to check if the following puzzle could be used or not."

        case "description": # WIP 
            arg=find_first_argument_of_first_function(puzzle)
            puzzle=puzzle.split('def g')[0].strip() + "\n\ndef g(...):\n\nassert f(g()) == True"
            prompt=prompt_gen_description.format(arg_sol=arg,arg_solb=arg,puzzle=puzzle)

        case "description+is_valid": # WIP
            arg=find_first_argument_of_first_function(puzzle)
            puzzle=puzzle.split('def g')[0].strip() + "\n\ndef g(...):\n\nassert f(g()) == True"
            prompt=prompt_gen_description.format(arg_sol=arg,arg_solb=arg,puzzle=puzzle)
            prompt += f"\nThen you should check if the following puzzle could be used or not to teach Python to {level}."

        case "give_skills":
            prompt = base_persona+"\n"
            prompt+= "The Professor want to evaluate the diversity of those puzzles, can you label the following puzzle given the following list of topics, please?"
            # prompt = "Your role is: given the following puzzle, and the list of topics, exctract the information requested."
            prompt += "\nThe list of topics is:\n"+ skills 

        case "general":
            prompt= "Given the following puzzle, exctract the information requested."
    
    prompt += "\n\nThe puzzle is:\n```python\n" + puzzle + "\n```\n"
            
    return prompt


def get_programming_puzzles_prompt(
        list_few_shot_example : List[str],
        code_batch: Optional[List[str]] = None,
        skill_targeted: Optional[List[int]]=None,
        n_fewshot_ex=2,
    ):
    """
    should change that to list_few_shot_example from list to Phenotype type
    skill_targeted list of binary vector [(0/1)]^n_skills indicating if the skill is targeted or not
    remove n_fewshot_ex
    """
    elm_mode=False
    prompt_elm=""
    prompt2add=""
    aces_mode=False
    if not code_batch is None:
        elm_mode = True
    if not skill_targeted is None:
        aces_mode = True

    if not isinstance(list_few_shot_example, list):
        list_few_shot_example = [list_few_shot_example]
    if all(isinstance(x, str) for x in list_few_shot_example):
        raise NameError("should be phenotype not str") 

    puzzles = [puzz for puzz in list_few_shot_example[:n_fewshot_ex]]
    
    examples = ""
    for i, puzzle in enumerate(puzzles):
        puzzle_description = puzzle.description # /!\ need to implement that puzzle.description /!\

        examples += f"\nPuzzle {i}:\nPuzzle description: {puzzle_description}\n```python\n{puzzle.program_str}\n```\n"
    if elm_mode:
        puzzle_description = code_batch[0].description
        p_str = code_batch[0].program_str
        prompt_elm=f", each being a **mutation** derived from Puzzle {i+1}" #the structure of Puzzle 2
        examples += f"\nPuzzle {i+1} (to mutate):\nPuzzle description: {puzzle_description}\n```python\n{p_str}\n```\n"

    # /!\ should use persona (could be an improved version) smthing like:
    # You are a helpful assistant to a Professor teaching an undergraduate programming course in Python. 
    # The teacher have assign to undergraduate student in CS to create a python programming puzzle.
    # The Professor want to evaluate the diversity of those puzzles, can you label the puzzles please?
    base_persona ="You are a helpful assistant to a Professor teaching a programming course in Python. "
    base_persona += "The Professor wants to give some puzzles to his master's students to teach them Python.\n" # student -> Master student
    prompt = base_persona 
    main_prompt = """    I already have a series of Python Programming Puzzles (P3). Each puzzle consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in constructing a SAT problem `f` and a function `g` such that `f(g())` evaluates to `True`.
    I will provide two existing puzzles for reference, and I need you to create five new distinct puzzles (Puzzle 2 to Puzzle 6){prompt_elm}.
    
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
    Create five new Python Programming Puzzles (Puzzle 2 to Puzzle 6)."""
    prompt += textwrap.dedent(main_prompt)
    if elm_mode == True:
        prompt2add = f" Ensure that each new puzzle is created by making mutations to Puzzle {i+1}."
    if aces_mode == True:
        skill_target=" "
        idx_skill_targeted = [idx for idx, val in enumerate(skill_targeted) if val]
        for idx in idx_skill_targeted:
            skill_target += f"\n- {skill_list[idx]}"

        # previous version
        prompt2add = " Ensure that each puzzle is meaningfully different from the provided examples and from each other." #The puzzles should be challenging."# and adhere to the specified format."
        # new version Make sure the new problems are no easier than the given problem
        prompt2add = " Ensure that the newly created puzzles are of comparable or greater difficulty."

        prompt2add += f" Please make sure that new puzzles are based on ** all ** the following skills:{skill_target}"

    prompt = prompt.format(examples=examples,prompt_elm=prompt_elm)
    prompt += prompt2add
    return prompt








def prompt_solve_puzzle_given_f(problem_str: str): 
    """
    prompt to solve a puzzle (generate g) given f
    """
    arg_sol= "..."#get_inputs(problem)
    f = problem_str.split("def g")[0]
    few_shot_ex = 2
    PY_SIMPLE_CHAT_INSTRUCTION_V2 = "You are a world-class Python developer with an eagle eye for unintended bugs and edge cases, that only responds with python code. You will be given a function and its docstring. Respond only in code with a correct, efficient implementation of the function."
    prompt_base = "Now, you need to generate the correct solutions (g), for the Problem "+str(few_shot_ex)+" that satisfies the condition f(g) == True."
    prompt_base += "\nYou will give the solution (def g("+arg_sol+")) to the last problem f. Don't forget that the first argument of f is the value returned by g(), so it is not given to g."
    fewshot_problems = f'''----
Problem 0:
```python
def f(stamps: List[int], target=80, max_stamps=4, options=[10, 32, 8]) -> bool:
    """Find a selection of at most max_stamps stamps whose total worth is the target value."""
    for s in stamps:
        assert s in options
    return len(stamps) <= max_stamps and sum(stamps) == target
```
Solution 0:
```python
def g(target = 80, max_stamps = 4, options = [10, 32, 8]):
    from itertools import combinations_with_replacement
    for n in range(max_stamps + 1):
        for c in combinations_with_replacement(options, n):
            if sum(c) == target:
                return list(c)

assert f(g()) == True
```

Problem 1:
```python
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
```python
def g(target = 2):
    return [[0, 2]] * target 

assert f(g()) == True
```

Now, you need to generate the correct solutions (def g), for the following Problem 2 that satisfies the condition f(g()) == True.
----
Problem 2:
```python
{f}
```'''
    full_prompt = PY_SIMPLE_CHAT_INSTRUCTION_V2 + "\n" + prompt_base + "\n" + fewshot_problems 
    return full_prompt




### mostly old stuff (need to check and remove it)











old_skill_list = ['Sorting and Searching', 'Counting and combinatorics', 'Tree and Graph problem', 'Math', 'Bit Manipulation', 'String Manipulation', 'Geometry', 'Recursion and Dynamic Programming', 'Stacks and Queues', 'Optimization']

SKILLS ="""
0 - Sorting or Searching: Sorting refers to arranging a data structure (list, string, grid,...) in a specific order, typically in ascending or descending order. Searching involves finding the location or presence of a particular element or pattern in a data structure (list, string, grid,...).
1 - Counting or combinatorics: Understanding principles of counting and combinatorial analysis, including permutations, combinations, and other counting techniques. These skills are essential for solving problems that involve counting the number of possibilities, occurrence or arrangements. Counting the number of occurrences of something also falls in this category.
2 - Tree and Graph problem: Analyzing and solving problems related to tree and graph structures involving nodes connected by edges. This includes tasks such as graph or tree traversal, finding shortest paths, detecting cycles, and determining connectivity between nodes, heap, problems on grids...
3 - Math: Strong understanding of mathematical concepts such as summations, probability, arithmetics, polynomials, solving equations, matrices, algebra, formal and informal logic....
4 - Bit Manipulation: Performing operations at the bit level to solve problems. This includes boolean operation (and, or, not, xor, etc), bit shifting, bit masking, and other bitwise operations.
5 - String Manipulation: Operations and algorithms specifically designed for working with strings. This includes tasks like concatenation, searching, replacing, parsing strings, and pattern matching.
6 - Geometry and Grid Problems: Understanding geometric concepts and algorithms for geometrical problem-solving. For instance puzzles involving shapes on the plane (triangles, etc), angles, figures, space, surfaces, curvature, 3d geometry, discrete geometry, rotations...
7 - Recursion or Dynamic Programming: Utilizing recursive techniques and dynamic programming to solve problems by factoring them down into smaller subproblems and building solutions incrementally. Puzzles that can be solved through a modular approach.
8 - Stacks or Queues: Data structures used to store and retrieve elements in a specific order. Stacks follow Last-In-First-Out, while queues follow First-In-First-Out. They are used for managing function calls, recursion, and implementing search algorithms.
9 - Optimization: Problems involving finding the best possible solution for a given problem by minimizing or maximizing an objective function. In this category go all puzzles involving finding maximal and minimal elements, shortest and longest paths, brute force search, etc...
"""

def skills_evaluation(puzzle):
    # import openai
    # openai.api_type = "azure"
    # openai.api_key = AZURE_OPENAI_KEY
    # openai.api_base = AZURE_OPENAI_ENDPOINT
    # openai.api_version = "2023-05-15"

    # zero shot evaluation of skills
    n_skills = 10
    skills = f"""{SKILLS}

"""

    start_prompt = "I will give you a Python programming puzzle f (and its solution g) and a list of programming skills. Your role is to say which programming skills are required to understand and solve the problem.\n"
    end_prompt = "\nFirst, you can write each category and explain with at most one sentence why it is required. Then summarize your answer by writing every index of categories in a Python list as follows (you must always write correctly the following text): Therefore, the list of indices for the problem is: <Python list>"
    format_answer='Then, you need to summarize your answer by writing the index of every required skills given your reasoning in a Python list, following the format provided below. Please ensure the correct usage of the following text where <Python list> is a list with numbers from 0 to 9: "Therefore, the list of skills for the puzzle is: <Python list>"'
    # end_prompt_v2 = 'After completing your reasoning (you can start by explaining the problem and the solution in a few sentences). Ensure you remove every listed skills that are unnecessary for understanding or solving the given problem.'+format_answer
    end_prompt_v2_b = 'After completing your reasoning (you can start by explaining the problem and the solution in a few sentences).' +format_answer
    end_prompt_v3 = 'First start by explaining the problem and the solution in a few sentences. Second write the name of each skills and explain if it is required or not required for understanding or solving the puzzle above.'+format_answer
    puzzle_prompt = "The puzzle is:\n```python\n" +puzzle + "\n```\n"
    end_prompt_v2 = 'After completing your reasoning (you can start by explaining the problem and the solution in a few sentences). Ensure you remove every listed skills that are unnecessary for understanding or solving the given problem. It is necessary to summarize your answer by writing every index of categories explicitly used in the problem or solution in a Python list, following the format provided below. Please ensure the correct usage of the following text where <Python list> is a list with numbers from 0 to 9: "Therefore, the list of indices for the problem is: <Python list>"' 

    full_prompt = start_prompt + str("skills: ")+ skills + puzzle_prompt + end_prompt_v2
    return full_prompt, n_skills




def P3_probsol_chat_med_seed(list_few_shot_example :Optional[List[str]] = [], code_batch: Optional[List[str]] = [],new_puzzles = 3) -> str: 
    """
    prompt for mutation
    new_puzzles: how many puzzles to generate should pass it as parameters
    elm_mode
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
        N_python_problem= N_python_problem-1
        list_few_shot_example=list_few_shot_example[:-1]
        # "Here is a new puzzle:"
    # prompt_elm = "Please mutate this new puzzle into a related but different puzzle that requires similar"
    # for puzz in range(len(list_few_shot_example)):
        
    #     few_shot_examples+=f"Puzzle {N_python_problem+puzz}:\n```\n{list_few_shot_example[puzz]}\n```\n"
    # N_python_problem= +len(code_batch)
    # puzzles for fixed prompt not used anymore
    all_puzzle_str = ""
    for idx_puzz in range(len(list_few_shot_example)):
        all_puzzle_str += f"Puzzle {idx_puzz}:\n```\n{list_few_shot_example[idx_puzz]}\n```\n---\n"
        
    # additional_instruct = "If you give examples, make sure that these are with different arguments than the arguments of the function."

    instruction_p3_puzzle= """Note that the first argument of f is the output g(). Make sure to define and set values for all arguments of the function 'f' (excluding the first argument, as it is the solution that needs to be found and given by g).
Both functions, 'f' and 'g' should have matching argument signatures: def f(arg0, arg1=value1, arg2=value2, ...) and def g(arg1=value1, arg2=value2, ...). Please provide all values (value1, value2, ... ) for all arguments. For example f(solution,arg1=1, arg2=2, ...) and g(arg1=1, arg2=2, ...). And you should not use f inside g.
Additionally, make sure to import any necessary libraries to ensure your code runs smoothly."""
    prompt = f'''I will give you {N_python_problem+elm_mode} (Puzzle 0 to Puzzle {N_python_problem-1+elm_mode}) Python Programming Puzzle (P3). A P3 consists of a problem f and its corresponding solution g. The puzzle is solved if f(g()) == True. Your role is to generate {new_puzzles} new puzzles (Puzzle {N_python_problem+elm_mode} to Puzzle {N_python_problem+new_puzzles+elm_mode-1}). {instruction_p3_puzzle}
----
{all_puzzle_str}'''
    if elm_mode:
        prompt += "Here is the puzzle to mutate:\n"
        prompt += f"Puzzle {len(list_few_shot_example)}:\n"
        prompt += f"```\n{mutate_pb}\n```\n---\n"
        prompt += f"Could you please mutate the Puzzle {len(list_few_shot_example)} into {new_puzzles} new correct Python Programming Puzzles (from Puzzle {N_python_problem+elm_mode} to Puzzle {N_python_problem+new_puzzles+elm_mode-1})? Please, ensure the mutated puzzles are meaningfully different from the existing puzzles."
    return prompt

def P3_probsol_chat_med_seed_goal_targeted(list_few_shot_example, skill_targeted: List[bool],new_puzzles = 3) -> str: 
    """
    prompt for guided goal mutation
    list_few_shot_example: list of Phenotype
    new_puzzles: how many puzzles to generate should pass it as parameters
    skill_targeted: list of boolean indicating if the skill is targeted or not  e.g [0,1,0,0,1,...]
    elm_mode
    
    """
    idx_skill_targeted = [idx for idx, val in enumerate(skill_targeted) if val]
        
    N_python_problem = len(list_few_shot_example)
    print()
    skill_list_str=""
    if len(skill_targeted)==0:
        skill_list_str+="None"
    else:
        for idx in idx_skill_targeted:
            skill_list_str+=f"{idx} - {skill_list[idx]}\n"
    all_puzzle_str = ""
    for idx_puzz in range(len(list_few_shot_example)):
        idx_curr_puzz = [idx for idx, val in enumerate(list_few_shot_example[idx_puzz].emb) if val]
        all_puzzle_str += f"Puzzle {idx_puzz}, required skills {idx_curr_puzz} :\n```\n{list_few_shot_example[idx_puzz].program_str}\n```\n---\n"
    skills = f"""{SKILLS}
"""
    instruction_p3_puzzle= """Note that the first argument of f is the output g(). Make sure to define and set values for all arguments of the function 'f' (excluding the first argument, as it is the solution that needs to be found and given by g).
Both functions, 'f' and 'g' should have matching argument signatures: def f(arg0, arg1=value1, arg2=value2, ...) and def g(arg1=value1, arg2=value2, ...). Please provide all values (value1, value2, ... ) for all arguments. For example f(solution,arg1=1, arg2=2, ...) and g(arg1=1, arg2=2, ...). And you should not use f inside g.
Additionally, make sure to import any necessary libraries to ensure your code runs smoothly."""

    prompt_skills = f"""In addition each of those puzzles are associated with a list of skills. Here is a detailed description of those skills: {skills}Your role is to generate {new_puzzles} new puzzles (Puzzle {N_python_problem} to Puzzle {N_python_problem+new_puzzles-1}) that require those skills: {idx_skill_targeted}.
{instruction_p3_puzzle} Please ensure the mutated puzzles fall into all those skills: {idx_skill_targeted}."""

    end_prompt_backup='''And respect all criteria:
-A problem function f that takes the output of a solution function g as its first argument, with additional arguments as necessary. Define and provide default values for all arguments of f (except the first one, which is provided by g).
-A solution function g with matching argument signatures to f, including all default values.
-Ensure the puzzle is considered solved if f(g()) == True.'''
    end_prompt_backup_v2='''And respect all criteria resumed here:
    - Make sure you add default args for all args of f except the first one as in the provided examples.
    - The puzzle is considered solved if f(g()) == True.'''

    end_prompt='''And respect all criteria resumed here:
    - Make sure you add default args for all args of f except the first one as in the provided examples.
    - The puzzle is considered solved if f(g()) == True.
    - You can't use f inside the function g
    - The function g can only return only one variable or result.'''
    prompt = f'''I will give you {N_python_problem} (Puzzle 0 to Puzzle {N_python_problem-1}) Python Programming Puzzle (P3). A P3 consists of a problem f and its corresponding solution g. The puzzle is solved if f(g()) == True. Your role is to generate new puzzles according to the instructions given.
{prompt_skills}
----
{all_puzzle_str}
Could you please write {new_puzzles} new correct Python Programming Puzzles (from Puzzle {N_python_problem} to Puzzle {N_python_problem+new_puzzles-1})? Please, ensure the new puzzles must necessitates the utilization of the following skills (required skills {idx_skill_targeted}):
{skill_list_str}
{end_prompt}
'''
    return prompt



P3_IMPORTS = "from typing import List \n"#"from typing import List\n" # The only import that's necessary as of P3 v0.2


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
    return [1, 2]2

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


__all__ = [
    "P3_PROBLEM_MED_SEED",
    "P3_PROBLEM_LONG_SEED",
    "P3_PROBSOL_LONG_SEED",
    "P3_IMPORTS",
    "get_programming_puzzles_prompt",
    "skill_list",
    "create_prompt_label",
    "Puzzle_Interestingness",
    "Puzzle_Diversity","get_class_PuzzleCheck"
]
