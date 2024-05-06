
prompt_gen_description="""A Python programming puzzle is defined by two functions, the problem f(solution, arg1=value1, arg2=value2, ..) and the solution. f defines an algorithmic puzzle, and the solution solves this puzzle.
You should pay a particular attention that the puzzle is solved if and only if **f(solution) == True**.
Your role is to write a one or two sentence the description of the puzzle's goal (what the solution should be), remember that the solution that satisfy the goal must be given as the first argument of `f`.
You can start by: 'Find the solution: {arg_sol} (describe its type shortly) that should (here you should speak about the solution: {arg_solb} and how it should solve all the constraints of the puzzle with respect to others args (describe their types shortly)) ...'. 
For example:
'Given a string `str1`, find the length of the longest substring without repeating characters.'
'Given two sorted arrays `nums1` and `nums2` of size `m` and `n` respectively, return the median of the two sorted arrays.'


The puzzle is:
```python
{puzzle}
```
"""

base_persona_code ="""You are a helpful assistant to a Professor teaching a programming course in Python. 
The Professor want to give Pyhton programming puzzles to his Computer Science student to teach them Python.
A Python programming puzzle is defined by two functions, the puzzle f(…) and the solution g(…). f defines an algorithmic challenge, and g solves this challenge. g is a solution to f if and only if f(g()) == True."""


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

Generate 5 P3 similar to the last Examples (Puzzle 2). Ensure that all new puzzles are more challenging than Puzzle from previous examples.
Please make sure that new puzzles test the all the following skills:{skill_target}
## New 5 problems inspired by Puzzle 2:
"""