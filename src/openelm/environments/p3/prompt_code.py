
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
The Professor want to give Pyhton programming puzzles to his {level} to teach them Python.
A Python programming puzzle is defined by two functions, the puzzle f(…) and the solution g(…). f defines an algorithmic challenge, and g solves this challenge. g is a solution to f if and only if f(g()) == True."""
