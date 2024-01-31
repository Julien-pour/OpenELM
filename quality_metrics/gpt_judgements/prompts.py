five_fold_ranking_prompt = """You are a helpful assistant to a teacher teaching an unergraduate programming course in Python. The teacher is proposing 5 exercises in the form of programming puzzles, and your role is to rank these puzzles from the most appropriate for the course to the least appropriate.

A Python programming puzzle is defined by two functions, the *puzzle* f(...) and the *solution* g(...). f defines an algorithmic challenge, and g solves this challenge. g is a solution to f if and only if `f(g()) == True`. There can be additional optional keyword arguments to both f and g; these arguments if they exist are the same for both functions. {optional: There is a description of the operations that should be performed by the solution in the description of the puzzle.}

Puzzle Format:
Problem description: ...
```python
def f(solution, args=...) -> bool:
    # Python code to test the solution returned by g.
    # This function is a test unit and must return True if the solution is correct, False otherwise.

def g(args=...) -> solution:
    # Python code to generate a solution for the problem.
    # The solution should generalize to all possible args.
    return solution

assert f(g()) == True

You are classifying puzzles for a second year undergraduate Python course. Puzzles and their solutions should be ranked highly if they provide important opportunities for learning for students of that level (both by searching for the solution and looking at the provided solution, so you should look at the quality of both f and g). Clarity is important. You will be presented with 5 puzzles and their solutions, numbered from 0 to 4, and you should respond by giving your ranking as a list of indices, from the most relevant to least relevant, like so: Answer: [...]

{examples}

Answer:"""