five_fold_ranking_prompt = """You are a helpful assistant to a teacher teaching a second-year undergraduate programming course in Python. The teacher is proposing 5 exercises in the form of programming puzzles, and your role is to rank these puzzles from the most appropriate for the course to the least appropriate.

A Python programming puzzle is defined by two functions, the *puzzle* f(...) and the *solution* g(...). f defines an algorithmic challenge, and g solves this challenge. g is a solution to f if and only if `f(g()) == True`. There can be additional optional keyword arguments to both f and g; these arguments, if they exist, are the same for both functions.

Format of the description, puzzle and solution:
```python
Problem description: ...

def f(solution, args=...) -> bool:
    # Python code to test the solution returned by g.
    # This function behaves like a unit test and must return True if the solution is correct, False otherwise.

def g(args=...) -> solution:
    # Python code to generate a solution for the problem.
    # The solution should generalize to all possible args.
    return solution
```

You are ranking puzzles for a second year undergraduate Python course. Puzzles and their solutions should be ranked highly if they provide important opportunities for learning for students of that level (both by searching for the solution and looking at the provided solution, so you should look at the quality of both f and g). Clarity is important. You will be presented with 5 puzzles and their solutions, numbered from 0 to 4, and you should respond by giving your ranking as a list of indices, from the most relevant to least relevant, like so: Answer: [...]

{examples}

Answer:"""

random_permutation_prompt = """Can you give me a random permutation of the list [0, 1, 2, 3, 4] please? Format this as a list of 5 elements from 0 to 4.

Answer:
"""

five_fold_ranking_prompt_scrambled = """You are a helpful assistant to a teacher teaching a second-year undergraduate programming course in Python. The teacher is proposing 5 exercises in the form of programming puzzles, and your role is to rank these puzzles from the most appropriate for the course to the least appropriate.

A Python programming puzzle is defined by two functions, the *puzzle* f(...) and the *solution* g(...). f defines an algorithmic challenge, and g solves this challenge. g is a solution to f if and only if `f(g()) == True`. There can be additional optional keyword arguments to both f and g; these arguments, if they exist, are the same for both functions.

Format of the description, puzzle and solution:
```python
Problem description: ...

def f(solution, args=...) -> bool:
    # Python code to test the solution returned by g.
    # This function behaves like a unit test and must return True if the solution is correct, False otherwise.

def g(args=...) -> solution:
    # Python code to generate a solution for the problem.
    # The solution should generalize to all possible args.
    return solution
```

You are ranking puzzles for a second year undergraduate Python course. Puzzles and their solutions should be ranked highly if they provide important opportunities for learning for students of that level (both by searching for the solution and looking at the provided solution, so you should look at the quality of both f and g). Clarity is important. You will be presented with 5 puzzles and their solutions, with random 4-letter ids. You should respond by giving your ranking as a list of ids, from the most relevant to least relevant. The ids should be inside double quotes, like so: Answer: ["aaaa", "bbbb", ...]

{examples}

Answer:"""


five_fold_ranking_prompt_scrambled = """You are a helpful assistant to a teacher teaching a second-year undergraduate programming course in Python. The teacher is proposing 5 exercises in the form of programming puzzles, and your role is to rank these puzzles from the most appropriate for the course to the least appropriate.

A Python programming puzzle is defined by two functions, the *puzzle* f(...) and the *solution* g(...). f defines an algorithmic challenge, and g solves this challenge. g is a solution to f if and only if `f(g()) == True`. There can be additional optional keyword arguments to both f and g; these arguments, if they exist, are the same for both functions.

Format of the description, puzzle and solution:
```python
Problem description: ...

def f(solution, args=...) -> bool:
    # Python code to test the solution returned by g.
    # This function behaves like a unit test and must return True if the solution is correct, False otherwise.

def g(args=...) -> solution:
    # Python code to generate a solution for the problem.
    # The solution should generalize to all possible args.
    return solution
```

You are ranking puzzles for a second year undergraduate Python course. Puzzles and their solutions should be ranked highly if they provide important opportunities for learning for students of that level (both by searching for the solution and looking at the provided solution, so you should look at the quality of both f and g). Clarity is important. You will be presented with 5 puzzles and their solutions, with random 4-letter ids. You should respond by giving your ranking as a list of ids, from the most relevant to least relevant. The ids should be inside double quotes, like so: Answer: ["aaaa", "bbbb", ...]

{examples}

Answer:"""


five_fold_ranking_prompt_scrambled_cot = """I am teaching a second-year undergraduate programming course in Python. I am proposing 5 exercises in the form of programming puzzles, and your role is to rank these puzzles from the most appropriate for the course to the least appropriate.

A Python programming puzzle is defined by two functions, the *puzzle* f(...) and the *solution* g(...). f defines an algorithmic challenge, and g solves this challenge. g is a solution to f if and only if `f(g()) == True`. There can be additional optional keyword arguments to both f and g; these arguments, if they exist, are the same for both functions.

Format of the description, puzzle and solution:
```python
Problem description: ...

def f(solution, args=...) -> bool:
    # Python code to test the solution returned by g.
    # This function behaves like a unit test and must return True if the solution is correct, False otherwise.

def g(args=...) -> solution:
    # Python code to generate a solution for the problem.
    # The solution should generalize to all possible args.
    return solution
```

You are ranking puzzles for a second year undergraduate Python course. The evaluation criteria for good puzzles are:

* Clarity (how easy is it to understand what needs to be done?);
* Opportunity for learning (how likely is it that a student will learn by trying to solve the puzzle and looking at its solution?);
* Difficulty appropriate for the course (is the puzzle too easy. tooo hard, or just right?);
* Engagement (is the puzzle fun or boring?);

You will be presented with 5 puzzles and their solutions, with random 4-letter ids. You should respond by giving your ranking as a list of ids, from the most relevant to least relevant. The ids should be inside double quotes. You should provide a justification for your ranking, and then the ranking.

Example: if the puzzle ids are "aaaa", "bbbb", "cccc". "dddd" and "eeee" you should respond like so: Reasoning: ... 
Answer: ["cccc", "bbbb", ...]

The puzzles are:

{examples}

Reasoning:
"""


two_fold_ranking_prompt_scrambled_cot = """I am teaching a second-year undergraduate programming course in Python. I am proposing 2 exercises in the form of programming puzzles, and your role is to tell me which of these puzzles is the most appropriate for the course.

A Python programming puzzle is defined by two functions, the *puzzle* f(...) and the *solution* g(...). f defines an algorithmic challenge, and g solves this challenge. g is a solution to f if and only if `f(g()) == True`. There can be additional optional keyword arguments to both f and g; these arguments, if they exist, are the same for both functions.

Format of the description, puzzle and solution:
```python
Problem description: ...

def f(solution, args=...) -> bool:
    # Python code to test the solution returned by g.
    # This function behaves like a unit test and must return True if the solution is correct, False otherwise.

def g(args=...) -> solution:
    # Python code to generate a solution for the problem.
    # The solution should generalize to all possible args.
    return solution
```

You are ranking puzzles for a second year undergraduate Python course. The evaluation criteria for good puzzles are:

* Clarity (how easy is it to understand what needs to be done?);
* Opportunity for learning (how likely is it that a student will learn by trying to solve the puzzle and looking at its solution?);
* Difficulty appropriate for the course (is the puzzle too easy. tooo hard, or just right?);
* Engagement (is the puzzle fun or boring?);

You will be presented with 2 puzzles and their solutions, with random 4-letter ids. You should respond by telling me which of the puzzle is best, by giving me a list of the 2 ids of the puzzles ranked by appropriateness inside double quotes. You should provide a justification for your ranking (where is says "Reasoning"), and then the list of ids (where it says "Answer").

Example: if the puzzle ids are "aaaa" and "bbbb", and you prefer puzzle "bbbb" you should respond like so: 
Reasoning: ... 
Answer: ["bbbb", "aaaa"]
 
The puzzles are:

{examples}

Reasoning:
"""


two_fold_ranking_prompt_scrambled_cot_2 = """I am teaching a first-year undergraduate programming course in Python. I am proposing 2 exercises in the form of programming puzzles, and your role is to tell me which of these puzzles is the most appropriate for the course.

A Python programming puzzle is defined by two functions, the *puzzle* f(...) and the *solution* g(...). f defines an algorithmic challenge, and g solves this challenge. g is a solution to f if and only if `f(g()) == True`. There can be additional optional keyword arguments to both f and g; these arguments, if they exist, are the same for both functions.

Format of the description, puzzle and solution:
```python
Problem description: ...

def f(solution, args=...) -> bool:
    # Python code to test the solution returned by g.
    # This function behaves like a unit test and must return True if the solution is correct, False otherwise.

def g(args=...) -> solution:
    # Python code to generate a solution for the problem.
    # The solution should generalize to all possible args.
    return solution
```

You are ranking puzzles for a first year undergraduate Python course and you should tell me which one is best. The evaluation criterion for good puzzles is their difficulty: they should be easy enough for first-year students. You will think of what is needed to solve the problem (what is needed to write g given f), estimate how hard that is, and judge whether that is appropriate for a first-year student.

You will be presented with 2 puzzles and their solutions, with random 4-letter ids. You should respond by telling me which of the puzzle is best, by giving me a list of the 2 ids of the puzzles ranked by appropriateness inside double quotes. You should provide a justification for your ranking (where is says "Reasoning"), and then the list of ids (where it says "Answer").

Example: if the puzzle ids are "aaaa" and "bbbb", and you prefer puzzle "bbbb" you should respond like so: 
Reasoning: ... 
Answer: ["bbbb", "aaaa"]
 
The puzzles are:

{examples}

Reasoning:
"""


two_fold_ranking_prompt_scrambled_cot_2 = """I am teaching a first-year undergraduate programming course in Python. I am proposing 2 exercises in the form of programming puzzles, and your role is to tell me which of these puzzles is the most appropriate for the course.

A Python programming puzzle is defined by two functions, the *puzzle* f(...) and the *solution* g(...). f defines an algorithmic challenge, and g solves this challenge. g is a solution to f if and only if `f(g()) == True`. There can be additional optional keyword arguments to both f and g; these arguments, if they exist, are the same for both functions.

Format of the description, puzzle and solution:
```python
Problem description: ...

def f(solution, args=...) -> bool:
    # Python code to test the solution returned by g.
    # This function behaves like a unit test and must return True if the solution is correct, False otherwise.

def g(args=...) -> solution:
    # Python code to generate a solution for the problem.
    # The solution should generalize to all possible args.
    return solution
```

You are ranking puzzles for a first year undergraduate Python course and you should tell me which one is best. The evaluation criterion for good puzzles is their difficulty: they should be easy enough for first-year students. You will think of what is needed to solve the problem (what is needed to write g given f), estimate how hard that is, and judge whether that is appropriate for a first-year student.

You will be presented with 2 puzzles and their solutions, with random 4-letter ids. You should respond by telling me which of the puzzle is best, by giving me a list of the 2 ids of the puzzles ranked by appropriateness inside double quotes. You should provide a justification for your ranking (where is says "Reasoning"), and then the list of ids (where it says "Answer").

Example: if the puzzle ids are "aaaa" and "bbbb", and you prefer puzzle "bbbb" you should respond like so: 
Reasoning: ... 
Answer: ["bbbb", "aaaa"]
 
The puzzles are:

{examples}

Reasoning:
"""


two_fold_ranking_prompt_scrambled_cot_2_easiest = """I am teaching a first-year undergraduate programming course in Python. I am proposing 2 exercises in the form of programming puzzles, and your role is to tell me which of these puzzles is the most appropriate for the course.

A Python programming puzzle is defined by two functions, the *puzzle* f(...) and the *solution* g(...). f defines an algorithmic challenge, and g solves this challenge. g is a solution to f if and only if `f(g()) == True`. There can be additional optional keyword arguments to both f and g; these arguments, if they exist, are the same for both functions.

Format of the description, puzzle and solution:
```python
Problem description: ...

def f(solution, args=...) -> bool:
    # Python code to test the solution returned by g.
    # This function behaves like a unit test and must return True if the solution is correct, False otherwise.

def g(args=...) -> solution:
    # Python code to generate a solution for the problem.
    # The solution should generalize to all possible args.
    return solution
```

You are ranking puzzles based on how easy they are. The easiest puzzle should come first. You will think of what is needed to solve the problem (what is needed to write g given f), estimate how hard that is, and use it to justify your answer.

You will be presented with 2 puzzles and their solutions, with random 4-letter ids. You should respond by telling me which of the puzzle is easiest, by giving me a list of the 2 ids of the puzzles ranked by easiness inside double quotes. You should provide a justification for your ranking (where is says "Reasoning"), and then the list of ids (where it says "Answer").

Example: if the puzzle ids are "aaaa" and "bbbb", and you prefer puzzle "bbbb" you should respond like so: 
Reasoning: ... 
Answer: ["bbbb", "aaaa"]
 
The puzzles are:

{examples}

Reasoning:
"""