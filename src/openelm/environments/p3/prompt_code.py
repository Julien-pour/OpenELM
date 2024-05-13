
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
{extra}

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
{extra}

## New 5 problems inspired by Puzzle 2:
"""

#run_8xA100
prompt_aces_V1= """Consider Python Programming Puzzles (P3). P3 consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in constructing a SAT problem `f` and a function `g` such that `f(g())` evaluates to `True`

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
{extra}

Please make sure that new puzzles have all the following skills{skill_target}
## New 5 problems inspired by Puzzle from Examples:
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
- Make sure that that each puzzle have just all required skills (see below)

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
{extra}

**Please make sure that new puzzles have JUST ALL the following skills**{skill_target}
## New 5 problems:
"""
#run_8xA100
prompt_aces_elm_v1= """Consider Python Programming Puzzles (P3). P3 consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in constructing a SAT problem `f` and a function `g` such that `f(g())` evaluates to `True`

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
{extra}

Please make sure that new puzzles have all the following skills{skill_target}
## New 5 problems inspired by Puzzle 2:
"""

prompt_aces_elm= """Consider Python Programming Puzzles (P3). P3 consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in constructing a SAT problem `f` and a function `g` such that `f(g())` evaluates to `True`

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
{extra}

**Please make sure that new puzzles have JUST ALL the following skills**{skill_target}
## New 5 problems inspired by Puzzle 2:
"""

instruction_solve_puzzle = '''You will be given a function. Respond only in code with a correct, efficient implementation of the function. You will need to generate the correct solutions (g), for the Problem 2 that satisfies the condition f(g()) == True.

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

Now you need to give the solution (def g({arg_g}):) to the following Problem 2 that satisfies the condition f(g()) == True.

Problem 2:
```python
{f}
```
'''

prompt_gen_subskills="""Consider Python Programming Puzzles (P3). P3 consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in constructing a SAT problem `f` and a function `g` such that `f(g())` evaluates to `True`. Keep in mind that the person who needs to find the solution will be able to look at the the problem (function `f`)

## Main Rules:

- Each puzzle includes two functions: `def f(...)` and `def g(...)`.
- The first argument of `f` is always the output from `g()`.
- Ensure `f` and `g` have matching argument signatures (e.g., `def f(solution, arg1=value1, arg2=value2, ...)` and `def g(arg1=value1, arg2=value2, ...)`). You also need to set the value of argument of f (arg1,arg2,...) and g when you define them.
- Avoid using `f` inside `g`, and `g` inside `f`.
- Include any necessary imports so your code runs smoothly.
- Give a clear Puzzle description that must be brief and diverse compared to the other puzzles.
- Make sure the puzzle is self-contained within these two functions.

## P3 Format:

Puzzle Description: A two to four-sentence summary of the puzzle's content. To explain what is the problem `f`, and how you can solve it with `g`.

```python
def f(solution, args=...) -> bool:
    # Python code to test the solution returned by g.
    # This function is a test unit and must return True if the solution is correct and False otherwise.

def g(args=...) -> solution:
    # Python code to generate a solution for the problem.
    # The solution should generalize to all possible args.
    return solution

assert f(g()) == True

```

Now I have a list of 20 following skills. Can you, for each skill, give me 20 diverse keywords or short ideas to help me create very **hard** Programming Puzzles:
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
Let's think step by step to help me create very **hard** Programming Puzzles that are still solvable in a few minutes and with less than 30-40 lines of code."""


def extract_subskills(filename):
    import re

    with open(filename, 'r') as file:
        data = file.read()

    # Use a regular expression to split by skill sections which start with '**'
    skills = re.split(r'\*\*(?=[^\*])', data)[1:]  # Skip the empty first split

    all_subskills = []

    for skill in skills:
        # Separate the title and the list of subskills using the first double newline
        sections = re.split(r'\n\n', skill, 1)
        if len(sections) < 2:
            continue  # Skip this loop iteration if there are not enough sections

        header, subskills = sections
        subskill_list = subskills.strip().split('\n')
        subskill_names = [subskill.split('. ', 1)[1] if '. ' in subskill else subskill for subskill in subskill_list]
        all_subskills.append(subskill_names)

    return all_subskills


list_subskills=[['Palindrome generator',
  'Anagram detection',
  'Caesar cipher decryption',
  'String compression',
  'Longest common substring finder',
  'Word chain builder',
  'Text justification',
  'URL shortener',
  'Password strength evaluator',
  'ROT13 encoder',
  'Sentence parsing',
  'Levenshtein distance calculator',
  'Grammar checker',
  'Character frequency analyzer',
  'Duplicate character remover',
  'Substring searcher',
  'Crossword puzzle filler',
  'Word ladder builder',
  'HTML entity encoder',
  'Regular expression matcher'],
 ['Prime number generator',
  'Fibonacci sequence calculator',
  "Euler's totient function calculator",
  'Modular exponentiation',
  'Quadratic equation solver',
  'Linear Diophantine equation solver',
  'Greatest common divisor calculator',
  'Least common multiple calculator',
  'Sine, cosine, and tangent calculator',
  'Cartesian coordinate system converter',
  'Geometric progression calculator',
  'Harmonic series calculator',
  'Arithmetico-geometric sequence calculator',
  "Pascal's triangle generator",
  'Binomial coefficient calculator',
  'Catalan number generator',
  'Euclidean norm calculator',
  'Matrix multiplication',
  'Vector dot product calculator',
  'Complex number arithmetic'],
 ['Truth table generator',
  'Boolean algebra simplifier',
  'Karnaugh map generator',
  'Digital circuit simulator',
  'Finite state machine builder',
  'Event-driven programming system',
  'Conditional probability calculator',
  "Bayes' theorem calculator",
  'Decision tree builder',
  'Flowchart generator',
  'Pseudocode interpreter',
  'Control flow graph analyzer',
  'Algorithm simulator',
  'Error handling system',
  'Exception handler',
  'Recursion detector',
  'Loop invariant finder',
  'Branch predictor',
  'Code analyzer',
  'Logical implication validator'],
 ['Fibonacci sequence calculator',
  'Factorial calculator',
  'Binary tree traversal',
  'N-queens problem solver',
  'Tower of Hanoi solver',
  'Permutation generator',
  'Combination generator',
  'Recursive downhill simplex method',
  'Fractal generator',
  'Recursively enumerable language recognizer',
  'Recursive function memoizer',
  'Functional programming emulator',
  'Recursive parser',
  'Dynamic programming problem solver',
  'Memoization cache manager',
  'Recursive descent parser',
  'Tree traverser',
  'Graph traverser',
  'RecursiveNN parser',
  'Gödel numbering system'],
 ['Exhaustive search algorithm',
  'Traveling salesman problem solver',
  'Knapsack problem solver',
  'Satisfiability problem (SAT) solver',
  'N-queens problem solver',
  'Brute force password cracker',
  'Cryptanalysis tool',
  'Frequency analysis tool',
  'Rainbow table generator',
  'Substitution cipher breaker',
  'Transposition cipher breaker',
  'Vigenère cipher breaker',
  'Caesar cipher breaker',
  'Hill cipher breaker',
  'Frequency hopping sequence generator',
  'Subsequence finder',
  'Pattern matching algorithm',
  'Brute force DES cracker',
  'Brute force AES cracker',
  'Password hash cracker'],
 ['Fibonacci sequence calculator',
  'Longest common subsequence finder',
  'Shortest path problem solver',
  'Knapsack problem solver',
  'Scheduling problem solver',
  'Activity selection problem solver',
  '0/1 knapsack problem solver',
  'Unbounded knapsack problem solver',
  'Matrix chain multiplication',
  'Optimal binary search tree',
  'Dynamic Huffman coding',
  'Edit distance calculator',
  'Levenshtein distance calculator',
  'Longest increasing subsequence finder',
  'Maximum subarray problem solver',
  'Minimum window that contains all elements',
  'Schedules with deadlines',
  'Coin changing problem solver',
  'Cutting stock problem solver',
  'Assembly line scheduling problem'],
 ['Coin changing problem solver',
  'Huffman coding',
  'Activity selection problem solver',
  'Scheduling problem solver',
  'Knapsack problem solver',
  'Set cover problem solver',
  'Vertex cover problem solver',
  'Interval scheduling problem solver',
  'Matroid intersection problem solver',
  'Greedy algorithm for subset sum',
  'Optimal caching',
  'Optimal storage allocation',
  'Graph coloring problem solver',
  'Edge coloring problem solver',
  'Interval graph coloring problem solver',
  'Chromatic number calculator',
  'Minimum spanning tree',
  "Prim's algorithm",
  "Kruskal's algorithm",
  "Boruvka's algorithm"],
 ['N-queens problem solver',
  'Sudoku solver',
  'Crossword puzzle solver',
  'Word search solver',
  'Hamiltonian cycle problem solver',
  'Traveling salesman problem solver',
  'Knapsack problem solver',
  'Satisfiability problem (SAT) solver',
  'Boolean satisfiability problem (SAT) solver',
  'Graph coloring problem solver',
  'Edge coloring problem solver',
  'Interval graph coloring problem solver',
  'Chromatic number calculator',
  'TSP with neighborhoods',
  'Job shop scheduling problem solver',
  'Flow shop scheduling problem solver',
  'Open shop scheduling problem solver',
  'Vehicle routing problem solver',
  'Capacitated vehicle routing problem solver',
  'Bin packing problem solver'],
 ['Union, intersection, and difference calculator',
  'Set complement calculator',
  'Cartesian product calculator',
  'Power set generator',
  'Subset generator',
  'Superset generator',
  'Set partition generator',
  'Set packing problem solver',
  'Set covering problem solver',
  'Hitting set problem solver',
  'Steiner tree problem solver',
  'Feedback vertex set problem solver',
  'Independent set problem solver',
  'Clique problem solver',
  'Dominating set problem solver',
  'Total dominating set problem solver',
  'Connected dominating set problem solver',
  'Set clustering problem solver',
  'Set classification problem solver',
  'Set regression problem solver'],
 ['Permutation generator',
  'Combination generator',
  'Multinomial coefficient calculator',
  'Binomial coefficient calculator',
  'Catalan number generator',
  'Fibonacci number generator',
  "Pascal's triangle generator",
  'Lottery number generator',
  'Shuffling algorithm',
  'Random permutation generator',
  'Random combination generator',
  'Permutation tester',
  'Combination tester',
  'Multiset generator',
  'Multiset permutation generator',
  'Multiset combination generator',
  'Partition generator',
  'Integer partition generator',
  'Young diagram generator',
  'Permutation statistics calculator'],
 ['Random number generator',
  'Coin flip simulator',
  'Dice roll simulator',
  'Random variable generator',
  'Probability density function calculator',
  'Cumulative distribution function calculator',
  'Inverse cumulative distribution function calculator',
  'Confidence interval calculator',
  'Hypothesis testing tool',
  'Statistical significance calculator',
  'Mean, median, and mode calculator',
  'Standard deviation calculator',
  'Variance calculator',
  'Correlation coefficient calculator',
  'Regression analysis tool',
  'Time series analysis tool',
  'Signal processing tool',
  'Filtering tool',
  'Prediction tool',
  'Reinforcement learning tool'],
 ['Regular expression matcher',
  'String matching algorithm',
  'Pattern recognition algorithm',
  'Image recognition tool',
  'Speech recognition tool',
  'Natural language processing tool',
  'Text classification tool',
  'Sentiment analysis tool',
  'Entity recognition tool',
  'Feature extraction tool',
  'Clustering algorithm',
  'Decision tree algorithm',
  'Random forest algorithm',
  'Support vector machine algorithm',
  'K-nearest neighbors algorithm',
  'Naive Bayes algorithm',
  'Gaussian mixture model algorithm',
  'Hidden Markov model algorithm',
  'Conditional random field algorithm',
  'Convolutional neural network algorithm'],
 ['Bubble sort algorithm',
  'Selection sort algorithm',
  'Insertion sort algorithm',
  'Merge sort algorithm',
  'Quick sort algorithm',
  'Heap sort algorithm',
  'Radix sort algorithm',
  'Timsort algorithm',
  'Dual pivot quick sort algorithm',
  'Concurrent sorting algorithm',
  'External sorting algorithm',
  'Adaptive sorting algorithm',
  'Stable sorting algorithm',
  'Sorting network algorithm',
  'Permutation sorting algorithm',
  'Topological sorting algorithm',
  'Partial order sorting algorithm',
  'Total order sorting algorithm',
  'Weak ordering sorting algorithm',
  'Strong ordering sorting algorithm'],
 ['Bitwise AND operator',
  'Bitwise OR operator',
  'Bitwise XOR operator',
  'Bitwise NOT operator',
  'Left shift operator',
  'Right shift operator',
  'Circular shift operator',
  'Rotate left operator',
  'Rotate right operator',
  'Bit manipulation algorithm',
  'Bit reversal algorithm',
  'Bit extraction algorithm',
  'Bit insertion algorithm',
  'Bit counting algorithm',
  'Hamming weight calculator',
  'Hamming distance calculator',
  'Binary search algorithm',
  'Binary tree traverser',
  'Binary heap traverser',
  'Binary search tree traverser'],
 ['Point class implementation',
  'Vector class implementation',
  'Matrix class implementation',
  'Coordinate system converter',
  'Geometric transformation tool',
  'Rotation matrix generator',
  'Translation matrix generator',
  'Scaling matrix generator',
  'Projection matrix generator',
  '2D and 3D geometry libraries',
  'Trigonometry library',
  'Analytic geometry library',
  'Synthetic geometry library',
  'Computational geometry library',
  'Geometric algorithms library',
  'Voronoi diagram generator',
  'Delaunay triangulation generator',
  'Convex hull algorithm',
  'Point in polygon tester',
  'Line clipping algorithm'],
 ['Time complexity analyzer',
  'Space complexity analyzer',
  'Algorithm profiler',
  'Cache optimizer',
  'Memoization optimizer',
  'Dynamic programming optimizer',
  'Greedy algorithm optimizer',
  'Divide and conquer optimizer',
  'Backtracking optimizer',
  'Branch and bound optimizer',
  'Cutting plane method optimizer',
  'Column generation optimizer',
  'Benders decomposition optimizer',
  'Lagrangian relaxation optimizer',
  'Semidefinite programming optimizer',
  'Quadratic programming optimizer',
  'Linear programming optimizer',
  'Integer programming optimizer',
  'Mixed-integer programming optimizer',
  'Stochastic optimization optimizer'],
 ['Prime number generator',
  'Factorization algorithm',
  'Greatest common divisor calculator',
  'Least common multiple calculator',
  "Euler's totient function calculator",
  'Carmichael number generator',
  'Prime number theorem calculator',
  'Riemann hypothesis calculator',
  'Modular arithmetic calculator',
  'Diophantine equation solver',
  'Linear Diophantine equation solver',
  "Pell's equation solver",
  'Pythagorean triple generator',
  'Perfect number generator',
  'Amicable number generator',
  'Abundant number generator',
  'Deficient number generator',
  'Weird number generator',
  'Frugal number generator',
  'Coprime number generator'],
 ['Graph class implementation',
  'Vertex class implementation',
  'Edge class implementation',
  'Path class implementation',
  'Cycle detection algorithm',
  'Shortest path algorithm',
  'Minimum spanning tree algorithm',
  'Maximum flow algorithm',
  'Topological sorting algorithm',
  'Strongly connected component algorithm',
  'Weakly connected component algorithm',
  'Graph traversal algorithm',
  'Breadth-first search algorithm',
  'Depth-first search algorithm',
  "Dijkstra's algorithm",
  'Bellman-Ford algorithm',
  'Floyd-Warshall algorithm',
  "Johnson's algorithm",
  "Prim's algorithm",
  "Kruskal's algorithm"],
 ['1D array indexing algorithm',
  '2D array indexing algorithm',
  '3D array indexing algorithm',
  'Jagged array indexing algorithm',
  'Sparse array indexing algorithm',
  'Array slicing algorithm',
  'Array concatenation algorithm',
  'Array partitioning algorithm',
  'Array permutation algorithm',
  'Array combination algorithm',
  'Array sorting algorithm',
  'Array searching algorithm',
  'Array filtering algorithm',
  'Array mapping algorithm',
  'Array reducing algorithm',
  'Array aggregating algorithm',
  'Array grouping algorithm',
  'Array indexing optimization algorithm',
  'Array caching algorithm',
  'Array parallelization algorithm'],
 ['Hash function generator',
  'Collision detection algorithm',
  'Hash table implementation',
  'Bloom filter implementation',
  'Cuckoo hash table implementation',
  'Hopscotch hash table implementation',
  'FNV-1a hash function implementation',
  'MurmurHash3 hash function implementation',
  'SHA-256 hash function implementation',
  'MD5 hash function implementation',
  'CRC32 hash function implementation',
  'Adler-32 hash function implementation',
  'Perfect hash function implementation',
  'Minimal perfect hash function implementation',
  'CHD (Compact Hashed Dictionary) implementation',
  'Hash table resizing algorithm',
  'Hash table caching algorithm',
  'Hash table parallelization algorithm',
  'Hash table compression algorithm',
  'Hash table encryption algorithm']]