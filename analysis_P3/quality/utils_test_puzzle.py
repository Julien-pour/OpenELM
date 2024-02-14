bad_puzzles =["""```python
def f(is_sorted: bool, arr: List[int]) -> bool:
    return is_sorted == all((arr[i] <= arr[i + 1] for i in range(len(arr) - 1)))

def g(arr: List[int]) -> bool:
    return all((arr[i] <= arr[i + 1] for i in range(len(arr) - 1)))
```""",
"""```python
def f(is_anagram: bool, word1: str='listen', word2: str='silent') -> bool:
    return is_anagram

def g(word1: str='listen', word2: str='silent') -> bool:
    return sorted(word1.lower()) == sorted(word2.lower())
```"""]

trivial_puzzles = ["""```python
def f(st: str, a="world", b="Hello world"):
    return st + a == b

def g(a="world", b="Hello world"):
    return b[:len(b) - len(a)]
```""",
"""```python
def f(st: str, a="s", b="tos"):
    return st + a == b

def g(a="s", b="tos"):
    return b[:len(b) - len(a)]
```"""]

appropriate_puzzles= ["""```python
def f(s: str):
    return s.count('o') == 1000 and s.count('oo') == 0

def g():
    return ('h' + 'o') * 1000
```""",
"""```python
def f(li: List[int]):
    return all([li.count(i) == i for i in range(10)])

def g():
    return [i for i in range(10) for j in range(i)]
```""",]
hard_puzzles = ["""```python
def f(nums: List[int], b=7, m=6):

    assert len(nums) == len(set(nums)) == m and min(nums) >= 0

    def gcd(i, j):
        r, s = max(i, j), min(i, j)
        while s >= 1:
            r, s = s, (r % s)
        return r

    for a in nums:
        nums = [(a + i + 1) ** 2 + (a + i + 1) + 1 for i in range(b)]
        assert all(any(i != j and gcd(i, j) > 1 for j in nums) for i in nums)

    return True

def g(b=7, m=6):
    ans = []

    seen = set()
    deltas = set()

    def go(a):
        if a < 0 or a in seen or len(ans) == m:
            return
        seen.add(a)
        nums = [(a + i + 1) ** 2 + (a + i + 1) + 1 for i in range(b)]
        if all(any(i != j and gcd(i, j) > 1 for j in nums) for i in nums):
            new_deltas = [abs(a - a2) for a2 in ans if a != a2 and abs(a - a2) not in deltas]
            ans.append(a)
            for delta in new_deltas:
                for a2 in ans:
                    go(a2 + delta)
                    go(a2 - delta)
            deltas.update(new_deltas)
            for delta in sorted(deltas):
                go(a + delta)

    def gcd(i, j):
        r, s = max(i, j), min(i, j)
        while s >= 1:
            r, s = s, (r % s)
        return r

    a = 0

    while len(ans) < m:
        go(a)
        a += 1

    return ans
""",
"""```python
def f(moves: List[List[int]], initial_state=[5, 8, 3, 0]):

    def bot_move():  # bot takes objects from the largest heap to make it match the second largest heap
        vals = sorted(state, reverse=True)
        i_largest = state.index(vals[0])  # largest heap
        state[i_largest] -= max(vals[0] - vals[1], 1)  # must take some, take 1 in case of tie

    state = initial_state[:]  # copy
    for i, n in moves:
        assert 0 < n <= state[i], "Illegal move"
        state[i] -= n
        if set(state) == {0}:
            return True  # you won!
        assert any(state), "You lost!"
        bot_move()

def g(initial_state=[5, 8, 3, 0]):

    state = initial_state[:]
    moves = []

    def bot_move():  # bot takes objects from the largest heap to make it match the second largest heap
        vals = sorted(state, reverse=True)
        i_largest = state.index(vals[0])  # largest heap
        state[i_largest] -= max(vals[0] - vals[1], 1)  # must take some, take 1 in case of tie

    def losing(h):
"""]

puzzle_dict = {}
puzzles = [bad_puzzles, trivial_puzzles, appropriate_puzzles, hard_puzzles]
keys = ["bad_puzzles", "trivial_puzzles", "appropriate_puzzles", "hard_puzzles"]

for i, puzzle_list in enumerate(puzzles):
    for j, puzzle in enumerate(puzzle_list):
        key = f"{keys[i]}:{j}"
        puzzle_dict[key] = puzzle

base_persona ="You are a helpful assistant to a Professor teaching a programming course in Python. "
base_persona += "The Professor want to give some puzzles to his master's student to teach them Python." # student -> Master student
prompt_instruction = base_persona 
prompt_instruction += """I already have a series of Python Programming Puzzles (P3) where each puzzle consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in constructing `g` such that `f(g())` evaluates to `True`.


Rules:
- Each puzzle includes two functions: `def f(...)` and `def g(...)`.
- The first argument of `f` is always the output from `g()`.
- Ensure `f` and `g` have matching argument signatures (e.g., `def f(arg0, arg1=value1, arg2=value2, ...)` and `def g(arg1=value1, arg2=value2, ...)`).
- Avoid using `f` inside `g`, and `g` inside `f`.
- Include any necessary imports for your code to run smoothly.
- Give a clear Puzzle description that must be brief and diverse compared to the other puzzles.
- Make sure the puzzle is self-contained within these two functions.
- f and g should be distinct (not copy-paste of each other)

Puzzle Format:
Puzzle description: A brief, one to two sentence summary of the puzzle's content.
```python
def f(solution, args=...) -> bool:
    # Python code to test the solution returned by g.
    # This function is a test unit and must return True if the solution is correct, False otherwise.

def g(args=...) -> solution:
    # Python code to generate a solution for the problem.
    # The solution should generalize to all possible args.
    return solution

assert f(g()) == True
```
Your Task:
Create a new Python Programming Puzzle."""