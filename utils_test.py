import numpy as np
from pebble import ProcessPool

def test_puzzle(test_fg):
    test_fg= "from typing import *\n"+test_fg
    try:
        exec(test_fg)
        return True,test_fg
    except Exception as e:
        # print(str(e))
        # print("program not working: "+test_fg)
        return False,test_fg

def judge_parallel(src_codes, timeout=10., max_workers=10):

    max_workers = min(len(src_codes), max_workers)

    codes = src_codes
    successes = set()
    with ProcessPool(max_workers=max_workers) as pool:
        future = pool.map(test_puzzle, [code for code in codes], timeout=timeout)

        results = future.result()
        i = 0
        while True:
            try:
                success, code = next(results)
                if success:
                    successes.add(codes[i])
            except StopIteration:
                break
            except (TimeoutError, Exception) as error:
                pass
            assert i < len(codes)
            i += 1
        assert i == len(codes)
    # utils.silence_std_err(False)
    return [code in successes for code in src_codes]


prompt_solve_puzzle='''You will be given a function and its docstring. Respond only in code with a correct, efficient implementation of the function.
You need to generate the correct solutions (g), for the Problem 3 that satisfies the condition f(g()) == True.
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
Problem 3:
```
{pb}
```
Solution 3:
```
{g_firstline}
'''


def pass_at_k(n, c, k):
    """
    Adapted from "Evaluating Large Language Models Trained on Code" (https://arxiv.org/abs/2107.03374)

    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@k
    """
    assert n >= k
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))