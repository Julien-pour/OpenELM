def f(stamps: List[int], target=80, max_stamps=4, options=[10, 32, 8]) -> bool:
    """Find a selection of at most max_stamps stamps whose total worth is the target value."""
    for s in stamps:
        assert s in options
    return len(stamps) <= max_stamps and sum(stamps) == target

def g(target=80, max_stamps=4, options=[10, 32, 8]):
    from itertools import combinations_with_replacement
    for n in range(max_stamps + 1):
        for c in combinations_with_replacement(options, n):
            if sum(c) == target:
                return list(c)

assert f(g())

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

assert f(g())

def f(n: int, v=313946483, w=806690290) -> bool:
    """Find the smallest n such that if v is tripled n times and w is doubled n times, v exceeds w."""
    for i in range(n):
        assert v <= w
        v *= 3
        w *= 2
    return v > w

def g(v=313946483, w=806690290):
    i = 0
    while v <= w:
        v *= 3
        w *= 2
        i += 1
    return i

assert f(g())

{pb}