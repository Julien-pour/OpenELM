from typing import List

def f(s: str):
    """Find a string that when concatenated onto 'Hello ' gives 'Hello world'."""
    return "Hello " + s == "Hello world"

def g():
    return "world"

assert f(g())

def f(s: str):
    """Find a string that when reversed and concatenated onto 'Hello ' gives 'Hello world'."""
    return "Hello " + s[::-1] == "Hello world"

def g():
    return "world"[::-1]

assert f(g())

def f(x: List[int]):
    """Find a list of two integers whose sum is 3."""
    return len(x) == 2 and sum(x) == 3

def g():
    return [1, 2]

assert f(g())

def f(s: List[str]):
    """Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b'."""
    return len(set(s)) == 1000 and all((x.count("a") > x.count("b")) and ('b' in x)
        for x in s)

def g():
    return ["a"*(i+2)+"b" for i in range(1000)]

assert f(g())

def f(n: int):
    """Find an integer whose perfect square begins with 123456789 in its decimal representation."""
    return str(n * n).startswith("123456789")

def g():
    return int(int("123456789" + "0"*9) ** 0.5) + 1

assert f(g())

{pb}