import re
from typing import List, Set, Tuple, Dict
import multiprocessing as mp
from pebble import ProcessPool

from copy import deepcopy

import utils


def no_print(*_args, **_kwargs):
    pass


def run_judge(judge, f, tests):
    answer_type = list(judge.__annotations__.values())[0]
    for x in tests:
        y = f(**deepcopy(x))  # so f cannot cheat and change the input x
        if not utils.type_check(y, answer_type):
            raise TypeError
        assert judge(y, **x) is True, f"{f} failed on test {x}"


_ENV = dict(
    List=List,
    Set=Set,
    Tuple=Tuple,
    Dict=Dict,
    type_check=utils.type_check,
    run_judge=run_judge,
    test_puzzle=utils.test_puzzle,
    os=None,
    sys=None,
    input=None,
    open=None,
    print=no_print,
    compile=None,
    copyright=None,
)

_UNSAFE = ["builtin", "__class", "open("]
_SAFE_IMPORTS = {"collections", "copy", "hashlib", "math", "random", "re", "string", "typing"}

MAX_WORKERS = mp.cpu_count() // 2


def unsafe_imports(code):
    """Check if code imports any unsafe modules.

    Args:
        code (str): The code to check.

    Returns:
        bool: True if code imports unsafe modules.
    """
    if "import" not in code:
        return False
    for line in code.split("\n"):
        if "import" in line:
            match = re.search(r"^\s*from\s+([\w\.]+)\s+import\s", line)
            if match:
                modules = [match.group(1)]
            else:
                match = re.search(r"^\s*import\s+(.+)", line)
                if match:
                    modules = match.group(1).split(",")
                else:
                    return True
            if any(m.strip() not in _SAFE_IMPORTS for m in modules):
                return True
    return False


def _judge(code_env):
    code, env = code_env
    if unsafe_imports(code) or any(u in code for u in _UNSAFE):
        return False, Exception(f"unsafe code"), code
    try:
        exec(code, env.copy())
        return True, None, code
    except Exception as e:
        return False, e, code


def test_puzzle(code_str):
    if unsafe_imports(code_str) or any(u in code_str for u in _UNSAFE):
        return False, Exception(f"unsafe code"), code_str
    try:
        exec(code_str)
        return True, None, code_str
    except Exception as e:
        return False, e, code_str


def judge_parallel(src_codes, timeout=10, max_workers=MAX_WORKERS, env=_ENV):
    codes = utils.dedup(src_codes)
    utils.info(
        f"Judging {len(src_codes):,} codes ({len(src_codes)-len(codes):,} duplicates) with {max_workers} workers"
    )
    successes = set()

    # print("writing to file for debugging before judging")
    # from train import save_json
    #
    # save_json(new_codes, "results/tmp/new_codes.json")
    utils.silence_std_err(True)
    with ProcessPool(max_workers=max_workers) as pool:
        future = pool.map(_judge, [(code, env) for code in codes], timeout=timeout)

        results = future.result()
        i = 0
        while True:
            try:
                success, exc, code = next(results)
                if success:
                    successes.add(codes[i])
            except StopIteration:
                break
            except (TimeoutError, Exception) as error:
                pass
            assert i < len(codes)
            i += 1
        assert i == len(codes)
    utils.silence_std_err(False)
    return [code in successes for code in src_codes]
