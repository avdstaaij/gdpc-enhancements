from typing import TypeVar
from contextlib import contextmanager
from copy import deepcopy
import sys
import numpy as np


T = TypeVar("T")


def isign(x: int):
    """ Note that isign(0) == 1 """
    return 1 if x >= 0 else -1


def filledList(n: int, element):
    return list(deepcopy(element) for _ in range(n))


# https://stackoverflow.com/a/14981125
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# Based on https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
@contextmanager
def stdoutToStderr():
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


def clamp(x: T, minimum: T, maximum: T) -> T:
    return max(minimum, min(maximum, x))


# Based on https://stackoverflow.com/a/21032099
def normalized(a, order=2, axis=-1):
    norm = np.atleast_1d(np.linalg.norm(a, order, axis))
    norm[norm==0] = 1
    return a / np.expand_dims(norm, axis)
