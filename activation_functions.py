import math
from functools import cache, lru_cache
from typing import Callable

ActivationFunction = Callable[[float], float]

_activation_functions: dict[str: ActivationFunction] = dict()


def get_activation_function(name: str) -> ActivationFunction:
    if name in _activation_functions:
        return _activation_functions[name]
    else:
        raise ValueError(f'Activation function {name} not found')


def activation_function(func):
    if func.__name__ not in _activation_functions:
        _activation_functions[func.__name__] = func
        return func
    raise ValueError(f'Activation function {func.__name__} already exists')


@activation_function
@lru_cache(maxsize=1000)
def relu(x):
    return max(0, x)


@activation_function
@lru_cache(maxsize=1000)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


@activation_function
@lru_cache(maxsize=1000)
def tanh(x):
    return math.tanh(x)


@activation_function
@lru_cache(maxsize=1000)
def identity(x):
    return x


@activation_function
@lru_cache(maxsize=1000)
def binary_step(x):
    return 1 if x >= 0 else 0


@activation_function
@lru_cache(maxsize=1000)
def gaussian(x):
    return math.exp(-x ** 2)


@activation_function
@lru_cache(maxsize=1000)
def absolute(x):
    return abs(x)


# print all activation functions in a pretty format
print('Activation functions:')
for name, func in _activation_functions.items():
    print(f'\t{name}: {func.__doc__ if func.__doc__ else "No description"}')
print()
