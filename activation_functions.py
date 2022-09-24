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
    """Linear function that will output the input directly if it is positive, otherwise, it will output zero"""
    return max(0, x)


@activation_function
@lru_cache(maxsize=1000)
def lrelu(x):
    """Linear function that will output the input directly if it is positive, otherwise, it will output a small number"""
    return max(0.01 * x, x)

@activation_function
@lru_cache(maxsize=1000)
def sigmoid(x):
    """Sigmoid function that will output a value between 0 and 1"""
    return 1 / (1 + math.exp(-x))


@activation_function
@lru_cache(maxsize=1000)
def tanh(x):
    """Hyperbolic tangent function that will output a value between -1 and 1"""
    return math.tanh(x)


@activation_function
@lru_cache(maxsize=1000)
def identity(x):
    """Linear function that will output the input directly"""
    return x


@activation_function
@lru_cache(maxsize=1000)
def binary_step(x):
    """Binary step function that will output 1 if the input is positive, otherwise, it will output 0"""
    return 1 if x >= 0 else 0


@activation_function
@lru_cache(maxsize=1000)
def gaussian(x):
    """Gaussian function that will output a value between 0 and 1"""
    return math.exp(-x ** 2)


@activation_function
@lru_cache(maxsize=1000)
def absolute(x):
    """Absolute function that will output the absolute value of the input"""
    return abs(x)


@activation_function
@lru_cache(maxsize=1000)
def softsign(x):
    """Softsign function that will output a value between -1 and 1"""
    return x / (1 + abs(x))


@activation_function
@lru_cache(maxsize=1000)
def sin(x):
    """Sine function that will output a value between -1 and 1"""
    return math.sin(x)



def main():
    print('Activation functions:')
    for name, func in _activation_functions.items():
        print(f'\t{name}: {func.__doc__ if func.__doc__ else "No description"}')
    print()

if __name__ == "__main__":
    main()