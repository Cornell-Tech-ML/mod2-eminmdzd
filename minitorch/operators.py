import math
from typing import Callable, Iterable


def mul(a: float, b: float) -> float:
    """Multiply two floats."""
    return a * b


def id(a: float) -> float:
    """Identity function."""
    return a


def add(a: float, b: float) -> float:
    """Add two floats."""
    return a + b


def neg(a: float) -> float:
    """Negate a float."""
    return -1.0 * a


def lt(a: float, b: float) -> float:
    """Less than."""
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Equal."""
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Maximum of two floats."""
    return a if a > b else b


def is_close(a: float, b: float) -> float:
    """Close function to check if two floats are close."""
    return abs(a - b) < 1e-2


def sigmoid(a: float) -> float:
    """Sigmoid function."""
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        exp_a = math.exp(a)
        return exp_a / (1.0 + exp_a)


def relu(a: float) -> float:
    """ReLU activation function."""
    return a if a > 0 else 0.0


EPS = 1e-6


def log(a: float) -> float:
    """Logarithm function."""
    return math.log(a + EPS)


def exp(a: float) -> float:
    """Exponential function."""
    return math.exp(a)


def inv(x: float) -> float:
    """Returns the inverse of x."""
    if x == 0:
        raise ZeroDivisionError("Inverse of zero is undefined.")
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of logarithm function

    Args:
    ----
    x (float): The input to the logarithm.
    y (float): The second argument multiplied by the logarithm.

    Returns:
    -------
    float: The derivative of the logarithm function.

    """
    return y / (x + EPS)


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of the reciprocal function.

    Args:
    ----
    x (float): The input for the reciprocal function.
    y (float): The second argument multiplied by the reciprocal.

    Returns:
    -------
    float: The derivative of the reciprocal function.

    """
    # Ensure x != 0 to avoid division by zero
    return -(1.0 / x**2) * y


def relu_back(x: float, dout: float) -> float:
    """Computes the gradient of the ReLU function times the gradient from the next layer.

    Args:
    ----
    x (float): The input to the ReLU function.
    dout (float): The gradient of the loss with respect to the output of ReLU.

    Returns:
    -------
    float: The gradient of the loss with respect to the input x.

    """
    return dout if x > 0 else 0.0


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a given function to each element of an iterable and returns an iterable of the same type."""

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Combines elements from two iterables using a given function and returns an iterable of the same type."""

    def _zipwith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipwith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduces an iterable to a single value using a binary function."""

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negates all elements in the list."""
    return map(neg)(ls)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Adds corresponding elements of two lists."""
    return zipWith(add)(lst1, lst2)


def sum(lst: Iterable[float]) -> float:
    """Sums all elements in the list."""
    return reduce(add, 0.0)(lst)


def prod(lst: Iterable[float]) -> float:
    """Multiplies all elements in the list."""
    return reduce(mul, 1.0)(lst)
