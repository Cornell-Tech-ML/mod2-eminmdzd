from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the forward function to the input Scalars and sets up their history for backward propagation."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the addition of two Scalars."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the gradients for addition."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the logarithm of a Scalar."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the logarithm function."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.
class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the multiplication of two Scalars."""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_out: float) -> Tuple[float, float]:
        """Computes the gradients for multiplication."""
        a, b = ctx.saved_values
        return d_out * b, d_out * a


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the inverse of a Scalar."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_out: float) -> float:
        """Computes the gradient of the inverse function."""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_out)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the negation of a Scalar."""
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_out: float) -> float:
        """Computes the gradient for negation."""
        return -1.0 * d_out


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) =  1 / (1 + exp(-x))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the sigmoid function for a Scalar."""
        sigmoid_value = operators.sigmoid(a)
        ctx.save_for_backward(sigmoid_value)
        return sigmoid_value

    @staticmethod
    def backward(ctx: Context, d_out: float) -> float:
        """Computes the gradient for the sigmoid function."""
        (sigmoid_value,) = ctx.saved_values
        return d_out * sigmoid_value * (1.0 - sigmoid_value)


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the ReLU (Rectified Linear Unit) of a Scalar."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_out: float) -> float:
        """Computes the gradient for the ReLU function."""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_out)


class Exp(ScalarFunction):
    """Exponential function $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the exponential of a Scalar."""
        exp_value = operators.exp(a)
        ctx.save_for_backward(exp_value)
        return exp_value

    @staticmethod
    def backward(ctx: Context, d_out: float) -> float:
        """Computes the gradient for the exponential function."""
        (exp_value,) = ctx.saved_values
        return d_out * exp_value


class LT(ScalarFunction):
    r"""Less than function $f(x, y) = 1.0 \text{ if } x < y \text{ else } 0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes whether one Scalar is less than another."""
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_out: float) -> Tuple[float, float]:
        """Returns zero gradients for the less-than operation."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    r"""Equality function $f(x, y) = 1.0 \text{ if } x == y \text{ else } 0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes whether two Scalars are equal."""
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_out: float) -> Tuple[float, float]:
        """Returns zero gradients for the equality operation."""
        return 0.0, 0.0
