from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_list = list(vals)
    vals_pos = vals_list.copy()
    vals_neg = vals_list.copy()
    vals_pos[arg] += epsilon
    vals_neg[arg] -= epsilon

    f_pos = f(*vals_pos)
    f_neg = f(*vals_neg)

    return (f_pos - f_neg) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative for the variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns the unique ID of the variable."""
        ...

    def is_leaf(self) -> bool:
        """Checks if the variable is a leaf node (created by the user)."""
        ...

    def is_constant(self) -> bool:
        """Checks if the variable is a constant (has no history)."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables that were used to create this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Performs the chain rule to backpropagate the gradient through the computation graph."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Parameters
    ----------
    variable : Variable
        The right-most variable (output of the computation graph).
    deriv : Any
        The initial derivative to propagate backward, typically set to 1.0 for the output variable.

    Returns
    -------
    None
        No return. The derivatives are accumulated and written to the leaf nodes via `accumulate_derivative`.

    """
    # Get the topological order of the variables in the graph
    queue = topological_sort(variable)

    # Dictionary to store the gradient for each variable, using unique_id as the key
    derivatives = {variable.unique_id: deriv}

    # Traverse in topological order (from output to inputs)
    for var in queue:
        deriv = derivatives[var.unique_id]

        if var.is_leaf():
            var.accumulate_derivative(deriv)

        else:
            # Accumulate derivatives for each parent variable using unique_id as key
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Saves values from the forward pass for use during backpropagation."""
        return self.saved_values
