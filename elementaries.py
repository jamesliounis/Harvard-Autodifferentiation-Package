import numpy as np
import pytest
from node import Node


def _check_log_domain_restrictions(x):

    if x.value <= 0:
        raise ValueError(f"Value {x.value} not valid for a logarithmic function")

def _check_sqrt_domain_restrictions(x):

    if x.value < 0:
        raise ValueError("Square roots of negative numbers not supported")


def sqrt(x):
    symbolic_representation = "sqrt({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_to_node(x)

    _check_sqrt_domain_restrictions(x)

    forward_trace = np.sqrt(x.value)
    tangent_trace = x.derivative / 2 * np.sqrt(x.value)
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)
    Node._insert_node_to_registry(new_node)
    return new_node


def ln(x):
    symbolic_representation = "ln({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_to_node(x)

    _check_log_domain_restrictions(x)

    forward_trace = np.log(x.value)
    tangent_trace = 1 / x.value
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)
    Node._insert_node_to_registry(new_node)
    return new_node


def log10(x):
    symbolic_representation = "log10({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_to_node(x)

    _check_log_domain_restrictions(x)

    forward_trace = np.log10(x.value)
    tangent_trace = 1 / (x.value * np.log(10))
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)
    Node._insert_node_to_registry(new_node)
    return new_node


def log2(x):
    symbolic_representation = "log2({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_to_node(x)

    _check_log_domain_restrictions(x)

    forward_trace = np.log2(x.value)
    tangent_trace = 1 / (x.value * np.log(2))
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)
    Node._insert_node_to_registry(new_node)
    return new_node


def exp(x):
    symbolic_representation = "exp({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_to_node(x)

    forward_trace = np.exp(x.value)
    tangent_trace = x.derivative * forward_trace
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)
    Node._insert_node_to_registry(new_node)
    return new_node


def sin(x):
    symbolic_representation = "sin({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_to_node(x)

    forward_trace = np.sin(x.value)
    tangent_trace = np.cos(x.value) * x.derivative
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)
    Node._insert_node_to_registry(new_node)
    return new_node


def cos(x):
    symbolic_representation = "cos({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_to_node(x)

    forward_trace = np.cos(x.value)
    tangent_trace = -np.sin(x.value) * x.derivative
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)
    Node._insert_node_to_registry(new_node)
    return new_node


def tan(x):
    symbolic_representation = "tan({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_to_node(x)

    forward_trace = np.tan(x.value)
    tangent_trace = x.derivative / (np.cos(x.val) ** 2)
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)
    Node._insert_node_to_registry(new_node)
    return new_node


def arcsin(x):
    symbolic_representation = "arcsin({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_to_node(x)

    forward_trace = np.arcsin(x.value)
    tangent_trace = x.derivative / np.sqrt(1 - x.value**2)
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)
    Node._insert_node_to_registry(new_node)
    return new_node


def arccos(x):
    symbolic_representation = "arccos({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_to_node(x)

    forward_trace = np.arccos(x.value)
    tangent_trace = -x.derivative / np.sqrt(1 - x.value**2)
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)
    Node._insert_node_to_registry(new_node)
    return new_node


def arctan(x):
    symbolic_representation = "arctan({})".format(str(x))
    if Node._check_node_exists(symbolic_representation):
        return Node._get_existing_node(symbolic_representation)

    x = Node._convert_to_node(x)

    forward_trace = np.arccos(x.value)
    tangent_trace = -x.derivative / (1 + x.value**2)
    new_node = Node(symbolic_representation, forward_trace, tangent_trace)
    Node._insert_node_to_registry(new_node)
    return new_node


if __name__ == "__main__":
    x = Node("x", 5, 1)
    z = log2(x)

    print(z.value)


"""Testing the `_check_log_domain_restrictions` function above"""

def test_check_log_domain_restrictions(self):
    
    node = Node('x', -15, 11)

    with pytest.raises(ValueError):
        _check_log_domain_restrictions(node)

