import numpy as np

from node import Node


def _check_log_domain_restrictions(x):
    """
    Checks if the value of a given input x is less than or equal to zero and therefore
    unable to be used as an input for a logrithmic function.
    Parameters
    ----------
    x: An instance of the node class, where x.value is the numeric value of the node and
    x.deriative is the derivative of the node.
    Returns
    -------
    Returns None if x > 0, raises a ValueError if x <= 0.
    
    Examples
    --------
    >>> _check_log_domain_restrictions(Node("1",1,0))
    None
    >>> _check_log_domain_restrictions(Node("0",0,0))
    ValueError: Value 0 not valid for a logarithmic function
    >>> _check_log_domain_restrictions(Node("-1",-1,0))
    ValueError: Value -1 not valid for a logarithmic functionNone
    """
    if x.value <= 0:
        raise ValueError(f"Value {x.value} not valid for a logarithmic function")


def _check_sqrt_domain_restrictions(x):
    """
    Checks if the value of a given input x is less zero and therefore
    unable to be used as an input for a square root function.
    Parameters
    ----------
    x: An instance of the node class, where x.value is the numeric value of the node and
    x.deriative is the derivative of the node.
    Returns
    -------
    Returns None if x >= 0, raises a ValueError if x < 0.
    
    Examples
    --------
    >>> _check_sqrt_domain_restrictions(Node("1",1,0))
    None
    >>> _check_sqrt_domain_restrictions(Node("0",0,0))
    None
    >>> _check_sqrt_domain_restrictions(Node("-1",-1,0))
    ValueError: Square roots of negative numbers not supported
    """
    if x.value < 0:
        raise ValueError("Square roots of negative numbers not supported")


def sqrt(x):
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based of the square
    root of the input node x.
    Parameters
    ----------
    x: An instance of the node class, where x.value is the numeric value of the node and
    x.deriative is the derivative of the node.
    Returns
    -------
    If the node already exists in the Node._NODE_REGISTRY, returns the existing node.
    
    Else, returns new_node, which is a new instance of the Node class and contains three 
    inputs: the symbolic representation of the node "sqrt({})", its forward trace, and 
    tangent trace.
    
    Examples
    --------
    >>> sqrt(Node("1",1,0))
    Node("sqrt(1)", 1, 0)
    >>> sqrt(Node("0",0,0))
    Node("sqrt(0)", 0, 0)
    >>> sqrt(Node("-1",-1,0))
    ValueError: Square roots of negative numbers not supported
    """
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
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the natural log
    of the input node x.
    Parameters
    ----------
    x: An instance of the node class, where x.value is the numeric value of the node and
    x.deriative is the derivative of the node.
    Returns
    -------
    If the node already exists in the Node._NODE_REGISTRY, returns the existing node.
    
    Else, returns new_node, which is a new instance of the Node class and contains three 
    inputs: the symbolic representation of the node "ln({})", its forward trace, and 
    tangent trace.
    
    Examples
    --------
    >>> ln(Node("1",1,0))
    Node("ln(1)", 0, 1)
    >>> ln(Node("0",0,0))
    ValueError: Value 0 not valid for a logarithmic function
    >>> ln(Node("-1",-1,0))
    ValueError: Value -1 not valid for a logarithmic function
    """
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
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the base 10
    logarithm of the input node x.
    Parameters
    ----------
    x: An instance of the node class, where x.value is the numeric value of the node and
    x.deriative is the derivative of the node.
    Returns
    -------
    If the node already exists in the Node._NODE_REGISTRY, returns the existing node.
    
    Else, returns new_node, which is a new instance of the Node class and contains three 
    inputs: the symbolic representation of the node "log10({})", its forward trace, and 
    tangent trace.
    
    Examples
    --------
    >>> log10(Node("1",1,0))
    Node("log10(1)", 0, 0.4343)
    >>> log10(Node("0",0,0))
    ValueError: Value 0 not valid for a logarithmic function
    >>> log10(Node("-1",-1,0))
    ValueError: Value -1 not valid for a logarithmic function
    """
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
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the base 2
    logarithm of the input node x.
    Parameters
    ----------
    x: An instance of the node class, where x.value is the numeric value of the node and
    x.deriative is the derivative of the node.
    Returns
    -------
    If the node already exists in the Node._NODE_REGISTRY, returns the existing node.
    
    Else, returns new_node, which is a new instance of the Node class and contains three 
    inputs: the symbolic representation of the node "log2({})", its forward trace, and 
    tangent trace.
    
    Examples
    --------
    >>> log2(Node("1",1,0))
    Node("log2(1)", 0, 1.4427)
    >>> log2(Node("0",0,0))
    ValueError: Value 0 not valid for a logarithmic function
    >>> log2(Node("-1",-1,0))
    ValueError: Value -1 not valid for a logarithmic function
    """
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
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the exponential
    value of the input node x.
    Parameters
    ----------
    x: An instance of the node class, where x.value is the numeric value of the node and
    x.deriative is the derivative of the node.
    Returns
    -------
    If the node already exists in the Node._NODE_REGISTRY, returns the existing node.
    
    Else, returns new_node, which is a new instance of the Node class and contains three 
    inputs: the symbolic representation of the node "exp({})", its forward trace, and 
    tangent trace.
    
    Examples
    --------
    >>> exp(Node("1",1,0))
    Node("exp(1)", 2.7183, 0)
    >>> exp(Node("0",0,0))
    Node("exp(0)", 1, 0)
    >>> exp(Node("-1",-1,0))
    Node("exp(-1)", 0.3679, 0)
    """
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
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the sine
    of the input node x.
    Parameters
    ----------
    x: An instance of the node class, where x.value is the numeric value of the node and
    x.deriative is the derivative of the node.
    Returns
    -------
    If the node already exists in the Node._NODE_REGISTRY, returns the existing node.
    
    Else, returns new_node, which is a new instance of the Node class and contains three 
    inputs: the symbolic representation of the node "sin({})", its forward trace, and 
    tangent trace.
    
    Examples
    --------
    >>> sin(Node("1",1,0))
    Node("sin(1)", 0.8415, 0)
    >>> sin(Node("0",0,0))
    Node("sin(0)", 0, 0)
    >>> sin(Node("-1",-1,0))
    Node("sin(-1)", -0.8415, 0)
    """
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
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the cosine
    of the input node x.
    Parameters
    ----------
    x: An instance of the node class, where x.value is the numeric value of the node and
    x.deriative is the derivative of the node.
    Returns
    -------
    If the node already exists in the Node._NODE_REGISTRY, returns the existing node.
    
    Else, returns new_node, which is a new instance of the Node class and contains three 
    inputs: the symbolic representation of the node "cos({})", its forward trace, and 
    tangent trace.
    
    Examples
    --------
    >>> cos(Node("1",1,0))
    Node("cos(1)", 0.5403, 0)
    >>> cos(Node("0",1,0))
    Node("cos(0)", 1, 0)
    >>> cos(Node("-1",-1,0))
    Node("cos(-1)", -0.5403, 0)
    """
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
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the tangent
    of the input node x.
    Parameters
    ----------
    x: An instance of the node class, where x.value is the numeric value of the node and
    x.deriative is the derivative of the node.
    Returns
    -------
    If the node already exists in the Node._NODE_REGISTRY, returns the existing node.
    
    Else, returns new_node, which is a new instance of the Node class and contains three 
    inputs: the symbolic representation of the node "tan({})", its forward trace, and 
    tangent trace.
    
    Examples
    --------
    >>> tan(Node("1",1,0))
    Node("tan(1)", 1.557, 0)
    >>> tan(Node("0",0,0))
    Node("tan(0)", 0, 0)
    >>> tan(Node("-1",-1,0))
    Node("tan(-1)", -1.557, 0)
    """
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
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the arcsin
    of the input node x.
    Parameters
    ----------
    x: An instance of the node class, where x.value is the numeric value of the node and
    x.deriative is the derivative of the node.
    Returns
    -------
    If the node already exists in the Node._NODE_REGISTRY, returns the existing node.
    
    Else, returns new_node, which is a new instance of the Node class and contains three 
    inputs: the symbolic representation of the node "arcsin({})", its forward trace, and 
    tangent trace.
    
    Examples
    --------
    >>> arcsin(Node("1",1,0))
    Node("arcsin(1)", 1.5708, 0)
    >>> arcsin(Node("0",0,0))
    Node("arcsin(0)", 0, 0)
    >>> arcsin(Node("-1",-1,0))
    Node("arcsin(-1)", -1.5708, 0)
    """
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
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the arccos
    of the input node x.
    Parameters
    ----------
    x: An instance of the node class, where x.value is the numeric value of the node and
    x.deriative is the derivative of the node.
    Returns
    -------
    If the node already exists in the Node._NODE_REGISTRY, returns the existing node.
    
    Else, returns new_node, which is a new instance of the Node class and contains three 
    inputs: the symbolic representation of the node "arccos({})", its forward trace, and 
    tangent trace.
    
    Examples
    --------
    >>> arccos(Node("1",1,0))
    Node("arccos(1)", 3.1416, 0)
    >>> arccos(Node("0",0,0))
    Node("arccos(0)", 1.5708, 0)
    >>> arccos(Node("-1",-1,0))
    Node("arccos(-1)", -3.1416, 0)
    """
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
    """
    Takes in an instance of the Node class and returns a new node with its symbolic
    representation, forward trace, and tangent trace, which are based on the arctan
    of the input node x.
    Parameters
    ----------
    x: An instance of the node class, where x.value is the numeric value of the node and
    x.deriative is the derivative of the node.
    Returns
    -------
    If the node already exists in the Node._NODE_REGISTRY, returns the existing node.
    
    Else, returns new_node, which is a new instance of the Node class and contains three 
    inputs: the symbolic representation of the node "arctan({})", its forward trace, and 
    tangent trace.
    
    Examples
    --------
    >>> arctan(Node("1",1,0))
    Node("arctan(1)", 0.7854, 0)
    >>> arctan(Node("0",0,0))
    Node("arctan(0)", 0, 0)
    >>> arctan(Node("-1",-1,0))
    Node("arctan(-1)", -0.7854, 0)
    """
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
