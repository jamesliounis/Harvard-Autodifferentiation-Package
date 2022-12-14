


#importing Node class:

import pytest

from node import Node

import numpy as np

class Tests_Node:

    """ Test class for Milestone 2 """

    """ The testing suite that follows attempts to test every single method implemented as part"""
    """ of the Node() class implemented in Milestone2 """

    # Defining other types that are able to be converted into nodes:
    _COMPATIBLE_TYPES = (int, float, np.array)
    
    def test_init(self):

        """
        Testing the instantiation of a node which is the foundation of a computational graph.
        :param symbol: Symbolic representation of a Node instance that acts as a unique identifier.
        :param value: Analytical value of the node.
        :param derivative: Derivative with respect to the value attribute. Default=1.
        We test here an example instantiation:
        > x = Node('x', 10, 1)
        """

        node = Node('x', 10, 1)
        assert node._symbol == 'x'
        assert node._value == 10
        assert node._derivative == 1


    
    def test_value(self):   
        
        """Testing the `value` @property method"""

        node = Node('x', 100, 1)
        assert node.value == 100

    
    def test_symbol(self):

        """Testing the `symbol` method"""

        node = Node('x', 10, 1)
        assert node.symbol == 'x'

    
    def test_derivative(self):

        """Testing the `derivative` method"""

        node = Node('x', 10, 1)
        assert node.derivative == 1



    def test_check_foreign_type_compatibility(other_type):
        """
        Testing the method that sees if a datatype can be represented as a node.
        :param other_type: Type to check for conversion compatibility.
        :return: boolean. True if other_type is a supported type. False otherwise.

        We need to test all cases that are included in the _COMPATIBLE_TYPES options. 

        """



        assert Node._check_foreign_type_compatibility(1) == isinstance(1, Node._COMPATIBLE_TYPES)
        assert Node._check_foreign_type_compatibility(1.0) == isinstance(1.00, Node._COMPATIBLE_TYPES)
        assert Node._check_foreign_type_compatibility(np.ndarray([1,2,3])) == isinstance(np.ndarray([1,2,3]), Node._COMPATIBLE_TYPES)

    def test_check_node_exists(key):
       
       """
        Testing method that checks if an instance of class Node has already been created.
        :param key: Symbolic representation of a Node instance that acts as a unique identifier.
        :return: boolean. True if key argument is found. False otherwise.
        """
        # We define an arbitrary v0 node in the node registry and check if it exists. 
        Node._NODE_REGISTRY = {'v0':Node('x', 10, 1)}

        assert Node._check_node_exists('v0') == True
        
    def test_get_existing_node(key):

        """
        Testing the static method that returns existing Node instance to avoid recomputing nodes.
        :param key: Symbolic representation of a Node instance that acts as a unique identifier.
        :return: Node instance that matches the specified key.
        """

        # Following same logic as above
        Node._NODE_REGISTRY = {'v0':Node('x', 10, 1)}
        
        assert Node._get_existing_node('v0') == Node._NODE_REGISTRY['v0']


