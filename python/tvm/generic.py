"""Generic opertor interface for overloading.
We follow the numpy naming convention for this interface
(e.g., tvm.generic.multitply ~ numpy.multiply).
"""
# pylint: disable=unused-argument

#Operator precedence used when overloading.
__op_priority__ = 0

def add(lhs, rhs):
    """generic add interface"""
    raise NotImplementedError

def subtract(lhs, rhs):
    """generic subtract interface"""
    raise NotImplementedError

def multiple(lhs, rhs):
    """generic multiple interface"""
    raise NotImplementedError

def divide(lhs, rhs):
    """generic divide interface"""
    raise NotImplementedError
