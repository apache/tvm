"""Statement classes"""
from __future__ import absolute_import as _abs
from ._ctypes._node import NodeBase, register_node

class Stmt(NodeBase):
    pass

@register_node
class LetStmt(Stmt):
    pass

@register_node
class AssertStmt(Stmt):
    pass

@register_node
class ProducerConsumer(Stmt):
    pass

@register_node
class For(Stmt):
    Serial = 0
    Parallel = 1
    Vectorized = 2
    Unrolled = 3

@register_node
class Store(Stmt):
    pass

@register_node
class Provide(Stmt):
    pass

@register_node
class Allocate(Stmt):
    pass

@register_node
class Free(Stmt):
    pass

@register_node
class Realize(Stmt):
    pass

@register_node
class Block(Stmt):
    pass

@register_node
class IfThenElse(Stmt):
    pass

@register_node
class Evaluate(Stmt):
    pass
