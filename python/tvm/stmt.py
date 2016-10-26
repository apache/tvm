from __future__ import absolute_import as _abs
from ._ctypes._api import NodeBase, register_node
from . import function as _func
from . import make as _make

class Stmt(NodeBase):
    def __repr__(self):
        return _func.format_str(self)

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
    pass

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
