from ._ctypes._api import NodeBase, register_node

class Expr(NodeBase):
    pass

@register_node("VarNode")
class Var(Expr):
    pass

@register_node("BinaryOpNode")
class BinaryOpNode(Expr):
    pass
