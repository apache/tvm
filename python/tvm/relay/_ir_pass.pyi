import tvm
from . import ir
from .base import NodeBase
from .env import Module


class PassContext(NodeBase):
    def __init__(self):
        ...

class PassInfo(NodeBase):
    name = ...  # type: str
    opt_level = ... # type: int
    required = ... # type: list

    def __init__(self, name, opt_level, required)
        # type: (str, int, list) -> None


class Pass(NodeBase):
    def __init__(self):
        ...


class ModulePass(Pass):
    pass_info = ...  # type: list
    pass_func = ...  # type: Callable

    def __init__(self, pass_info, pass_func):
        # type: (list, Callable) -> None
        ...


class FunctionPass(Pass):
    pass_info = ...  # type: list
    pass_func = ...  # type: Callable

    def __init__(self, pass_info, pass_func):
        # type: (list, Callable) -> None
        ...


class SequentialPass(Pass):
    pass_info = ...  # type: list
    passes = ...  # type: list
    disabled = ... # type: list

    def __init__(self, pass_info, passes, disabled):
        # type: (str, list, list, list) -> None
        ...


def check_expr(env: Module, expr: ir.Expr) -> ir.Type: ...
def generalize(env: Module, expr: ir.Expr) -> ir.Expr: ...
def _get_checked_type(expr: ir.Expr) -> ir.Type: ...
def well_formed(expr: ir.Expr) -> bool: ...
def dead_code_elimination(expr: ir.Expr) -> ir.Expr: ...
