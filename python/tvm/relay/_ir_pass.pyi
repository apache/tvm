import tvm
from . import ir
from .base import NodeBase
from .env import Module


class PassContext(NodeBase):
    def __init__(self):
        ...


class Pass(NodeBase):
    name = ...  # type: str
    opt_level = ... # type: int


class ModulePass(Pass):
    name = ...  # type: str
    opt_level = ... # type: int
    pass_func = ...  # type: Callable

    def __init__(self, name, opt_level, pass_func):
        # type: (str, int, Callable) -> None
        ...


class FunctionPass(Pass):
    name = ...  # type: str
    opt_level = ... # type: int
    pass_func = ...  # type: Callable

    def __init__(self, name, opt_level, pass_func):
        # type: (str, int, Callable) -> None
        ...


class SequentialPass(Pass):
    name = ...  # type: str
    opt_level = ... # type: int
    passes = ...  # type: list
    disabled = ... # type: list

    def __init__(self, name, opt_level, passes, disabled):
        # type: (str, int, list, list) -> None
        ...


def check_expr(env: Module, expr: ir.Expr) -> ir.Type: ...
def generalize(env: Module, expr: ir.Expr) -> ir.Expr: ...
def _get_checked_type(expr: ir.Expr) -> ir.Type: ...
def well_formed(expr: ir.Expr) -> bool: ...
def dead_code_elimination(expr: ir.Expr) -> ir.Expr: ...
