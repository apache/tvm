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
    name = ...  # type: str
    opt_level = ...  # type: int
    pass_func = ...  # type: Callable
    required = ...  # type: list

    def __init__(self, name, opt_level, pass_func, required):
        # type: (str, int, Callable, list) -> None
        ...


class FunctionPass(Pass):
    name = ...  # type: str
    opt_level = ...  # type: int
    pass_func = ...  # type: Callable
    required = ...  # type: list

    def __init__(self, name, opt_level, pass_func, required):
        # type: (str, int, Callable, list) -> None
        ...


class SequentialPass(Pass):
    name = ...  # type: str
    opt_level = ...  # type: int
    passes = ...  # type: list
    required = ...  # type: list
    disabled = ... # type: list

    def __init__(self, name, opt_level, passes, required, disabled):
        # type: (str, int, list, list, list) -> None
        ...


def check_expr(env: Module, expr: ir.Expr) -> ir.Type: ...
def generalize(env: Module, expr: ir.Expr) -> ir.Expr: ...
def _get_checked_type(expr: ir.Expr) -> ir.Type: ...
def well_formed(expr: ir.Expr) -> bool: ...
def dead_code_elimination(expr: ir.Expr) -> ir.Expr: ...
