import tvm
from .base import NodeBase


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
