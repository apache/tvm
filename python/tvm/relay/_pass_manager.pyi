import tvm
from .base import NodeBase


class PassContext(NodeBase):
    def __init__(self):
        ...


class Pass(NodeBase):
    name = ...  # type: str
    opt_level = ... # type: int
    pass_kind = ... # type: PassKind


class ModulePass(Pass):
    name = ...  # type: str
    opt_level = ... # type: int
    pass_kind = ... # type: PassKind
    pass_func = ...  # type: Callable
    enabled = ... # type: bool

    def __init__(self, name, opt_level, pass_kind, pass_func, enabled=True):
        # type: (str, int, int(PassKind), Callable, bool) -> None
        ...


class FunctionPass(Pass):
    name = ...  # type: str
    opt_level = ... # type: int
    pass_kind = ... # type: PassKind
    pass_func = ...  # type: Callable
    enabled = ... # type: bool

    def __init__(self, name, opt_level, pass_kind, pass_func, enabled=True):
        # type: (str, int, int(PassKind), Callable, bool) -> None
        ...


class SequentialPass(Pass):
    name = ...  # type: str
    opt_level = ... # type: int
    pass_kind = ... # type: PassKind
    passes = ...  # type: list

    def __init__(self, name, opt_level, pass_kind, passes):
        # type: (str, int, int(PassKind), list, bool) -> None
        ...
