import tvm
from .base import NodeBase


class PassState(NodeBase):
    mod = ...  # type: tvm.relay.Module

    def __init__(self, mod):
        # type: tvm.relay.Module -> None
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

    def __init__(self, name, opt_level, pass_kind, pass_func):
        # type: (str, int, int(PassKind), Callable) -> None
        ...


class FunctionPass(Pass):
    name = ...  # type: str
    opt_level = ... # type: int
    pass_kind = ... # type: PassKind
    pass_func = ...  # type: Callable

    def __init__(self, name, opt_level, pass_kind, pass_func):
        # type: (str, int, int(PassKind), Callable) -> None
        ...


class ExprPass(Pass):
    name = ...  # type: str
    opt_level = ... # type: int
    pass_kind = ... # type: PassKind
    pass_func = ...  # type: Callable

    def __init__(self, name, opt_level, pass_kind, pass_func):
        # type: (str, int, int(PassKind), Callable) -> None
        ...
