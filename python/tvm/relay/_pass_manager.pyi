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


class SequentialPass(Pass):
    name = ...  # type: str
    opt_level = ... # type: int
    pass_kind = ... # type: PassKind
    passes = ...  # type: list
    disabled = ... # type: list

    def __init__(self, name, opt_level, pass_kind, passes, disabled):
        # type: (str, int, int(PassKind), list, list) -> None
        ...
