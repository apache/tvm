# pylint: disable=no-else-return
# pylint: disable=unidiomatic-typecheck
"""The pass manager for Relay.

This file exposes differen granularity of interfaces for users to implement and
use passes more conveniently.
"""
from abc import abstractmethod
from enum import IntEnum

from . import _optimizer as _opt
from .base import RelayNode, register_relay_node


class PassKind(IntEnum):
    """The different granularity of passes for optimization/analysis."""
    ModuleKind = 1
    FunctionKind = 2


@register_relay_node
class PassContext(RelayNode):
    """The basis where a Relay optimization/analysis runs on.
    Each pass context contains a number of auxiliary information that is used
    to help an optimization pass. Such information includes the error reporter
    to record the errors of during the performing the optimization, etc.
    """

    def __init__(self):
        self.__init_handle_by_constructor__(_opt.PassContext)


@register_relay_node
class Pass(RelayNode):
    """The base class of all passes. This class is designed as a pure virtual
    class that will be implemented by the subclasses.

    Parameters
    ----------
    name : str
        The pass name.

    opt_level : int
        The optimization level of this pass.

    pass_kind : PassKind
        The type of pass for optimization/analysis.

    enabled : bool
        The flag indicates if a pass is enabled.

    required_passes : List[str]
        The list of dependent passes to perform this pass.
    """

    @abstractmethod
    def run(self, mod, pass_ctx=None):
        """Execute the pass. It is an abstract function that will be
        implemented by subclasses.

        Parameters
        ----------
        mod : tvm.relay.Module
            The module that a certain optimization is performed on.

        pass_ctx : Optional[PassContext]
            The auxiliary information that helps perform the pass.

        Returns
        -------
        mod : tvm.relay.Module
            The updated module after applying this pass.
        """
        raise NotImplementedError("Pure virtual function is not implemented.")

    def __call__(self, mod, pass_ctx=None):
        return run(mod, pass_ctx)


@register_relay_node
class ModulePass(Pass):
    """A pass that works on tvm.relay.Module.

    Parameters
    ----------
    name : str
        The pass name.

    opt_level : int
        The optimization level of this pass.

    pass_func : Callable[PassContext: tvm.relay.Module -> tvm.relay.Module]

    enabled : bool
        The flag indicates if a pass is enabled.

    required_passes : List[str]
        The list of dependent passes to perform this pass.
       The implemented optimization pass.
    """

    def __init__(self, name, opt_level, pass_func, enabled=True,
                 required_passes=None):
        required_passes = required_passes if required_passes else []
        if not isinstance(required_passes, list):
            raise TypeError("required_passes is expected to be a list of str.")
        self.__init_handle_by_constructor__(_opt.ModulePass, name, opt_level,
                                            pass_func, enabled, required_passes)

    def run(self, mod, pass_ctx=None):
        """Execute a module pass.

        Parameters
        ----------
        mod : tvm.relay.Module
            The module that the module pass is executed on.

        pass_ctx : Optional[PassContext]
            The auxiliary information that helps perform the pass.

        Returns
        -------
        ret : tvm.relay.Module
            The updated module.
        """
        return _opt.RunModulePass(self, mod, pass_ctx)


@register_relay_node
class FunctionPass(Pass):
    """A pass that works on each tvm.relay.Function in a module.

    Parameters
    ----------
    name : str
        The pass name.

    opt_level : int
        The optimization level of this pass.

    pass_func : Callable[PassContext: tvm.relay.Function -> tvm.relay.Function]
        The implemented optimization pass.

    enabled : bool
        The flag indicates if a pass is enabled.

    required_passes : List[str]
        The list of dependent passes to perform this pass.
    """

    def __init__(self, name, opt_level, pass_func, enabled=True,
                 required_passes=None):
        required_passes = required_passes if required_passes else []
        if not isinstance(required_passes, list):
            raise TypeError("required_passes is expected to be a list of str.")
        self.__init_handle_by_constructor__(_opt.FunctionPass, name, opt_level,
                                            pass_func, enabled, required_passes)

    def run(self, mod, pass_ctx=None):
        """Execute a function pass.

        Parameters
        ----------
        mod : tvm.relay.Module
            The module that the function pass is executed on.

        pass_ctx : Optional[PassContext]
            The auxiliary information that helps perform the pass.

        Returns
        -------
        ret : tvm.relay.Module
            The updated module.
        """
        return _opt.RunFunctionPass(self, mod, pass_ctx)


def build_pass(pass_name, opt_level, pass_kind, pass_func, enabled=True,
               required_passes=None):
    """Create a pass using a defined optimization function from Python.

    Parameters
    ----------
    pass_name : str
        The name of the pass.

    opt_level : int
        The optimization level of this pass.

    pass_kind : PassKind
        The type of pass for optimization/analysis.

    pass_func : Callable[PassContext: Module/Function/Expr -> Module/Function/Expr]
        The implemented optimization pass.

    enabled : bool
        The flag indicates if a pass is enabled.

    required_passes : List[str]
        The list of dependent passes to perform this pass.

    Returns
    -------
    ret : Pass
        The pass the built through pass_func.
    """
    required_passes = required_passes if required_passes else []
    if not isinstance(required_passes, list):
        raise TypeError("required_passes is expected to be a list of str.")

    if not isinstance(pass_kind, PassKind):
        raise TypeError("pass_kind is expected to be the type of PassKind.")

    if pass_kind == PassKind.ModuleKind:
        return _opt.ModulePass(pass_name, opt_level, pass_func, enabled,
                               required_passes)
    else:
        return _opt.FunctionPass(pass_name, opt_level, pass_func, enabled,
                                 required_passes)


def optimize(passes, mod, pass_ctx=None):
    """Run a host of passes on a given tvm.relay.Module through the optimizer.

    Parameters
    ----------
    passes : List[Pass]
        The passes to be executed.

    mod : tvm.relay.Module
        The module that the passes are executed on.

    pass_ctx : Optional[PassContext]
        The auxiliary information that helps perform the passes. The provided
        information should be shared by all passes.

    Returns
    -------
    ret : tvm.relay.Module
        The updated module after running the provided passes.
    """
    return _opt.Optimize(passes, mod, pass_ctx)
