# pylint: disable=no-else-return
# pylint: disable=unidiomatic-typecheck
"""The pass manager for Relay.

This file exposes differen granularity of interfaces for users to implement and
use passes more conveniently.
"""
from abc import abstractmethod
from enum import IntEnum

from . import _optimizer as _opt
from .module import Module
from .base import RelayNode, register_relay_node


class PassKind(IntEnum):
    """The different granularity of passes for optimization/analysis."""
    ModuleKind = 1
    FunctionKind = 2
    ExprKind = 3


@register_relay_node
class PassState(RelayNode):
    """The basis where a Relay optimization/analysis runs on.

    Each pass state contains a tvm.relay.Module object and some other
    information such as the error reporter to record the state of a pass on its
    completion.

    Parameters
    ----------
    mod : tvm.relay.Module
        The module object where optimization should run on.
    """

    def __init__(self, mod):
        if not isinstance(mod, Module):
            raise TypeError(
                "mod is expected to be the type of tvm.relay.Module")
        self.__init_handle_by_constructor__(_opt.PassState, mod)


@register_relay_node
class Pass(RelayNode):
    """The base class of all passes.

    Parameters
    ----------
    name : str
        The pass name.

    opt_level : int
        The optimization level of this pass.

    pass_kind : PassKind
        The type of pass for optimization/analysis.
    """

    @abstractmethod
    def run(self, pass_state):
        """Execute the pass. It is an abstract function that will be
        implemented by subclasses.
        """
        raise NotImplementedError("Pure virtual function is not implemented.")


@register_relay_node
class ModulePass(Pass):
    """A pass that works on tvm.relay.Module.

    Parameters
    ----------
    name : str
        The pass name.

    opt_level : int
        The optimization level of this pass.

    pass_func : Callable[PassState: tvm.relay.Module -> tvm.relay.Module]
        The implemented optimization pass.
    """

    def __init__(self, name, opt_level, pass_func):
        self.__init_handle_by_constructor__(_opt.ModulePass, name, opt_level,
                                            pass_func)

    def run(self, pass_state):
        """Execute a module pass.

        Parameters
        ----------
        pass_state : PassState
            The pass state where the module pass is executed on.

        Returns
        -------
        ret : PassState
            The updated pass state.
        """
        return _opt.RunModulePass(self, pass_state)


@register_relay_node
class FunctionPass(Pass):
    """A pass that works on tvm.relay.Function.

    Parameters
    ----------
    name : str
        The pass name.

    opt_level : int
        The optimization level of this pass.

    pass_func : Callable[PassState: tvm.relay.Function -> tvm.relay.Function]
        The implemented optimization pass.
    """

    def __init__(self, name, opt_level, pass_func):
        self.__init_handle_by_constructor__(_opt.FunctionPass, name, opt_level,
                                            pass_func)

    def run(self, pass_state):
        """Execute a function pass.

        Parameters
        ----------
        pass_state : PassState
            The pass state where the function pass is executed on.

        Returns
        -------
        ret : PassState
            The updated pass state.
        """
        return _opt.RunFunctionPass(self, pass_state)


@register_relay_node
class ExprPass(Pass):
    """A pass that works on tvm.relay.Expr.

    Parameters
    ----------
    name : str
        The pass name.

    opt_level : int
        The optimization level of this pass.

    pass_func : Callable[PassState: tvm.relay.Expr -> tvm.relay.Expr]
        The implemented optimization pass.
    """

    def __init__(self, name, opt_level, pass_func):
        self.__init_handle_by_constructor__(_opt.ExprPass, name, opt_level,
                                            pass_func)

    def run(self, pass_state):
        """Execute an expr pass.

        Parameters
        ----------
        pass_state : PassState
            The pass state where the expr pass is executed on.

        Returns
        -------
        ret : PassState
            The updated pass state.
        """
        return _opt.RunExprPass(self, pass_state)


def build_pass(pass_name, opt_level, pass_kind, pass_func):
    """Create a pass using a defined optimization function from Python.

    Parameters
    ----------
    pass_name : str
        The name of the pass.

    opt_level : int
        The optimization level of this pass.

    pass_kind : PassKind
        The type of pass for optimization/analysis.

    pass_func : Callable[PassState: Module/Function/Expr -> Module/Function/Expr]
        The implemented optimization pass.

    Returns
    -------
    ret : Pass
        The pass the built through pass_func.
    """
    if not isinstance(pass_kind, PassKind):
        raise TypeError("pass_kind is expected to be the type of PassKind.")

    if pass_kind == PassKind.ModuleKind:
        return _opt.ModulePass(pass_name, opt_level, pass_func)
    elif pass_kind == PassKind.FunctionKind:
        return _opt.FunctionPass(pass_name, opt_level, pass_func)
    else:
        return _opt.ExprPass(pass_name, opt_level, pass_func)


def optimize(passes, pass_state):
    """Run a host of passes on a given pass state through the optimizer.

    Parameters
    ----------
    passes : List[Pass]
        The passes to be executed.

    pass_state : PassState
        The pass state where the passes are executed on.

    Returns
    -------
    ret : PassState
        The updated pass state after running the provided passes.
    """
    return _opt.Optimize(passes, pass_state)
