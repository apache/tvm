# pylint: disable=no-else-return
# pylint: disable=unidiomatic-typecheck
"""The pass manager for Relay.

This file exposes differen granularity of interfaces for users to implement and
use passes more conveniently.
"""
from abc import abstractmethod
from enum import IntEnum

from . import _ir_pass
from .base import RelayNode, register_relay_node


class PassKind(IntEnum):
    """The different granularity of passes for optimization/analysis."""
    ModuleKind = 1
    FunctionKind = 2
    SequentialKind = 3


@register_relay_node
class PassContext(RelayNode):
    """The basis where a Relay optimization/analysis runs on.
    Each pass context contains a number of auxiliary information that is used
    to help an optimization pass. Such information includes the error reporter
    to record the errors of during the performing the optimization, etc.
    """

    def __init__(self):
        self.__init_handle_by_constructor__(_ir_pass.PassContext)


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
    """

    @abstractmethod
    def run(self, mod):
        """Execute the pass. It is an abstract function that will be
        implemented by subclasses.

        Parameters
        ----------
        mod : tvm.relay.Module
            The module that a certain optimization is performed on.

        Returns
        -------
        mod : tvm.relay.Module
            The updated module after applying this pass.
        """
        raise NotImplementedError("Pure virtual function is not implemented.")

    def __call__(self, mod):
        return self.run(mod)


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
        The curried callback that sketches a certain optimization.

    enabled : bool
        The flag indicates if a pass is enabled.
    """

    def __init__(self, name, opt_level, pass_func, enabled=True):
        self.__init_handle_by_constructor__(_ir_pass.CreateModulePass, name,
                                            opt_level, pass_func, enabled)

    def run(self, mod):
        """Execute a module pass.

        Parameters
        ----------
        mod : tvm.relay.Module
            The module that the module pass is executed on.

        Returns
        -------
        ret : tvm.relay.Module
            The updated module.
        """
        return _ir_pass.RunModulePass(self, mod)

    def __call__(self, mod):
        return self.run(mod)

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
        The curried callback that sketches a certain optimization.

    enabled : bool
        The flag indicates if a pass is enabled.
    """

    def __init__(self, name, opt_level, pass_func, enabled=True):
        self.__init_handle_by_constructor__(_ir_pass.CreateFunctionPass, name,
                                            opt_level, pass_func, enabled)

    def run(self, mod):
        """Execute a function pass.

        Parameters
        ----------
        mod : tvm.relay.Module
            The module that the function pass is executed on.

        Returns
        -------
        ret : tvm.relay.Module
            The updated module.
        """
        return _ir_pass.RunFunctionPass(self, mod)

    def __call__(self, mod):
        return self.run(mod)

@register_relay_node
class SequentialPass(Pass):
    """A pass that works on each tvm.relay.Function in a module.

    Parameters
    ----------
    name : str
        The pass name.

    opt_level : int
        The optimization level of this pass.

    passes : Map[Pass, bool]
        The mapping of pass and bool that indicates if a pass is enabled.
    """

    def __init__(self, name, opt_level, passes):
        self.__init_handle_by_constructor__(_ir_pass.CreateSequentialPass,
                                            name, opt_level, passes)

    def run(self, mod):
        """Execute a sequence of passes.

        Parameters
        ----------
        mod : tvm.relay.Module
            The module that the function pass is executed on.

        Returns
        -------
        ret : tvm.relay.Module
            The updated module.
        """
        return _ir_pass.RunSequentialPass(self, mod)

    def __call__(self, mod):
        return self.run(mod)


def create_pass(pass_name, opt_level,
                pass_kind=PassKind.FunctionKind,
                pass_func=None, sequential_passes=None,
                enabled=True):
    """Create a pass using a defined optimization function from Python.

    Parameters
    ----------
    pass_name : str
        The name of the pass.

    opt_level : int
        The optimization level of this pass.

    pass_kind : Optional[PassKind]
        The type of pass for optimization/analysis.

    pass_func : Optional[Callable[PassContext: Module/Function/Expr ->
                Module/Function/Expr]]
        The implemented optimization pass.

    sequential_passes : Optional[List[Pass]]
        A sequence of passes candidate for optimization.

    enabled : Optional[bool]
        The flag indicates if a pass is enabled.

    Returns
    -------
    ret : Pass
        The pass built through pass_func.
    """
    if not isinstance(pass_kind, PassKind):
        raise TypeError("pass_kind is expected to be the type of PassKind.")

    if pass_kind == PassKind.ModuleKind:
        if not pass_func:
            raise TypeError("pass_func must be defined for Module pass")
        return _ir_pass.CreateModulePass(pass_name, opt_level, pass_func,
                                         enabled)
    elif pass_kind == PassKind.FunctionKind:
        if not pass_func:
            raise TypeError("pass_func must be defined for Function pass")
        return _ir_pass.CreateFunctionPass(pass_name, opt_level, pass_func,
                                           enabled)
    else:
        if not isinstance(sequential_passes, (list, tuple)):
            raise TypeError(
                "sequential_passes must be a list of Pass objects.")
        return _ir_pass.CreateSequentialPass(pass_name, opt_level,
                                             sequential_passes)
