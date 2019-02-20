# pylint: disable=no-else-return
# pylint: disable=unidiomatic-typecheck
"""The pass manager for Relay.

This file exposes differen granularity of interfaces for users to implement and
use passes more conveniently.
"""
from abc import abstractmethod
from enum import IntEnum

from . import _pass_manager as _pass_mgr
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
        self.__init_handle_by_constructor__(_pass_mgr.PassContext)


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

    def set_pass_context(self, pass_ctx):
        """Setup the pass context for analysis and optimizations. This context
        could be shared by different passes for sequential passes.

        Parameters
        ----------
        pass_ctx : PassContext
            The context that is used to help perform a certain pass or a series
            of passes.

        Returns
        -------
        pass : Pass
            The updated pass.
        """
        if not isinstance(pass_ctx, PassContext):
            raise TypeError("pas_ctx is expected to be the PassContext type")
        return _pass_mgr.SetPassContext(self, pass_ctx)

    def __call__(self, mod, pass_ctx=None):
        return self.run(mod, pass_ctx)


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
        self.__init_handle_by_constructor__(_pass_mgr.ModulePass, name, opt_level,
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
        return _pass_mgr.RunModulePass(self, mod, pass_ctx)


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
        self.__init_handle_by_constructor__(_pass_mgr.FunctionPass, name, opt_level,
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
        return _pass_mgr.RunFunctionPass(self, mod, pass_ctx)


@register_relay_node
class SequentialPass(Pass):
    """A pass that works on each tvm.relay.Function in a module.

    Parameters
    ----------
    name : str
        The pass name.

    opt_level : int
        The optimization level of this pass.

    passes : List[Pass]
        The pass candidates for optimization.

    enabled : bool
        The flag indicates if a pass is enabled.

    required_passes : List[str]
        The list of dependent passes to perform this pass.
    """

    def __init__(self, name, opt_level, passes, enabled=True,
                 required_passes=None):
        required_passes = required_passes if required_passes else []
        if not isinstance(required_passes, list):
            raise TypeError("required_passes is expected to be a list of str.")
        self.__init_handle_by_constructor__(_pass_mgr.SequentialPass, name, opt_level,
                                            passes, enabled, required_passes)

    def run(self, mod, pass_ctx=None):
        """Execute a sequence of passes.

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
        return _pass_mgr.RunSequentialPass(self, mod, pass_ctx)


def build_pass(pass_name, opt_level,
               pass_kind=PassKind.FunctionKind,
               pass_func=None, sequential_passes=None,
               enabled=True, required_passes=None):
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

    required_passes : Optional[List[str]]
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
        if not pass_func:
            raise TypeError("pass_func must be defined for Module pass")
        return _pass_mgr.ModulePass(pass_name, opt_level, pass_func, enabled,
                                    required_passes)
    elif pass_kind == PassKind.FunctionKind:
        if not pass_func:
            raise TypeError("pass_func must be defined for Function pass")
        return _pass_mgr.FunctionPass(pass_name, opt_level, pass_func, enabled,
                                      required_passes)
    else:
        if not isinstance(sequential_passes, (list, tuple)):
            raise TypeError("sequential_passes must be a list of Pass objects.")
        return _pass_mgr.SequentialPass(pass_name, opt_level,
                                        sequential_passes, enabled,
                                        required_passes)
