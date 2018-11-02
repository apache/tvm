# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable, wildcard-import
"""A global module storing everything needed to interpret or compile a Relay program."""
from .base import register_relay_node, RelayNode
from .._ffi import base as _base
from . import _make
from . import _module
from . import expr as _expr


@register_relay_node
class Module(RelayNode):
    """The global Relay module containing collection of functions.

    Each global function is identified by an unique tvm.relay.GlobalVar.
    tvm.relay.GlobalVar and Module is necessary in order to enable
    recursions in function to avoid cyclic reference in the function.x

    Parameters
    ----------
    functions : dict, optional.
        Map of global var to Function
    """
    def __init__(self, functions=None):
        if functions is None:
            functions = {}
        elif isinstance(functions, dict):
            mapped_funcs = {}
            for k, v in functions.items():
                if isinstance(k, _base.string_types):
                    k = _expr.GlobalVar(k)
                if not isinstance(k, _expr.GlobalVar):
                    raise TypeError("Expect functions to be Dict[GlobalVar, Function]")
                mapped_funcs[k] = v
            functions = mapped_funcs
        self.__init_handle_by_constructor__(_make.Module, functions)

    def __setitem__(self, var, func):
        """Add a function to the module.

        Parameters
        ---------
        var: GlobalVar
            The global variable which names the function.

        func: Function
            The function.
        """
        return self._add(var, func)

    def _add(self, var, func, update=False):
        if isinstance(var, _base.string_types):
            var = _expr.GlobalVar(var)
        return _module.Module_Add(self, var, func, update)

    def __getitem__(self, var):
        """Lookup a global function by name or by variable.

        Parameters
        ----------
        var: str or GlobalVar
            The name or global variable.

        Returns
        -------
            func: Function
                The function referenced by :code:`var`.
        """
        if isinstance(var, _base.string_types):
            return _module.Module_Lookup_str(self, var)
        else:
            return _module.Module_Lookup(self, var)

    def update(self, other):
        """Insert functions in another Module to current one.

        Parameters
        ----------
        other: Module
            The module to merge into the current Module.
        """
        if isinstance(other, dict):
            other = Module(other)
        return _module.Module_Update(self, other)

    def get_global_var(self, name):
        """Get a global variable in the function by name.

        Parameters
        ----------
        name: str
            The name of the global variable.

        Returns
        -------
        global_var: GlobalVar
            The global variable mapped to :code:`name`.

        Raises
        ------
        tvm.TVMError if we cannot find corresponding global var.
        """
        return _module.Module_GetGlobalVar(self, name)
