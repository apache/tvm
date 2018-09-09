# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable, wildcard-import
"""A global environment storing everything needed to interpret or compile a Relay program."""
from .base import register_relay_node, NodeBase
from . import _make
from . import _env

@register_relay_node
class Environment(NodeBase):
    """The global Relay environment containing functions,
    options and more.
    """
    def __init__(self, funcs) -> None:
        """Construct an environment.

        Parameters
        ------
        funcs: list of relay.Function

        Returns
        ------
        env: A new environment containing :py:class:`~relay.env.Environment`.
        """
        self.__init_handle_by_constructor__(_make.Environment, funcs)

    def add(self, var, func) -> None:
        """Add a function to the environment.

        Parameters
        ---------
        var: GlobalVar
            The global variable which names the function.

        func: Function
            The function.
        """
        if isinstance(var, str):
            var = _env.Environment_GetGlobalVar(self, var)

        _env.Environment_Add(self, var, func)

    def merge(self, other):
        """Merge two environments.

        Parameters
        ----------
        other: Environment
            The environment to merge into the current Environment.
        """
        return _env.Environment_Merge(self, other)

    def global_var(self, var):
        """Get a global variable by name."""
        return _env.Environment_GetGlobalVar(self, var)

    def lookup(self, var):
        """Lookup a global function by name or by variable."""
        if isinstance(var, str):
            return _env.Environment_Lookup_str(self, var)
        else:
            return _env.Environment_Lookup(self, var)

    def transform(self, transformer):
        """Apply a transformer function to the environment."""
        _env.Environment_Transform(self, transformer)
