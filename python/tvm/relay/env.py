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

    def __init__(self, funcs=None):
        """Construct an environment.

        Parameters
        ------
        funcs : optional, dict
            Map of global var to Function

        Returns
        ------
        env: A new environment containing :py:class:`~relay.env.Environment`.
        """
        funcs = funcs if funcs else {}
        self.__init_handle_by_constructor__(_make.Environment, funcs)

    def add(self, var, func):
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

    def global_var(self, name):
        """Get a global variable by name.

        Parameters
        ----------
        name: str
            The name of the global variable.

        Returns
        -------
            global_var: GlobalVar
                The global variable mapped to :code:`name`.
        """
        return _env.Environment_GetGlobalVar(self, name)

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
        if isinstance(var, str):
            return _env.Environment_Lookup_str(self, var)
        else:
            return _env.Environment_Lookup(self, var)
