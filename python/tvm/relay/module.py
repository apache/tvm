# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable, wildcard-import
"""A global module storing everything needed to interpret or compile a Relay program."""
from .base import register_relay_node, RelayNode
from .._ffi import base as _base
from . import _make
from . import _module
from . import expr as _expr
from . import ty as _ty

@register_relay_node
class Module(RelayNode):
    """The global Relay module containing collection of functions.

    Each global function is identified by an unique tvm.relay.GlobalVar.
    tvm.relay.GlobalVar and Module is necessary in order to enable
    recursions in function to avoid cyclic reference in the function.x

    Parameters
    ----------
    functions: Optional[dict].
        Map of global var to Function
    """
    def __init__(self, functions=None, type_definitions=None):
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
        if type_definitions is None:
            type_definitions = {}
        elif isinstance(type_definitions, dict):
            mapped_type_defs = {}
            for k, v in type_definitions.items():
                if isinstance(k, _base.string_types):
                    k = _ty.GlobalTypeVar(k)
                if not isinstance(k, _ty.GlobalTypeVar):
                    raise TypeError("Expect type_definitions to be Dict[GlobalTypeVar, Type]")
                mapped_type_defs[k] = v
            type_definitions = mapped_type_defs
        self.__init_handle_by_constructor__(_make.Module, functions, type_definitions)


    def __setitem__(self, var, val):
        """Add a mapping to the module.

        Parameters
        ---------
        var: GlobalVar
            The global variable.

        val: Union[Function, Type]
            The value.
        """
        return self._add(var, val)

    def _add(self, var, val, update=False):
        if isinstance(val, _expr.Expr):
            if isinstance(var, _base.string_types):
                if _module.Module_ContainGlobalVar(self, var):
                    var = _module.Module_GetGlobalVar(self, var)
                else:
                    var = _expr.GlobalVar(var)
            _module.Module_Add(self, var, val, update)
        else:
            assert isinstance(val, _ty.Type)
            if isinstance(var, _base.string_types):
                var = _ty.GlobalTypeVar(var)
            _module.Module_AddDef(self, var, val)

    def __getitem__(self, var):
        """Lookup a global definition by name or by variable.

        Parameters
        ----------
        var: Union[String, GlobalVar, GlobalTypeVar]
            The name or global variable.

        Returns
        -------
        val: Union[Function, Type]
            The definition referenced by :code:`var` (either a function or type).
        """
        if isinstance(var, _base.string_types):
            return _module.Module_Lookup_str(self, var)
        elif isinstance(var, _expr.GlobalVar):
            return _module.Module_Lookup(self, var)
        else:
            return _module.Module_LookupDef(self, var)

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

    def get_global_type_var(self, name):
        """Get a global type variable in the function by name.

        Parameters
        ----------
        name: str
            The name of the global type variable.

        Returns
        -------
        global_type_var: GlobalTypeVar
            The global variable mapped to :code:`name`.

        Raises
        ------
        tvm.TVMError if we cannot find corresponding global type var.
        """
        return _module.Module_GetGlobalTypeVar(self, name)

    def get_constructor(self, tag):
        """Look up an ADT constructor by tag.

        Parameters
        ----------
        tag: int
            The tag for a constructor.

        Returns
        -------
        constructor: Constructor
           The constructor associated with the given tag,

        Raises
        ------
        tvm.TVMError if the corresponding constructor cannot be found.
        """
        return _module.Module_LookupTag(self, tag)

    @staticmethod
    def from_expr(expr, functions=None, type_defs=None):
        """Construct a module from a standalone expression.

        Parameters
        ----------
        expr: Expr
            The starting expression
        global_funcs: Optional[dict]
            Map of global vars to function definitions
        type_defs: Optional[dict]
            Map of global type vars to type definitions


        Returns
        -------
        mod: Module
            A module containing the passed definitions,
            where expr is set as the entry point
            (wrapped in a function if necessary)
        """
        funcs = functions if functions is not None else {}
        defs = type_defs if type_defs is not None else {}
        return _module.Module_FromExpr(expr, funcs, defs)
