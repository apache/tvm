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
"""IRModule that holds the functions and type definitions."""
from tvm._ffi.base import string_types
import tvm._ffi

from .base import Node
from . import expr as _expr
from . import type as _ty
from . import _ffi_api


@tvm._ffi.register_object("IRModule")
class IRModule(Node):
    """IRModule that holds functions and type definitions.

    IRModule is the basic unit for all IR transformations across the stack.

    Parameters
    ----------
    functions: Optional[dict].
        Map of global var to BaseFunc
    """

    def __init__(self, functions=None, type_definitions=None):
        if functions is None:
            functions = {}
        elif isinstance(functions, dict):
            mapped_funcs = {}
            for k, v in functions.items():
                if isinstance(k, string_types):
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
                if isinstance(k, string_types):
                    k = _ty.GlobalTypeVar(k)
                if not isinstance(k, _ty.GlobalTypeVar):
                    raise TypeError("Expect type_definitions to be Dict[GlobalTypeVar, Type]")
                mapped_type_defs[k] = v
            type_definitions = mapped_type_defs
        self.__init_handle_by_constructor__(_ffi_api.IRModule, functions, type_definitions)

    def __setitem__(self, var, val):
        """Add a mapping to the module.

        Parameters
        ---------
        var: GlobalVar
            The global variable.

        val: Union[Function, Type]
            The value.
        """
        return self._add(var, val, True)

    def _add(self, var, val, update=True):
        if isinstance(val, _expr.RelayExpr):
            if isinstance(var, string_types):
                if _ffi_api.Module_ContainGlobalVar(self, var):
                    var = _ffi_api.Module_GetGlobalVar(self, var)
                else:
                    var = _expr.GlobalVar(var)
            _ffi_api.Module_Add(self, var, val, update)
        else:
            assert isinstance(val, _ty.Type)
            if isinstance(var, string_types):
                var = _ty.GlobalTypeVar(var)
            _ffi_api.Module_AddDef(self, var, val, update)

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
        if isinstance(var, string_types):
            return _ffi_api.Module_Lookup_str(self, var)
        if isinstance(var, _expr.GlobalVar):
            return _ffi_api.Module_Lookup(self, var)
        return _ffi_api.Module_LookupDef(self, var)

    def update(self, other):
        """Insert functions in another Module to current one.

        Parameters
        ----------
        other: IRModule
            The module to merge into the current Module.
        """
        if isinstance(other, dict):
            other = IRModule(other)

        return _ffi_api.Module_Update(self, other)

    def update_func(self, var, func):
        """Update the function corresponding to a global variable in the
        module.

        Parameters
        ----------
        var: GlobalVar
            The global variable.

        func: tvm.relay.Function
            The function to be inserted.
        """
        return _ffi_api.Module_UpdateFunction(self, var, func)

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
        tvm.error.TVMError if we cannot find corresponding global var.
        """
        return _ffi_api.Module_GetGlobalVar(self, name)

    def get_global_vars(self):
        """Collect all global vars defined in this module.

        Returns
        -------
        global_vars: Array[GlobalVar]
            An array of global vars.
        """
        return _ffi_api.Module_GetGlobalVars(self)

    def get_global_type_vars(self):
        """Collect all global type vars defined in this module.

        Returns
        -------
        global_type_vars: Array[GlobalTypeVar]
            An array of global type vars.
        """
        return _ffi_api.Module_GetGlobalTypeVars(self)

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
        tvm.error.TVMError if we cannot find corresponding global type var.
        """
        return _ffi_api.Module_GetGlobalTypeVar(self, name)

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
        tvm.error.TVMError if the corresponding constructor cannot be found.
        """
        return _ffi_api.Module_LookupTag(self, tag)

    def get_type(self, name):
        ty_var = self.get_global_type_var(name)
        ty_data = self.type_definitions[ty_var]
        return tuple([ty_var] + list(ty_data.constructors))

    @staticmethod
    def from_expr(expr, functions=None, type_defs=None):
        """Construct a module from a standalone expression.

        Parameters
        ----------
        expr: RelayExpr
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
        return _ffi_api.Module_FromExpr(expr, funcs, defs)

    def _import(self, file_to_import):
        return _ffi_api.Module_Import(self, file_to_import)

    def import_from_std(self, file_to_import):
        # TODO(@jroesch): clean up prelude
        _ffi_api.Module_ImportFromStd(self, file_to_import)
        return tvm.relay.transform.InferType()(self)

    def __str__(self):
        return _ffi_api.PrettyPrint(self)

    def __repr__(self):
        return self.astext()

    def script(self, tir_prefix: str = "T", show_meta: bool = False) -> str:
        """Print IRModule into TVMScript

        Parameters
        ----------
        tir_prefix : str
            The tir namespace prefix

        show_meta : bool
            Whether to show meta information

        Returns
        -------
        script : str
            The TVM Script of the IRModule
        """
        return tvm._ffi.get_global_func("script.AsTVMScript")(
            self, tir_prefix, show_meta
        )  # type: ignore

    def get_attr(self, attr_key):
        """Get the IRModule attribute.

        Parameters
        ----------
        attr_key : str
            The attribute key.

        Returns
        -------
        attr_value : Any
            Attribute value
        """

        return _ffi_api.Module_GetAttr(self, attr_key)

    def with_attr(self, attr_key, attr_value):
        """Copy the IRModule and add an attribute to it.

        Parameters
        ----------
        attr_key : str
            The attribute key.

        attr_value : Object
            The new attribute value.

        Returns
        -------
        mod : IRModule
            A new copy of the IRModule with the attribute
        """

        return _ffi_api.Module_WithAttr(self, attr_key, attr_value)
