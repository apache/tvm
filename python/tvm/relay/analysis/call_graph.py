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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, unused-import
"""Call graph used in Relay."""

from ...ir import IRModule
from ...runtime import Object
from ..expr import GlobalVar
from . import _ffi_api


class CallGraph(Object):
    """Class to represent a call graph."""

    def __init__(self, module):
        """Construct a call graph.

        Parameters
        ----------
        module : tvm.ir.IRModule
            The IR module used to create a call graph

        Returns
        -------
        call_graph: CallGraph
            A constructed call graph.
        """
        self.__init_handle_by_constructor__(_ffi_api.CallGraph, module)

    @property
    def module(self):
        """Return the contained Relay IR module.

        Parameters
        ----------
        None

        Returns
        -------
        ret : tvm.ir.IRModule
            The contained IRModule
        """
        return _ffi_api.GetModule(self)

    def ref_count(self, var):
        """Return the number of references to the global var

        Parameters
        ----------
        var : Union[String, tvm.relay.GlobalVar]

        Returns
        -------
        ret : int
            The number reference to the global var
        """
        var = self._get_global_var(var)
        return _ffi_api.GetRefCountGlobalVar(self, var)

    def global_call_count(self, var):
        """Return the number of global function calls from a given global var.

        Parameters
        ----------
        var : Union[String, tvm.relay.GlobalVar]

        Returns
        -------
        ret : int
            The number of global function calls from the given var.
        """
        var = self._get_global_var(var)
        return _ffi_api.GetGlobalVarCallCount(self, var)

    def is_recursive(self, var):
        """Return if the function corresponding to a var is a recursive
        function.

        Parameters
        ----------
        var : Union[String, tvm.relay.GlobalVar]

        Returns
        -------
        ret : Boolean
            If the function corresponding to var is recurisve.
        """
        var = self._get_global_var(var)
        return _ffi_api.IsRecursive(self, var)

    def _get_global_var(self, var):
        """Return the global var using a given name or GlobalVar.

        Parameters
        ----------
        var : Union[String, tvm.relay.GlobalVar]

        Returns
        -------
        ret : tvm.relay.GlobalVar
            The global var.
        """
        if isinstance(var, str):
            mod = self.module
            var = mod.get_global_var(var)

        if isinstance(var, GlobalVar):
            return var
        else:
            raise TypeError("var should be either a string or GlobalVar")

    def print_var(self, var):
        """Print a call graph of a global function by name or by variable.

        Parameters
        ----------
        var: Union[String, tvm.relay.GlobalVar]
            The name or global variable.

        Returns
        -------
        ret : String
            The call graph represented in string.
        """
        var = self._get_global_var(var)
        return _ffi_api.PrintCallGraphGlobalVar(self, var)

    def __str__(self):
        """Print the call graph in the topological order."""
        return _ffi_api.PrintCallGraph(self)
