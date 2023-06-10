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
"""Suppliers that are used to guarantee uniqueness of names and GlobalVars."""
import tvm
from tvm import Object, IRModule
from . import _ffi_api


@tvm._ffi.register_object("NameSupply")
class NameSupply(Object):
    """NameSupply that can be used to generate unique names.

    Parameters
    ----------
    prefix: The prefix to be added to the generated names.
    """

    def __init__(self, prefix=""):
        self.__init_handle_by_constructor__(_ffi_api.NameSupply, prefix)

    def fresh_name(self, name, add_prefix=True, add_underscore=True):
        """Generates a unique name from this NameSupply.

        Parameters
        ----------
        name: String
            The name from which the generated name is derived.

        add_prefix: bool
            If set to true, then the prefix of this NameSupply will be prepended to the name.

        add_underscore: bool
            If set to True, adds '_' between prefix and digit.
        """
        return _ffi_api.NameSupply_FreshName(self, name, add_prefix, add_underscore)

    def reserve_name(self, name, add_prefix=True):
        """Reserves an existing name with this NameSupply.

        Parameters
        ----------
        name: String
            The name to be reserved.

        add_prefix: bool
            If set to true, then the prefix of this NameSupply will be prepended to the name
            before reserving it.
        """
        return _ffi_api.NameSupply_ReserveName(self, name, add_prefix)

    def contains_name(self, name, add_prefix=True):
        """Checks if this NameSupply already generated a name.

        Parameters
        ----------
        name: String
            The name to check.

        add_prefix: bool
            If set to true, then the prefix of this NameSupply will be prepended to the name
            before checking for it.
        """
        return _ffi_api.NameSupply_ContainsName(self, name, add_prefix)


@tvm._ffi.register_object("GlobalVarSupply")
class GlobalVarSupply(Object):
    """GlobalVarSupply that holds a mapping between names and GlobalVars.

    GlobalVarSupply can be used to generate new GlobalVars with a unique name.
    It also can be used to retrieve previously generated GlobalVars based on a name.

    Parameters
    ----------
    value: Union[List[IRModule], IRModule, NameSupply]
        The IRModules used to build this GlobalVarSupply or a NameSupply.
    """

    def __init__(self, value=None):
        if value is None:
            name_supply = NameSupply("")
            self.__init_handle_by_constructor__(_ffi_api.GlobalVarSupply_NameSupply, name_supply)
        elif isinstance(value, NameSupply):
            self.__init_handle_by_constructor__(_ffi_api.GlobalVarSupply_NameSupply, value)
        elif isinstance(value, (list, tvm.container.Array)):
            self.__init_handle_by_constructor__(_ffi_api.GlobalVarSupply_IRModules, value)
        elif isinstance(value, IRModule):
            self.__init_handle_by_constructor__(_ffi_api.GlobalVarSupply_IRModule, value)

    def fresh_global(self, name, add_prefix=True):
        """Generates a unique GlobalVar from this supply.

        Parameters
        ----------
        name: String
            The name from which the name of the GlobalVar is derived.

        add_prefix: bool
            If set to true, then the prefix of the contained NameSupply will be prepended
            to the name.
        """
        return _ffi_api.GlobalVarSupply_FreshGlobal(self, name, add_prefix)

    def unique_global_for(self, name, add_prefix=True):
        """Looks up for a GlobalVar with the given name in this supply. If no entry is found
        , creates one, places it in the cache and returns it.

        Parameters
        ----------
        name: String
            The name of the GlobalVar to search for.

        add_prefix: bool
            If set to true, the prefix of the contained NameSupply will be prepended to the
            name before performing the search.
        """
        return _ffi_api.GlobalVarSupply_UniqueGlobalFor(self, name, add_prefix)

    def reserve_global(self, global_var, allow_conflict=False):
        """Reserves an existing GlobalVar with this supply.

        Parameters
        ----------
        global_var: GlobalVar
            The GlobalVar to be registered.

        allow_conflict: bool
            Allow conflict with other GlobalVars that have the same name
        """
        return _ffi_api.GlobalVarSupply_ReserveGlobalVar(self, global_var, allow_conflict)
