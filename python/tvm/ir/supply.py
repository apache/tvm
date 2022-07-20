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

    def fresh_name(self, name, add_prefix=True):
        return _ffi_api.NameSupply_FreshName(self, name, add_prefix)

    def reserve_name(self, name, add_prefix=True):
        return _ffi_api.NameSupply_ReserveName(self, name, add_prefix)

    def contains_name(self, name, add_prefix=True):
        return _ffi_api.NameSupply_ContainsName(self, name, add_prefix)

    def clear(self):
        return _ffi_api.NameSupply_Clear(self)


@tvm._ffi.register_object("GlobalVarSupply")
class GlobalVarSupply(Object):
    """GlobalVarSupply that holds a mapping between names and GlobalVars.

    GlobalVarSupply can be used to generate new GlobalVars with an unique name.
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
        return _ffi_api.GlobalVarSupply_FreshGlobal(self, name, add_prefix)

    def unique_global_for(self, name, add_prefix=True):
        return _ffi_api.GlobalVarSupply_UniqueGlobalFor(self, name, add_prefix)

    def reserve_global(self, global_var, allow_conflict=False):
        return _ffi_api.GlobalVarSupply_ReserveGlobalVar(self, global_var, allow_conflict)
