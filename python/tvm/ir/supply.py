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
from tvm import Object
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


@tvm._ffi.register_object("GlobalVarSupply")
class GlobalVarSupply(Object):
    """GlobalVarSupply that holds a mapping between names and GlobalVars.

    GlobalVarSupply can be used to generate new GlobalVars with an unique name.
    It also can be used to retrieve previously generated GlobalVars based on a name.

    Parameters
    ----------
    name_supply: The NameSupply to be used by this GlobalVarSupply.
    """

    def __init__(self, name_supply=None):
        name_supply = name_supply if name_supply is not None else NameSupply("")
        self.__init_handle_by_constructor__(_ffi_api.GlobalVarSupply, name_supply)

    def fresh_global(self, name, add_prefix=True):
        return _ffi_api.GlobalVarSupply_FreshGlobal(self, name, add_prefix)

    def unique_global_for(self, name, add_prefix=True):
        return _ffi_api.GlobalVarSupply_UniqueGlobalFor(self, name, add_prefix)
