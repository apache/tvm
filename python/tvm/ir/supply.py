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
"""Suppliers that are used to guarantee uniqueness of names."""

import tvm_ffi

from tvm import Object

from . import _ffi_api


@tvm_ffi.register_object("ir.UniqueNameSupply")
class UniqueNameSupply(Object):
    """UniqueNameSupply that can be used to generate unique names.

    Parameters
    ----------
    prefix: The prefix to be added to the generated names.
    """

    def __init__(self, prefix=""):
        self.__init_handle_by_constructor__(_ffi_api.UniqueNameSupply, prefix)

    def fresh_name(self, name, add_prefix=True, add_underscore=True):
        """Generates a unique name from this UniqueNameSupply.

        Parameters
        ----------
        name: String
            The name from which the generated name is derived.

        add_prefix: bool
            If set to true, then the prefix of this UniqueNameSupply will be prepended to the name.

        add_underscore: bool
            If set to True, adds '_' between prefix and digit.
        """
        return _ffi_api.UniqueNameSupply_FreshName(self, name, add_prefix, add_underscore)

    def reserve_name(self, name, add_prefix=True):
        """Reserves an existing name with this UniqueNameSupply.

        Parameters
        ----------
        name: String
            The name to be reserved.

        add_prefix: bool
            If set to true, then the prefix of this UniqueNameSupply will be prepended to the name
            before reserving it.
        """
        return _ffi_api.UniqueNameSupply_ReserveName(self, name, add_prefix)

    def contains_name(self, name, add_prefix=True):
        """Checks if this UniqueNameSupply already generated a name.

        Parameters
        ----------
        name: String
            The name to check.

        add_prefix: bool
            If set to true, then the prefix of this UniqueNameSupply will be prepended to the name
            before checking for it.
        """
        return _ffi_api.UniqueNameSupply_ContainsName(self, name, add_prefix)
