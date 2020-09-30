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
# pylint: disable=invalid-name
"""Algebraic data type definitions."""
import tvm._ffi

from .type import Type
from .expr import RelayExpr
from . import _ffi_api


@tvm._ffi.register_object("relay.Constructor")
class Constructor(RelayExpr):
    """Relay ADT constructor.

    Parameters
    ----------
    name_hint : str
        Name of constructor (only a hint).

    inputs : List[Type]
        Input types.

    belong_to : GlobalTypeVar
        Denotes which ADT the constructor belongs to.
    """

    def __init__(self, name_hint, inputs, belong_to):
        self.__init_handle_by_constructor__(_ffi_api.Constructor, name_hint, inputs, belong_to)

    def __call__(self, *args):
        """Call the constructor.

        Parameters
        ----------
        args: List[RelayExpr]
            The arguments to the constructor.

        Returns
        -------
        call: RelayExpr
            A call to the constructor.
        """
        # pylint: disable=import-outside-toplevel
        from tvm import relay

        return relay.Call(self, args)


@tvm._ffi.register_object("relay.TypeData")
class TypeData(Type):
    """Stores the definition for an Algebraic Data Type (ADT) in Relay.

    Note that ADT definitions are treated as type-level functions because
    the type parameters need to be given for an instance of the ADT. Thus,
    any global type var that is an ADT header needs to be wrapped in a
    type call that passes in the type params.

    Parameters
    ----------
    header: GlobalTypeVar
        The name of the ADT.
        ADTs with the same constructors but different names are
        treated as different types.

    type_vars: List[TypeVar]
        Type variables that appear in constructors.

    constructors: List[Constructor]
        The constructors for the ADT.
    """

    def __init__(self, header, type_vars, constructors):
        self.__init_handle_by_constructor__(_ffi_api.TypeData, header, type_vars, constructors)
