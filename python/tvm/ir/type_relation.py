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
"""Type relation and function for type checking."""
import tvm._ffi

from .type import Type, TypeConstraint
from . import _ffi_api


@tvm._ffi.register_object("TypeCall")
class TypeCall(Type):
    """Type function application.

    Parameters
    ----------
    func: tvm.ir.Type
        The function.

    args: List[tvm.ir.Type]
        The arguments.

    Returns
    -------
    type_call: TypeCall
        The type function application.
    """

    def __init__(self, func, args):
        self.__init_handle_by_constructor__(_ffi_api.TypeCall, func, args)


@tvm._ffi.register_object("TypeRelation")
class TypeRelation(TypeConstraint):
    """User defined type relation, it is an input-output relation on types.

    TypeRelation is more generalized than TypeCall as it allows inference
     of both inputs and outputs.

    Parameters
    ----------
    func : EnvFunc
        User defined relation function.

    args : [tvm.ir.Type]
        List of types to the func.

    num_inputs : int
        Number of input arguments in args,
        this act as a hint for type inference.

    attrs : Attrs
        The attribute attached to the relation information

    Returns
    -------
    type_relation : tvm.ir.TypeRelation
        The type relation.
    """

    def __init__(self, func, args, num_inputs, attrs):
        self.__init_handle_by_constructor__(_ffi_api.TypeRelation, func, args, num_inputs, attrs)
