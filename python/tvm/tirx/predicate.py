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
# pylint: disable=no-member
"""Async structures for TIRX"""

import inspect
from collections.abc import Callable

from tvm_ffi import register_object

from tvm.runtime import Object
from tvm.tirx import PrimExpr, Var

from . import _ffi_api


@register_object("tirx.Predicate")
class Predicate(Object):
    """A predicate object for TIRX"""

    vars: list[Var]
    pred: PrimExpr

    def __init__(self, f_pred: Callable[..., PrimExpr]):
        vars = [Var(name, "int32") for name in inspect.signature(f_pred).parameters]
        pred = f_pred(*vars)
        self.__init_handle_by_constructor__(_ffi_api.Predicate, vars, pred)

    def apply(self, indices: list[PrimExpr]) -> PrimExpr:
        """Apply the predicate to the given indices"""
        return _ffi_api.PredicateApply(self, indices)
