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
"""Bound deduction."""
from . import _ffi_api


def deduce_bound(var, cond, hint_map, relax_map):
    """Deduce the bound of the target variable in the cond.

    Parameters
    ----------
    var : Var
        The target variable to be deduced.

    cond : PrimExpr
        The condition

    hint_map : Map[Var, IntSet]
        Domain of variables used to help deduction.

    relax_map : Map[Var, IntSet]
        The fomain of the variables to be relaxed
        using the provided domain.
    """
    return _ffi_api.DeduceBound(var, cond, hint_map, relax_map)
