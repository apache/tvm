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
"""Utils for BYOC pattern matching"""

from tvm.relax import DataflowVar
from tvm.relax.transform import PatternCheckContext


def has_leaking_intermediate_variables(context: PatternCheckContext) -> bool:
    """
    Check whether intermediate variables in the region to be fused are used outside
    the fused region.
    """
    defined_vars = set(context.matched_bindings.keys())
    output_var = context.value_to_bound_var[context.matched_expr]
    intermediate_vars = {v for v in context.matched_bindings if v != output_var}

    if any(not isinstance(v, DataflowVar) for v in intermediate_vars):
        # If intermediate variable is not a DataflowVar, it can be accessed and potentially
        # used outside the DataflowBlock.
        return True

    # Check whether all users of an intermediate variable are inside the fused region.
    for var in intermediate_vars:
        if any(var_user not in defined_vars for var_user in context.var_usages[var]):
            return True

    return False
