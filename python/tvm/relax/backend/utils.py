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

from typing import Tuple
from tvm import relax
from tvm.relax import DataflowVar, PyExprMutator
from tvm.relax.transform import PatternCheckContext
from tvm.target import Target


class BackendDispatcher(PyExprMutator):
    """Base class for backend dispatcher"""

    def __init__(self, mod):
        super().__init__(mod)

    @staticmethod
    def is_gpu_target(target: Target) -> bool:
        """Check if the target is a GPU target."""
        return "gpu" in target.keys

    @staticmethod
    def get_shape_dtype(expr: relax.Expr) -> Tuple[relax.ShapeExpr, str]:
        """Get shape and dtype from an expression.
        If the shape and dtype is unknown, raise an error."""
        sinfo = expr.struct_info
        if not isinstance(expr.struct_info, relax.TensorStructInfo):
            raise ValueError(
                f"Expecting a expr with TensorStructInfo, but got {expr} with {expr.struct_info}"
            )

        shape, dtype = sinfo.shape, sinfo.dtype
        if shape is None:
            raise ValueError(
                f"Expecting a expr with known shape, but got {expr} with unknown shape"
            )

        return shape, dtype

    def _get_target(self, sinfo: relax.StructInfo) -> Target:
        # Get target information from TensorStructInfo
        if isinstance(sinfo, relax.TensorStructInfo):
            vdevice = sinfo.vdevice
            if vdevice is not None:
                return vdevice.target
        elif isinstance(sinfo, relax.TupleStructInfo):
            for f in sinfo.fields:
                tgt = self._get_target(f)
                if tgt != Target.current():
                    return tgt
        # Return the target in current context
        target = Target.current()
        if target is None:
            raise ValueError(
                "Target not found. Please ensure that the target is annotated within the module, "
                "or alternatively, execute this within a specified target context."
            )
        return target


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
