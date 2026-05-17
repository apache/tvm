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
"""Codegen registry for CUDA HW ops.

User-facing Python wrappers are hand-written in :mod:`tvm.tirx.op` so that
editors / static analyzers (Cursor, Pyright) can see their signatures. This
module only handles the backend codegen side.
"""

import functools

import tvm_ffi

CODEGEN_REGISTRY = {}
_CALL_EFFECT_KIND_OPAQUE = 4
_registered_attrs: set[str] = set()


@tvm_ffi.register_global_func("tirx.intrinsics.cuda.get_codegen")
def get_codegen(op):
    """get the codegen function for a given op"""
    return CODEGEN_REGISTRY.get(op, None)


def register_codegen(op, backend="cuda"):
    """Register a codegen function for a given op.

    The codegen function should return a ``cuda_func_call`` statement, and
    optionally a list of tags that the codegen function needs.
    """

    def decorator(func):
        full_op_name = "tirx." + op
        _ensure_op_registered(full_op_name)

        @functools.wraps(func)
        def wrapper(arg_list):
            res = func(*arg_list)  # pylint: disable=not-callable
            if isinstance(res, tuple):
                return res[0], res[1]
            return res, list()

        CODEGEN_REGISTRY[full_op_name] = wrapper
        return wrapper

    return decorator


def _ensure_op_registered(op_name: str) -> None:
    """Ensure dynamic TIRx ops also have a purity/effect attribute."""
    try:
        tvm_ffi.get_global_func("ir.RegisterOp")(op_name, "")
    except Exception:
        pass
    if op_name in _registered_attrs:
        return
    try:
        tvm_ffi.get_global_func("ir.RegisterOpAttr")(
            op_name, "TCallEffectKind", _CALL_EFFECT_KIND_OPAQUE, 10
        )
        _registered_attrs.add(op_name)
    except Exception:
        pass
