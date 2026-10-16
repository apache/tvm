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

User-facing Python wrappers are hand-written in :mod:`tvm_tirx.tirx.op` so that
editors / static analyzers (Cursor, Pyright) can see their signatures. This
module only handles the backend codegen side.
"""

import functools

import tvm_ffi

CODEGEN_REGISTRY = {}


@tvm_ffi.register_global_func("tirx.intrinsics.cuda.get_codegen")
def get_codegen(op):
    """get the codegen function for a given op"""
    return CODEGEN_REGISTRY.get(op, None)


def register_codegen(op, backend="cuda"):
    """Register a codegen function for a given op.

    The codegen function should return a ``func_call`` statement, and
    optionally a list of tags that the codegen function needs.
    """

    def decorator(func):
        full_op_name = "tirx." + op

        @functools.wraps(func)
        def wrapper(arg_list, attrs=None):
            # Keep operand-only helpers callable directly while forwarding the
            # typed static options of an attributed Call to its lowering.
            kwargs = {} if attrs is None else {"attrs": attrs}
            res = func(*arg_list, **kwargs)  # pylint: disable=not-callable
            if isinstance(res, tuple):
                return res[0], res[1]
            return res, list()

        CODEGEN_REGISTRY[full_op_name] = wrapper
        return wrapper

    return decorator
