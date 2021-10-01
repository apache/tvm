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

"""CMSIS-NN functions for testing networks"""

import platform

import numpy as np
import pytest

import tvm
from tvm import relay


def skip_if_no_reference_system(func):
    return pytest.mark.skipif(
        platform.machine() == "i686", reason="Reference system unavailable in i386 container"
    )(func)


def count_num_calls(mod):
    """Count number of CallNode in the IRModule"""

    class CallCounter(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.count = 0

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op):
                self.count += 1

            super().visit_call(call)

    counter = CallCounter()
    for var in mod.get_global_vars():
        counter.visit(mod[var.name_hint])
    return counter.count


def get_range_for_dtype_str(dtype):
    """
    Produce the min,max for a give data type.

    Parameters
    ----------
    dtype : str
        a type string (e.g., int8)

    Returns
    -------
    type_info.min : int
        the minimum of the range
    type_info.max : int
        the maximum of the range
    """

    try:
        type_info = np.iinfo(dtype)
    except ValueError:
        type_info = np.finfo(dtype)
    return type_info.min, type_info.max


def make_module(func):
    """Create IRModule from Function"""
    func = relay.Function(relay.analysis.free_vars(func), func)
    mod = tvm.IRModule.from_expr(func)
    return relay.transform.InferType()(mod)
