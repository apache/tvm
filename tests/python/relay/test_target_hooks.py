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
"""Unit tests for target hooks."""
import sys
import numpy as np
import pytest

from tvm import relay, IRModule

from utils.external_codegen import (
    set_external_func_attr,
    check_aot_executor_result,
    check_graph_executor_result,
)


@pytest.mark.parametrize("check_result", [check_aot_executor_result, check_graph_executor_result])
def test_tir_external_generation(check_result):
    shape = (8,)
    x_data = np.random.randint(255, size=shape).astype("float32")
    y_data = np.random.randint(255, size=shape).astype("float32")
    inputs = {"x": x_data, "y": y_data}

    x0 = relay.var("x0", shape=shape, dtype="float32")
    y0 = relay.var("y0", shape=shape, dtype="float32")
    z = x0 + y0
    f = relay.Function([x0, y0], z)
    f = set_external_func_attr(f, "example_target_hook", "replace_add_with_subtract")

    x = relay.var("x", shape=(8,), dtype="float32")
    y = relay.var("y", shape=(8,), dtype="float32")
    call = relay.Call(f, [x, y])
    func = IRModule.from_expr(call)

    check_result(func, inputs, (8,), x_data - y_data)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
