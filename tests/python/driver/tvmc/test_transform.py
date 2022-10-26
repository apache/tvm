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
import pytest
import numpy as np

import tvm
from tvm import relay
from tvm.driver.tvmc.transform import convert_graph_layout


def test_layout_transform():
    """
    Test layout is correctly transformed and constant folding is applied.
    """
    dtype = "int8"
    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max

    x = relay.var("x", shape=(1, 4, 2, 2), dtype=dtype)
    weight = relay.const(
        np.random.randint(data_min, data_max, size=(2, 4, 2, 2), dtype=dtype), dtype=dtype
    )
    x = relay.nn.conv2d(x, weight)
    func = relay.Function(relay.analysis.free_vars(x), x)
    mod = tvm.IRModule.from_expr(func)

    desired_layout = "NHWC"
    mod = convert_graph_layout(mod, desired_layout)

    main_expr = mod["main"].body
    conv = main_expr.args[0]
    assert conv.op.name == "nn.conv2d"
    assert conv.attrs["data_layout"] == "NHWC"
    assert conv.attrs["kernel_layout"] == "HWIO"

    # Ensure transform has been folded into the constant
    weights = conv.args[1]
    assert isinstance(weights, relay.expr.Constant)


if __name__ == "__main__":
    tvm.testing.main()
