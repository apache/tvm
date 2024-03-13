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
import tvm.testing

from tvm import relax
from tvm.script import relax as R
from tvm.relax.dpl import make_fused_bias_activation_pattern, is_op, wildcard
from tvm.contrib.pickle_memoize import memoize


@tvm.script.ir_module
class Conv2dResidualBlock:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), "float32"),
        weight1: R.Tensor((64, 64, 3, 3), "float32"),
        weight2: R.Tensor((64, 64, 3, 3), "float32"),
    ):
        with R.dataflow():
            conv1 = relax.op.nn.relu(relax.op.nn.conv2d(data, weight1, padding=(1, 1)))
            conv2 = relax.op.nn.relu(relax.op.nn.conv2d(conv1, weight2, padding=(1, 1)))
            out = relax.op.add(conv2, data)
            R.output(out)

        return out


has_tensorrt = tvm.get_global_func("relax.ext.tensorrt", True)

tensorrt_enabled = pytest.mark.skipif(
    not has_tensorrt,
    reason="TENSORRT not enabled.",
)

pytestmark = [tensorrt_enabled]


def build_and_run(mod, inputs_np, target, legalize=False):
    dev = tvm.device(target, 0)
    with tvm.transform.PassContext(config={"relax.transform.apply_legalize_ops": legalize}):
        ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, dev)
    f = vm["main"]
    inputs = [tvm.nd.array(inp, dev) for inp in inputs_np]
    return f(*inputs).numpy()


def test_tensorrt_offload():
    @memoize("relax.tests.test_codegen_tensorrt.conv2d_residual")
    def get_ref():
        data_np = np.random.randn(1, 64, 56, 56).astype("float32")
        weight1_np = np.random.randn(64, 64, 3, 3).astype("float32")
        weight2_np = np.random.randn(64, 64, 3, 3).astype("float32")
        inputs = [data_np, weight1_np, weight2_np]
        ref = build_and_run(Conv2dResidualBlock, inputs, "llvm", legalize=True)
        return inputs, ref

    inputs, ref = get_ref()

    conv_pat = make_fused_bias_activation_pattern(
        "relax.nn.conv2d", with_bias=False, activation=None
    )
    relu_pat = is_op("relax.nn.relu")(wildcard())
    add_pat = is_op("relax.add")(wildcard(), wildcard())

    patterns = [
        ("tensorrt.nn.conv2d", conv_pat),
        ("tensorrt.nn.relu", relu_pat),
        ("tensorrt.add", add_pat),
    ]

    params_np = {"weight1": inputs[1], "weight2": inputs[2]}

    mod = tvm.transform.Sequential(
        [
            relax.transform.BindParams("main", params_np),
            relax.transform.FuseOpsByPattern(patterns),
            relax.transform.MergeCompositeFunctions(),
            relax.transform.RunCodegen(),
        ]
    )(Conv2dResidualBlock)

    out = build_and_run(mod, inputs[:1], "cuda")

    tvm.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_tensorrt_offload()
