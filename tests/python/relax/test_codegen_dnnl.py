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

from tvm import relax, relay
from tvm.script import relax as R
from tvm.relax.dpl import make_fused_bias_activation_pattern


def get_relay_conv2d_relu_x2(d_shape, w_shape):
    data = relay.var("data", shape=d_shape)
    weight1 = relay.var("weight1", shape=w_shape)
    weight2 = relay.var("weight2", shape=w_shape)
    conv1 = relay.nn.relu(
        relay.nn.conv2d(
            data=data,
            weight=weight1,
            kernel_size=w_shape[2:],
            padding=(1, 1),
        )
    )
    return relay.nn.relu(
        relay.nn.conv2d(
            data=conv1,
            weight=weight2,
            kernel_size=w_shape[2:],
            padding=(0, 0),
        )
    )


@tvm.script.ir_module
class Conv2dReLUx2:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), "float32"),
        weight1: R.Tensor((64, 64, 3, 3), "float32"),
        weight2: R.Tensor((64, 64, 3, 3), "float32"),
    ):
        with R.dataflow():
            conv1 = relax.op.nn.relu(relax.op.nn.conv2d(data, weight1, padding=(1, 1)))
            conv2 = relax.op.nn.relu(relax.op.nn.conv2d(conv1, weight2, padding=(0, 0)))
            R.output(conv2)

        return conv2


has_dnnl = tvm.get_global_func("relax.ext.dnnl", True)

dnnl_enabled = pytest.mark.skipif(
    not has_dnnl,
    reason="DNNL note enabled.",
)

pytestmark = [dnnl_enabled]


def test_dnnl_offload():
    pat = make_fused_bias_activation_pattern(
        "relax.nn.conv2d", with_bias=False, activation="relax.nn.relu"
    )

    seq = tvm.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern([("dnnl.conv2d_relu", pat)]),
            relax.transform.MergeCompositeFunctions(),
            relax.transform.RunCodegen(),
        ]
    )

    mod = seq(Conv2dReLUx2)

    target = tvm.target.Target("llvm")
    ex = relax.vm.build(mod, target)

    vm = relax.VirtualMachine(ex, tvm.cpu())
    f = vm["main"]

    data_np = np.random.randn(1, 64, 56, 56).astype("float32")
    weight1_np = np.random.randn(64, 64, 3, 3).astype("float32")
    weight2_np = np.random.randn(64, 64, 3, 3).astype("float32")
    out = f(tvm.nd.array(data_np), tvm.nd.array(weight1_np), tvm.nd.array(weight2_np)).numpy()

    relay_mod = tvm.IRModule.from_expr(get_relay_conv2d_relu_x2(data_np.shape, weight1_np.shape))

    ref = (
        relay.create_executor("graph", mod=relay_mod, device=tvm.cpu(0), target="llvm")
        .evaluate()(*[data_np, weight1_np, weight2_np])
        .numpy()
    )

    tvm.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)

    profiler_vm = relax.VirtualMachine(ex, tvm.cpu(), profile=True)
    report = profiler_vm.profile(
        "main", tvm.nd.array(data_np), tvm.nd.array(weight1_np), tvm.nd.array(weight2_np)
    )

    print(report)


if __name__ == "__main__":
    test_dnnl_offload()
