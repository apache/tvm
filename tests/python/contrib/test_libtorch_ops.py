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

import tvm.relay
from tvm.relay.op.contrib import torchop
from tvm.testing import requires_libtorch

import_torch_error = None

try:
    import torch
except ImportError as e:
    torch = None
    import_torch_error = str(e)


@pytest.mark.skipif(torch is None, reason=f"PyTorch is not available: {import_torch_error}")
@requires_libtorch
def test_backend():
    @torch.jit.script
    def script_fn(x, y):
        res = x * y
        return res

    for torch_dt, dt in (
        (torch.int32, "int32"),
        (torch.float32, "float32"),
        (torch.float64, "float64"),
    ):
        x2 = tvm.relay.var("x", shape=[1, 2], dtype=dt)
        y2 = tvm.relay.var("y", shape=[2, 2], dtype=dt)

        x3 = tvm.relay.var("x", shape=[1, 3], dtype=dt)
        y3 = tvm.relay.var("y", shape=[3, 3], dtype=dt)

        test_body = tvm.relay.sum(torchop(script_fn, x2, y2)) + tvm.relay.sum(
            torchop(script_fn, x3, y3)
        )
        test_fn = tvm.relay.Function([x2, y2, x3, y3], test_body)
        mod = tvm.IRModule({"main": test_fn})

        tvm.relay.transform.InferType()(mod)

        # mod = tvm.relay.transform.AnnotateTarget("target.torch")(mod)
        mod = tvm.relay.transform.MergeCompilerRegions()(mod)
        mod = tvm.relay.transform.PartitionGraph()(mod)
        mod = tvm.relay.transform.InferType()(mod)

        target = "llvm"
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.relay.build(mod, target, params={})

        ctx = tvm.cpu(0)
        rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](ctx))

        # int does not have randn, so we cast...
        x2t = torch.randn(1, 2).to(dtype=torch_dt)
        y2t = torch.randn(2, 2).to(dtype=torch_dt)
        x3t = torch.randn(1, 3).to(dtype=torch_dt)
        y3t = torch.randn(3, 3).to(dtype=torch_dt)
        # Set inputs
        rt_mod.set_input(0, x2t)
        rt_mod.set_input(1, y2t)
        rt_mod.set_input(2, x3t)
        rt_mod.set_input(3, y3t)
        # Execute
        rt_mod.run()
        # Get outputs
        tvm_output = rt_mod.get_output(0).numpy()
        expected = (script_fn(x2t, y2t).sum() + script_fn(x3t, y3t).sum()).numpy()
        print(tvm_output.dtype)
        print(expected.dtype)
        tvm.testing.assert_allclose(tvm_output, expected)


if __name__ == "__main__":
    tvm.testing.main()
