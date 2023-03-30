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

""" Tests strategy selection for Relay ops """
import pytest
import tvm
from tvm import relay
from tvm import te
from tvm.relay.testing import run_infer_type
import tvm.testing


@pytest.mark.parametrize(
    "target, expected_implementation",
    [("llvm", "concatenate.cpu"), ("llvm -device=arm_cpu", "concatenate.arm_cpu")],
)
def test_concatenate(target, expected_implementation):
    target = tvm.target.Target(target)

    shape = (1, 1, 1, 3)
    dtype = "float32"
    axis = 1
    inputs = []
    inputs.append(relay.var("var0", shape=shape, dtype=dtype))
    inputs.append(relay.var("var1", shape=shape, dtype=dtype))
    input_tuple = relay.Tuple(inputs)
    out = relay.op.concatenate(input_tuple, axis)
    out = run_infer_type(out)

    impl, xx = relay.backend.te_compiler.select_implementation(
        relay.op.get("concatenate"),
        out.attrs,
        [te.placeholder(shape)],
        out.checked_type,
        target,
        use_autotvm=False,
    )
    assert impl.name == expected_implementation


if __name__ == "__main__":
    tvm.testing.main()
