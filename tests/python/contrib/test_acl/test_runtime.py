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
"""ACL runtime tests."""

import numpy as np

import tvm
from tvm import relay

from .infrastructure import skip_runtime_test, build_and_run, verify
from .infrastructure import Device


def test_multiple_ops():
    """
    Test multiple operators destined for acl.
    ACL will expect these ops as in 2 separate functions.
    """
    if skip_runtime_test():
        return

    device = Device()

    def get_model(input_shape, var_names):
        """Return a model and any parameters it may have."""
        a = relay.var(next(var_names), shape=input_shape, dtype="float32")
        out = relay.reshape(a, (1, 1, 1000))
        out = relay.reshape(out, (1, 1000))
        return out

    inputs = {
        "a": tvm.nd.array(np.random.uniform(0, 1, (1, 1, 1, 1000)).astype("float32"))
    }

    outputs = []
    for acl in [False, True]:
        func = get_model(inputs["a"].shape, iter(inputs))
        outputs.append(build_and_run(func, inputs, 1, None, device,
                                     enable_acl=acl))
    verify(outputs, atol=0.002, rtol=0.01)


def test_multiple_runs():
    """
    Test that multiple runs of an operator work.
    Note: the result isn't checked.
    """
    if skip_runtime_test():
        return

    device = Device()

    def get_model():
        a = relay.var("a", shape=(1, 28, 28, 512), dtype="float32")
        w = tvm.nd.array(np.ones((256, 1, 1, 512), dtype="float32"))
        weights = relay.const(w, "float32")
        conv = relay.nn.conv2d(
            a,
            weights,
            kernel_size=(1, 1),
            data_layout="NHWC",
            kernel_layout="OHWI",
            strides=(1, 1),
            padding=(0, 0),
            dilation=(1, 1)
        )
        params = {"w": w}
        return conv, params

    inputs = {
        "a": tvm.nd.array(np.ones((1, 28, 28, 512), dtype="float32")),
    }

    func, params = get_model()
    build_and_run(func, inputs, 1,
                  params, device,
                  enable_acl=True,
                  no_runs=3)


if __name__ == "__main__":
    test_multiple_ops()
    test_multiple_runs()
