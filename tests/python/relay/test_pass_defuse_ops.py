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
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass


def test_defuse_simple():
    """Simple testcase."""

    def before():
        x = relay.var("x", shape=(10, 20))
        y = relay.add(x, relay.const(1, "float32"))
        z = relay.exp(y)
        w = relay.squeeze(z)
        return relay.Function([x], w)

    x = before()
    x = run_opt_pass(x, transform.InferType())
    fused = run_opt_pass(x, transform.FuseOps())
    defused = run_opt_pass(fused, transform.DefuseOps())

    assert tvm.ir.structural_equal(x, defused)


def test_inception_like():
    def conv(data):
        y = relay.nn.conv2d(data, relay.var("w"), kernel_size=(3, 3), padding=(1, 1), channels=16)
        return relay.nn.relu(data=y)

    def inception_like(data):
        c0 = conv(data)
        c1 = conv(data)
        return relay.concatenate((c0, c1), axis=1)

    def before(dshape):
        x = relay.var("x", shape=dshape)
        in1 = inception_like(x)
        in2 = inception_like(in1)
        return relay.Function(relay.analysis.free_vars(in2), in2)

    dshape = (1, 16, 64, 64)
    x = before(dshape)
    x = run_opt_pass(x, transform.InferType())
    fused = run_opt_pass(x, transform.FuseOps())
    defused = run_opt_pass(fused, transform.DefuseOps())

    assert tvm.ir.structural_equal(x, defused)


if __name__ == "__main__":
    test_defuse_simple()
    test_inception_like()
