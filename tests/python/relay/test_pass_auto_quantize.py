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
from tvm.relay import testing


def quantize_and_build(out):
    f = relay.Function(relay.analysis.free_vars(out), out)
    mod, params = testing.create_workload(f)

    with relay.quantize.qconfig(skip_conv_layers=[]):
        qmod = relay.quantize.quantize(mod, params)

    relay.build(qmod, "llvm", params=params)


def test_mul_rewrite():
    """a test case where rhs of mul is not constant"""
    data = relay.var("data", shape=(1, 16, 64, 64))
    multiplier = relay.sigmoid(relay.var("data", shape=(1, 16, 1, 1)))
    conv = relay.nn.conv2d(data, relay.var("weight"),
                           kernel_size=(3, 3),
                           padding=(1, 1),
                           channels=16)
    act = relay.nn.relu(data=conv)

    quantize_and_build(act * multiplier)

    pool = relay.nn.global_avg_pool2d(data=act)

    quantize_and_build(act * pool)

if __name__ == "__main__":
    test_mul_rewrite()
