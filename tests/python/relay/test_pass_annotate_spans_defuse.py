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
"""Unit tests for annotating spans."""


import tvm
import tvm.relay as relay
from tvm.relay import testing
import tvm.testing


def test_annotate_spans_compatibility():
    data = relay.var("data", relay.TensorType((1, 3, 64, 64), "float32"))
    weight = relay.var("weight")

    bn_gamma = relay.var("bn_gamma")
    bn_beta = relay.var("bn_beta")
    bn_mmean = relay.var("bn_mean")
    bn_mvar = relay.var("bn_var")

    simple_net = relay.nn.conv2d(
        data=data, weight=weight, kernel_size=(3, 3), channels=3, padding=(1, 1)
    )
    simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
    simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)

    module, params = testing.create_workload(simple_net)

    # Apply some simple passes to legalize the IR.
    with tvm.transform.PassContext(opt_level=0):
        module, params = relay.optimize(
            module, target=tvm.testing.enabled_targets()[0][0], params=params
        )

    seq = tvm.transform.Sequential([relay.transform.AnnotateSpans(), relay.transform.DefuseOps()])
    with tvm.transform.PassContext(opt_level=3):
        module = seq(module)


if __name__ == "__main__":
    tvm.testing.main()
