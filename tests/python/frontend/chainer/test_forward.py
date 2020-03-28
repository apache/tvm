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
# pylint: disable=import-self, invalid-name, unused-argument
"""Unit tests for various models and operators"""
from time import time
import sys
from scipy.stats import t as tdistr
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay.testing.config import ctx_list
from tvm.relay.frontend.pytorch import get_graph_input_names

def verify_model(model, input_data=[], ctx_list=ctx_list()):
    """Assert that the output of a compiled model matches with that of its
    baseline."""

    # Run Chainer Model for the input
    baseline_outputs = model(chainer.Variable(input_data[0])).data
    
    # Convert Chainer model to TVM and get the output
    mod, params = relay.frontend.from_chainer(model, input_data[0].shape, "float32")

    with relay.build_config(opt_level=3):
        for target, ctx in ctx_list:
            relay_graph, relay_lib, relay_params = relay.build(mod, target=target, params=params)
            relay_model = graph_runtime.create(relay_graph, relay_lib, ctx)
            relay_model.set_input(**relay_params)
            relay_model.set_input("var2", input_data[0])
            relay_model.run()

            compiled_output = relay_model.get_output(0).asnumpy()

            tvm.testing.assert_allclose(baseline_outputs, compiled_output,
                                        rtol=1e-3, atol=1e-3)

# Single operator tests
def test_forward_relu():
    class Link(chainer.Chain):
        def __call__(self, x):
            return F.relu(x)
    input_data = np.random.uniform(-1, 1, (1, 3, 7, 7)).astype(np.float32)
    verify_model(Link(), [input_data])

if __name__ == "__main__":
    # Single operator tests
    test_forward_relu()
