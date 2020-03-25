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
import chainer.links as L

from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay.testing.config import ctx_list
from tvm.relay.frontend.pytorch import get_graph_input_names

def verify_model(model, input_data=[], ctx_list=ctx_list()):
    """Assert that the output of a compiled model matches with that of its
    baseline."""

    # Run Chainer Model for the input
    
    # Convert Chainer model to TVM and get the output
    mod, params = relay.frontend.from_chainer(model, input_shapes)
    compiled_input = dict(zip(input_names,
                              [inp.cpu().numpy() for inp in baseline_input]))

    with relay.build_config(opt_level=3):
        for target, ctx in ctx_list:
            relay_graph, relay_lib, relay_params = relay.build(mod, target=target, params=params)
            relay_model = graph_runtime.create(relay_graph, relay_lib, ctx)
            relay_model.set_input(**relay_params)
            for name, inp in compiled_input.items():
                relay_model.set_input(name, inp)
            relay_model.run()

            for i, baseline_output in enumerate(baseline_outputs):
                compiled_output = relay_model.get_output(i).asnumpy()

                assert_shapes_match(baseline_output, compiled_output)
                tvm.testing.assert_allclose(baseline_output, compiled_output,
                                            rtol=1e-3, atol=1e-3)

# Single operator tests
def test_forward_add():

def test_forward_subtract():

def test_forward_multiply():

def test_forward_concatenate():

def test_forward_relu():

def test_forward_conv():

if __name__ == "__main__":
    # Single operator tests
    test_forward_add()
    test_forward_subtract()
    test_forward_multiply()
    test_forward_concatenate()
    test_forward_relu()
    test_forward_conv()
