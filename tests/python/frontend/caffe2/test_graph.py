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
"""Test graph equality of caffe2 models."""
from model_zoo import c2_squeezenet, relay_squeezenet
import tvm
from tvm import relay
from tvm.relay import transform


def compare_graph(lhs_mod, rhs_mod):
    lhs_mod = transform.InferType()(lhs_mod)
    rhs_mod = transform.InferType()(rhs_mod)
    assert tvm.ir.structural_equal(lhs_mod["main"], rhs_mod["main"])


def test_squeeze_net():
    shape_dict = {"data": (1, 3, 224, 224)}
    dtype_dict = {"data": "float32"}
    mod, _, = relay.frontend.from_caffe2(
        c2_squeezenet.init_net, c2_squeezenet.predict_net, shape_dict, dtype_dict
    )
    relay_mod, _ = relay_squeezenet()
    compare_graph(mod, relay_mod)


if __name__ == "__main__":
    test_squeeze_net()
