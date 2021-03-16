#!/usr/bin/env python3
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

"""Builds a simple graph for testing."""

from os import path as osp

import numpy as np
import tvm
from tvm import te
from tvm import relay, runtime
from tvm.relay import testing

CWD = osp.dirname(osp.abspath(osp.expanduser(__file__)))


def _get_model(dshape):
    data = relay.var("data", shape=dshape)
    fc = relay.nn.dense(data, relay.var("dense_weight"), units=dshape[-1] * 2)
    fc = relay.nn.bias_add(fc, relay.var("dense_bias"))
    left, right = relay.split(fc, indices_or_sections=2, axis=1)
    one = relay.const(1, dtype="float32")
    return relay.Tuple([(left + one), (right - one), fc])


def main():
    dshape = (32, 16)
    net = _get_model(dshape)
    mod, params = testing.create_workload(net)
    graph, lib, params = relay.build(mod, "llvm", params=params)

    with open(osp.join(CWD, "graph.json"), "w") as f_resnet:
        f_resnet.write(graph)
    with open(osp.join(CWD, "graph.params"), "wb") as f_params:
        f_params.write(runtime.save_param_dict(params))


if __name__ == "__main__":
    main()
