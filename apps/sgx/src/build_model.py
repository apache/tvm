#!/usr/bin/python3

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

"""Creates a simple TVM modules."""

import os
from os import path as osp
import sys

from tvm import relay, runtime
from tvm.relay import testing
import tvm
from tvm import te


def main():
    dshape = (1, 28, 28)
    net, params = relay.testing.mlp.get_workload(batch_size=dshape[0], dtype="float32")

    dshape = (1, 3, 224, 224)
    net, params = relay.testing.resnet.get_workload(
        layers=18, batch_size=dshape[0], image_shape=dshape[1:]
    )

    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(net, "llvm --system-lib", params=params)

    build_dir = osp.abspath(sys.argv[1])
    if not osp.isdir(build_dir):
        os.makedirs(build_dir, exist_ok=True)

    lib.save(osp.join(build_dir, "model.o"))
    with open(osp.join(build_dir, "graph.json"), "w") as f_graph_json:
        f_graph_json.write(graph)
        with open(osp.join(build_dir, "params.bin"), "wb") as f_params:
            f_params.write(runtime.save_param_dict(params))


if __name__ == "__main__":
    main()
