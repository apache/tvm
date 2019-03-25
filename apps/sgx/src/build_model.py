<<<<<<< HEAD:apps/sgx/enclave/src/build_model.py
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
=======
#!/usr/bin/python3

>>>>>>> cec2cf25... Use Fortanix SGX:apps/sgx/src/build_model.py
"""Creates a simple TVM modules."""

import os
from os import path as osp
import sys

import tvm
from tvm import relay
import tvm.relay.testing


def main():
    dshape = (1, 28, 28)
    net, params = relay.testing.mlp.get_workload(batch_size=dshape[0], dtype='float32')

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build_module.build(
            net, target='llvm --system-lib',  params=params)

    build_dir = osp.abspath(sys.argv[1])
    if not osp.isdir(build_dir):
        os.makedirs(build_dir, exist_ok=True)

    lib.save(osp.join(build_dir, 'model.o'))
    with open(osp.join(build_dir, 'graph.json'), 'w') as f_graph_json:
        f_graph_json.write(graph)
    with open(osp.join(build_dir, 'params.bin'), 'wb') as f_params:
        f_params.write(relay.save_param_dict(params))


if __name__ == '__main__':
    main()
