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

import argparse
import os
from tvm import relay
import tvm
import logging
import json

def build_module(opts):
    dshape = (1, 3, 224, 224)
    from mxnet.gluon.model_zoo.vision import get_model
    block = get_model('mobilenet0.25', pretrained=True)
    shape_dict = {'data': dshape}
    mod, params = relay.frontend.from_mxnet(block, shape_dict)
    func = mod["main"]
    func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(
            func, 'llvm --system-lib', params=params)

    build_dir = os.path.abspath(opts.out_dir)
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir)

    lib.save(os.path.join(build_dir, 'model.o'))
    with open(os.path.join(build_dir, 'graph.json'), 'w') as f_graph_json:
        f_graph_json.write(graph)
    with open(os.path.join(build_dir, 'params.bin'), 'wb') as f_params:
        f_params.write(relay.save_param_dict(params))

def build_bridge(opts):
    build_dir = os.path.abspath(opts.out_dir)
    json_data = {}
    with open(os.path.join(build_dir, 'graph.json'), 'rt') as fp:
        json_data = json.loads(fp.read())
    if json_data == {}:
        return
    else:
        nodes = [node['attrs']['func_name'] for node in json_data['nodes'] if node['op'] == "tvm_op"]
    with open(os.path.join(build_dir, 'bridge.c'), 'w') as f_bridge:
        f_bridge.write("#include \"../../../src/runtime/crt/packed_func.h\"\n")
        f_bridge.write("\n")
        f_bridge.write("#define REGISTER_PACKED_FUNC(func_name) \\\n")
        f_bridge.write("  do { \\\n")
        f_bridge.write("    strcpy(fexecs[idx].name, #func_name ); \\\n")
        f_bridge.write("    fexecs[idx].fexec = func_name ; \\\n")
        f_bridge.write("    idx ++; \\\n")
        f_bridge.write("  } while (0)\n")
        f_bridge.write("\n")
        for node in nodes:
            f_bridge.write("int %s(TVMValue * args, int * arg_type_ids, int num_args, TVMRetValueHandle ret, void * res);\n" % (node,))
        f_bridge.write("\n")
        f_bridge.write("void PackedFunc_SetupExecs() {\n")
        f_bridge.write("  int32_t idx = 0;\n")
        for node in nodes:
            f_bridge.write("  REGISTER_PACKED_FUNC(%s);\n" % (node,))
        f_bridge.write("}\n")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out-dir', default='.')
    opts = parser.parse_args()

    build_module(opts)
    build_bridge(opts)
