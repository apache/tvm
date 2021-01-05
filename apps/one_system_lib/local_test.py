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

import platform

import numpy as np
import tvm.relay.testing
from mxnet.gluon.model_zoo import vision
from tvm import relay
from tvm.contrib import graph_runtime

target = "llvm --system-lib"

if __name__ == "__main__":
    dshape = (1, 3, 224, 224)

    resnet18 = vision.resnet18_v1(pretrained=True)
    alexnet = vision.alexnet(pretrained=True)
    shape_dict = {"data": dshape}

    models = {}

    mod, params = relay.frontend.from_mxnet(resnet18, shape_dict)
    models["resnet18"] = [mod["main"], params]

    mod, params = relay.frontend.from_mxnet(alexnet, shape_dict)
    models["alexnet"] = [mod["main"], params]

    with tvm.transform.PassContext(opt_level=3):
        ret_mods = relay.build_module.build_one_system_lib(models, target=target)

    # alexnet's lib and resnet18's lib is same one
    lib = ret_mods["resnet18"].get_lib()

    if platform.system() == "Darwin":
        lib_name = "main.dylib"
    elif platform.system() == "Linux":
        lib_name = "main.so"
    elif platform.system() == "Windows":
        lib_name = "main.dll"
    else:
        raise Exception("unknown system " + platform.system())

    print("export_library main lib")
    lib.export_library(lib_name)

    # or save object file for deploy usage
    # lib.save(os.path.join(work_root, binary_dir, 'model.o'))

    print("load main lib")
    sysLib = tvm.runtime.load_module(lib_name)

    ctx = tvm.cpu(0)

    input_data = np.random.random(dshape).astype(np.float32)

    for fk in ret_mods:
        mg = ret_mods[fk].get_json()
        mp = ret_mods[fk].get_params()
        print("test " + fk + "   ------------------------------------")
        module = graph_runtime.create(mg, sysLib, ctx)
        module.load_params(relay.save_param_dict(mp))
        module.set_input("data", tvm.nd.array(input_data))
        module.run()
        num_output = module.get_num_outputs()
        for idx in range(num_output):
            print(module.get_output(idx).shape)
