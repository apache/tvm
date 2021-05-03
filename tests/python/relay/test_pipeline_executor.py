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

import numpy as np
import tvm
import tvm.testing
from tvm import relay
from tvm.relay import transform
from tvm.contrib import graph_executor, pipeline_executor


def run_modules(mods, dev, target, dname, data):
    for mod in mods:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target)

        m = graph_executor.GraphModule(lib["default"](dev))
        m.set_input(dname, data)
        m.run()
        n = m.get_num_outputs()
        output = m.get_output(0).asnumpy()
        data = output

    return output


def get_mannual_mod():
    mods = []
    dshape = (3, 3)
    data = relay.var("data", relay.TensorType(dshape, "float32"))
    mvalue1 = np.full((1), 5).astype("float32")
    mvalue2 = np.full((1), 2).astype("float32")
    mvalue3 = np.full((1), 3).astype("float32")
    mvalue4 = np.full((1), 4).astype("float32")
    mv1 = relay.Constant(tvm.nd.array(mvalue1))
    mv2 = relay.Constant(tvm.nd.array(mvalue2))
    mv3 = relay.Constant(tvm.nd.array(mvalue3))
    mv4 = relay.Constant(tvm.nd.array(mvalue4))
    net1 = relay.multiply(data, mv1)

    net2 = relay.add(data, mv2)
    net2 = relay.add(net2, mv3)

    net3 = relay.multiply(data, mv4)

    net4 = relay.subtract(data, mv1)

    mods.append(tvm.IRModule.from_expr(relay.Function([data], net1)))
    mods.append(tvm.IRModule.from_expr(relay.Function([data], net2)))
    mods.append(tvm.IRModule.from_expr(relay.Function([data], net3)))
    mods.append(tvm.IRModule.from_expr(relay.Function([data], net4)))

    return mods, dshape


def test_pipeline():
    """
    #split compute graph into 4 pipeline
    """
    mods, dshape = get_mannual_mod()
    """
    #Prepare batch data for pipeline feeding
    """
    datas = []
    for i in range(len(mods) + 1):
        datas.append(np.full(dshape, 3 + i).astype("float32"))

    """
    #Run with graph executor for verification purpose
    """
    outs = [run_modules(mods, tvm.cpu(), "llvm", "data", data) for data in datas]

    """
    #Parameter use for pipeline executor creation
    #Build module and append module and device type into variable that
    #use for pipeline creation.
    #first and second pipeline use cuda when cuda enable, second and 
    #last pipeline use cpu
    """
    mod_config = {}
    for i in range(len(mods)):
        mconfig = {"target_host": None, "mod_name": "default", "build": None, "params": None}
        # if cuda enabled, first 2 module us cuda as target
        if i < 2 and tvm.testing.device_enabled("cuda"):
            mconfig["target"] = "cuda"
            mconfig["dev"] = tvm.gpu()
        else:
            mconfig["target"] = "llvm"
            mconfig["dev"] = tvm.cpu()

        mod_config[mods[i]] = mconfig

    """
    #build and create pipeline module
    """
    with relay.build_config(opt_level=3):
        pipeline_module = pipeline_executor.create(mods, mod_config)

    """
    #Use pipeline executor to pipeline the said pipeline which use different backend
    """
    for data in datas:
        pipeline_module.set_input("data", data)
        pipeline_module.run()

    """
    Get result
    """
    pipeline_outputs = []
    for i in range(len(datas)):
        pipeline_outputs.append(pipeline_module.get_output()[0].asnumpy())

    """
    #Stop pipeline execution.
    """
    pipeline_module.stop()
    """

    #Verify result
    """
    for ref_out, out in zip(outs, pipeline_outputs):
        tvm.testing.assert_allclose(ref_out, out)


if __name__ == "__main__":
    test_pipeline()
