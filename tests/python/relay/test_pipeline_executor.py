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

class PipelineModuleConfig:
    class interface:
        def __init__(self, owner, itype, name):
            self.owner_ = owner
            self.itype_ = itype
            self.name_ = name
            self.dependent_ = []
            return

        def get_dependent_str(self):
            name = ""
            for dependent in self.dependent_:
                name = name + dependent.name_
            return name

        def addDependent(self, dependent):
            self.dependent_.append(dependent)

    class instance:
        def __init__(self):
            self.interfaces_ = {1:{}, 2:{}}
            return

        def get_interface(self, itype,  name):
            if name not in self.interfaces_[itype]:
                self.interfaces_[itype][name] = PipelineModuleConfig.interface(0, itype, name)

            return self.interfaces_[itype][name]

        def input(self, name):
            return self.get_interface(1, name)

        def output(self, index):
            return self.get_interface(2, index)


    def __init__(self, mods):
        self.pipe_instance = self.instance()
        self.mod_instance = {m:self.instance() for m in mods}
        return

    def __str__(self):
        dump = "Inputs\n"
        for input_name in self.pipe_instance.interfaces_[1]:
            inf = self.pipe_instance.interfaces_[1][input_name]
            dump = dump + "  |" +input_name + ": " + inf.get_dependent_str() + "\n"
        return dump

    def __getitem__(self, key):
        return self.mod_instance[key]

    def pipe_input(self, name):
        return self.pipe_instance.input(name)

    def pipe_output(self, index):
        return self.pipe_instance.output(index)

    def connect(self, left:interface, right:interface):
        left.addDependent(right)
        return



def get_mannual_mod():
    mods = []
    dshape = (3, 3)
    data = relay.var("data_0", relay.TensorType(dshape, "float32"))
    data21 = relay.var("data_1", relay.TensorType(dshape, "float32"))
    data_net1_output_1 = relay.var("data_0", relay.TensorType(dshape, "float32"))
    data_net1_output_2 = relay.var("data_1", relay.TensorType(dshape, "float32"))
    data_net2_output_1 = relay.var("data_0", relay.TensorType(dshape, "float32"))
    mvalue1 = np.full((1), 1).astype("float32")
    mvalue2 = np.full((1), 2).astype("float32")
    mvalue3 = np.full((1), 3).astype("float32")
    mv1 = relay.Constant(tvm.nd.array(mvalue1))
    mv2 = relay.Constant(tvm.nd.array(mvalue2))
    mv3 = relay.Constant(tvm.nd.array(mvalue3))

    # net1 have three output, output3 is final output
    net_output1 = relay.add(data, mv1)
    net_output2 = relay.subtract(data, mv2)
    net_output3 = relay.multiply(data, mv3)

    # net2 use net1 output1 as input
    net2 = relay.add(data_net1_output_1, mv2)
    net2 = relay.add(net2, data21)
    net2 = relay.add(net2, mv3)

    # net3 use net2 output1 and net1 outpu2 as input
    net3 = relay.multiply(data_net2_output_1, mv3)
    net3 = relay.add(net3, data_net1_output_2)

    mods.append(
        tvm.IRModule.from_expr(
            relay.Function([data], relay.Tuple([net_output1, net_output2, net_output3]))
        )
    )
    mods.append(tvm.IRModule.from_expr(relay.Function([data_net1_output_1, data21], net2)))
    mods.append(
        tvm.IRModule.from_expr(relay.Function([data_net1_output_2, data_net2_output_1], net3))
    )

    return mods, dshape


def pipeline(target):
    """
    #Get 4 pipeline module.
    """
    mods, dshape = get_mannual_mod()
    """
    #Prepare batch data for pipeline feeding
    """
    datas = []
    for i in range(len(mods) + 1):
        datas.append(np.full(dshape, 3 + i).astype("float32"))

    # set configure
    indx = 0
    mod_config = {}
    mconfig = {"target_host": None, "mod_name": "default", "build": None, "params": None}
    mconfig1 = mconfig.copy()
    mconfig1["target"] = target[0]
    mconfig1["dev"] = target[1]
    # third output is final output, second output for mod3, first for mod2
    # input
    mconfig1["pipeline"] = {
        "mod_indx": 1,
        "output": [
            {"output_indx": 1, "dependent": [{"mod_indx": 2, "input_name": "data_0"}]},
            {"output_indx": 2, "dependent": [{"mod_indx": 3, "input_name": "data_0"}]},
            {"output_indx": 3, "dependent": [{"mod_indx": 0, "input_name": "1"}]},
        ],
    }
    mod_config[mods[0]] = mconfig1

    mconfig2 = mconfig.copy()
    mconfig2["target"] = "llvm"
    mconfig2["dev"] = tvm.cpu(0)
    mconfig2["pipeline"] = {
        "mod_indx": 2,
        "output": [
            {"output_indx": 1, "dependent": [{"mod_indx": 3, "input_name": "data_1"}]},
        ],
    }
    mod_config[mods[1]] = mconfig2

    mconfig3 = mconfig.copy()
    mconfig3["target"] = "llvm"
    mconfig3["dev"] = tvm.cpu(0)

    mconfig3["pipeline"] = {
        "mod_indx": 3,
        "output": [{"output_indx": 1, "dependent": [{"mod_indx": 0, "input_name": "2"}]}],
    }
    mod_config[mods[2]] = mconfig3
    """
    #build and create pipeline module
    """
    with relay.build_config(opt_level=3):
        pipeline_mods, string_config = pipeline_executor.build_pipeline(mod_config)

    pipeline_module = pipeline_executor.create(pipeline_mods, string_config)


def test_pipeline():
    if pipeline_executor.pipeline_executor_enabled():
        target_list = tvm.testing.enabled_targets()
        for target in target_list:
            pipeline(target)

def test_config():
    (mod1, mod2, mod3), dshape = get_mannual_mod()
    pipe_config = PipelineModuleConfig([mod1, mod2, mod3])
    pipe_config.connect(pipe_config.pipe_input("data_0"),
                        pipe_config[mod1].input("data_0"))

    pipe_config.connect(pipe_config.pipe_input("data_1"),
                        pipe_config[mod2].input("data_1"))

    pipe_config.connect(pipe_config[mod1].output(0),
                        pipe_config[mod2].input("data_0"))

    pipe_config.connect(pipe_config[mod1].output(1),
                        pipe_config[mod3].input("data_0"))

    pipe_config.connect(pipe_config[mod2].output(0),
                        pipe_config[mod3].input("data_1"))

    pipe_config.connect(pipe_config[mod1].output(2),
                        pipe_config.pipe_output("0"))

    pipe_config.connect(pipe_config[mod3].output(0),
                        pipe_config.pipe_output("1"))

    print(pipe_config)

if __name__ == "__main__":
    #test_pipeline()
    test_config()
