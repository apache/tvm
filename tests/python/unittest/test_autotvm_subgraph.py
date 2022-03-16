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
"""Test autotvm tuning by subgraph"""
import tvm.relay.testing
from tvm import relay
from tvm import autotvm
from tvm.autotvm.env import GLOBAL_SCOPE
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import os
import tvm.contrib.graph_executor as runtime
import numpy as np

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)

    if name == "resnet-18":
        mod, params = relay.testing.resnet.get_workload(num_layers=18, batch_size=batch_size)
    elif name == "resnet3d-18":
        mod, params = relay.testing.resnet_3d.get_workload(num_layers=18, batch_size=batch_size)
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size)
    elif name == "dcgan":
        mod, params = relay.testing.dcgan.get_workload(batch_size=batch_size)
        input_shape = (batch_size, 100)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape


def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.isfile(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        print(tsk.name)
        print(tsk.config_space)

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)

def test_tune_subgraphs():
    target = "llvm"
    conv2d = relay.op.get("nn.conv2d")
    dense = relay.op.get("nn.dense")
    tune = True
    model_name = "mobilenet"
    mod, params, input_shape = get_network(model_name, batch_size=1)
    dtype = "float32"
    # tune_subgraph should be set to True after converting the front model to relay
    GLOBAL_SCOPE.tune_subgraph = True
    if GLOBAL_SCOPE.tune_subgraph:
        log_file = "%s/test_tune_subgraph_%s.log" % (os.getcwd(), model_name)
    else:
        log_file = "%s/test_tune_single_%s.log" % (os.getcwd(), model_name)
    if tune:
        # extract workloads from relay program
        print("Extract tasks...")
        tasks = autotvm.task.extract_from_program(
            mod, target=target, params=params, ops=(conv2d, dense)
        )
        print(len(tasks))
        tuning_option = {
            "log_filename": log_file,
            "tuner": "ga",
            "n_trial": 2000,
            "early_stopping": None,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.LocalRunner(
                    number=1, repeat=10, min_repeat_ms=0
                ),
            ),
        }
        # run tuning tasks
        tune_tasks(tasks, **tuning_option)
    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            lib = relay.build_module.build(
                mod, target=target, params=params)
        dev = tvm.cpu()
        module = runtime.GraphModule(lib["default"](dev))
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input("data", data_tvm)
        # evaluate
        print("Evaluate inference time cost...")
        print(module.benchmark(dev, number=100, repeat=3))
                
if __name__ == "__main__":
    test_tune_subgraphs()