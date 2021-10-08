#!/usr/bin/env python

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
"""`compile` api that convert torch module to torch tvm module"""
import os
import tvm
from tvm.relay.op.tensor import exp
import tvm.testing
from tvm import relay, autotvm
from tvm.runtime import load_module
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor
from . import GraphModule


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
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

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
    os.remove(tmp_log_file)


def get_tuning_opt(log_file="tuning.log", n_trial=200):
    tuning_opt = {
        "log_filename": log_file,
        "tuner": "random",
        "n_trial": n_trial,
        "early_stopping": 60,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        ),
    }
    return tuning_opt


tvm_assets = ["mod.so", "graph.json", "params"]


class PyTorchTVMModule():
    def __init__(self) -> None:
        self.script_module = None
        self.input_infos = None
        self.default_dtype = "float32"
        self.mod = None
        self.params = None
        self.tasks = None
        self.target = "cuda"
        self.dev = tvm.cuda(0)
        self.log_file = None
        self.tvm_module = None
        self.tvm_graph = None
        self.tvm_lib = None
        self.tvm_params = None

    def from_pytorch(self, script_module, input_infos, default_dtype="float32"):
        self.script_module = script_module
        self.input_infos = input_infos
        self.default_dtype = default_dtype
        self.mod, self.params = relay.frontend.from_pytorch(script_module, input_infos, default_dtype=default_dtype)

    def tune_tvm(self, log_file="tuning.log", n_trial=200):
        self.tasks = autotvm.task.extract_from_program(
            self.mod["main"], target=self.target, params=self.params,
        )
        self.log_file = log_file
        tuning_opt = get_tuning_opt(log_file, n_trial)
        tune_tasks(self.tasks, **tuning_opt)

    def build_tvm(self, export_dir, debug_runtime=False):
        tvm_mod = self._build_tvm(debug_runtime)
        self._export_tvm(export_dir)
        return tvm_mod

    def _build_tvm(self, debug_runtime=False):
        # compile kernels with history best records
        with autotvm.apply_history_best(self.log_file):
            with tvm.transform.PassContext(opt_level=3):
                self.tvm_graph, self.tvm_lib, self.tvm_params = relay.build(
                    self.mod, target=self.target, params=self.params)

        if not debug_runtime:
            self.tvm_module = graph_executor.create(self.tvm_graph, self.tvm_lib, device=self.dev)
        else:
            self.tvm_module = debug_executor.create(self.tvm_graph, self.tvm_lib, device=self.dev)
        self.tvm_module.set_input(**self.tvm_params)
        return self.tvm_module

    def _export_tvm(self, export_dir):
        if not os.path.isdir(export_dir):
            os.makedirs(export_dir)
        self.export_dir = export_dir
        self.tvm_lib.export_library(os.path.join(export_dir, tvm_assets[0]))
        with open(os.path.join(export_dir, tvm_assets[1]), 'w') as fout:
            fout.write(self.tvm_graph)
        with open(os.path.join(export_dir, tvm_assets[2]), 'wb') as fout:
            fout.write(relay.save_param_dict(self.tvm_params))

    def load_tvm(self, export_dir):
        self.export_dir = export_dir
        self.tvm_lib = load_module(os.path.join(export_dir, tvm_assets[0]))
        with open(os.path.join(export_dir, tvm_assets[1]), 'r') as f:
            self.tvm_graph = f.read()
        with open(os.path.join(export_dir, tvm_assets[2]), 'rb') as f:
            self.tvm_params = relay.load_param_dict(f.read())

        self.tvm_module = graph_executor.create(self.tvm_graph, self.tvm_lib, device=self.dev)
        self.tvm_module.set_input(**self.tvm_params)
        return self.tvm_module

    def build_pytorch_op(self, num_inputs, num_outputs, input_infos=None):
        assert self.export_dir, "you must build_tvm or load_tvm before"
        input_infos = input_infos or self.input_infos
        assert input_infos
        assert len(input_infos) == num_inputs
        assets = [os.path.join(self.export_dir, i) for i in tvm_assets]
        input_shapes = [i[1] for i in input_infos]
        mod = GraphModule(num_inputs=num_inputs, num_outputs=num_outputs).to(self.target)
        mod.init(input_shapes, *assets)
        return mod


def compile(script_module, option):
    """
        option = {
            "input_infos": [
                ("x", (1, 3, 244, 244)),
            ],
            "default_dtype": "float16",
            "export_dir": "pytorch_compiled",
            "num_outputs": 1,
            "tuning_n_trials": 20,  # set zero to skip tuning
            "tuning_log_file": "tuning.log",
        }
        script_module = torch.jit.script(model)
        pytorch_tvm_module = compile(script_module, option)
        pytorch_tvm_module("model_tvm.pt")
    """
    mod = PyTorchTVMModule()
    print("Converting...")
    input_infos = option["input_infos"]
    default_dtype = option.get("default_dtype", "float32")
    export_dir = option.get("export_dir", "pytorch_compiled")
    tuning_log_file = option.get("tuning_log_file", "tuning.log")
    tuning_n_trials = option.get("tuning_n_trials", 20)
    num_outputs = option.get("num_outputs", 1)

    mod.log_file = tuning_log_file
    mod.from_pytorch(script_module, input_infos, default_dtype)

    if tuning_n_trials > 0:
        print("Tuning...")
        mod.tune_tvm(log_file=tuning_log_file, n_trial=tuning_n_trials)

    print("Building...")
    mod.build_tvm(export_dir)
    pytorch_mod = mod.build_pytorch_op(num_inputs=len(input_infos), num_outputs=num_outputs)
    return pytorch_mod
