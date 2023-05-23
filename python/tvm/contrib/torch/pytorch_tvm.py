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
# pylint: disable=redefined-builtin
"""`compile` api that convert torch module to torch tvm module"""
import os
import warnings
import tvm
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
    """Tune tasks and generate tuning log to file"""
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = f"[Task {i + 1:2d}/{len(tasks):2d}] "

        # create tuner
        if tuner == "xgb":
            tuner_obj = XGBTuner(tsk, loss_type="reg")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="curve")
        elif tuner == "xgb_rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_rank_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_rank_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_rank_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve")
        elif tuner == "xgb_rank_binary":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary")
        elif tuner == "xgb_rank_binary_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="knob")
        elif tuner == "xgb_rank_binary_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="itervar")
        elif tuner == "xgb_rank_binary_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="curve")
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
    if not os.path.exists(log_filename):
        with open(log_filename, "w", encoding="utf-8"):
            pass
    if os.path.exists(tmp_log_file):
        autotvm.record.pick_best(tmp_log_file, log_filename)
        os.remove(tmp_log_file)


def get_tuning_opt(log_file="tuning.log", n_trial=200):
    """Returns tuning options"""
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


TVM_ASSETS = ["mod.so", "graph.json", "params"]


class PyTorchTVMModule:
    """Helper class for compiling pytorch module to tvm module"""

    def __init__(self, target="cuda", device=tvm.cuda(0)) -> None:
        self.script_module = None
        self.input_infos = None
        self.default_dtype = "float32"
        self.mod = None
        self.params = None
        self.tasks = None
        self.target = target
        self.dev = device
        self.log_file = None
        self.tvm_module = None
        self.tvm_graph = None
        self.tvm_lib = None
        self.tvm_params = None

    def from_pytorch(self, script_module, input_infos, default_dtype="float32"):
        self.script_module = script_module
        self.input_infos = input_infos
        self.default_dtype = default_dtype
        self.mod, self.params = relay.frontend.from_pytorch(
            script_module, input_infos, default_dtype=default_dtype
        )

    def tune_tvm(self, log_file="tuning.log", n_trial=200):
        self.tasks = autotvm.task.extract_from_program(
            self.mod["main"],
            target=self.target,
            params=self.params,
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
                    self.mod, target=self.target, params=self.params
                )

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
        self.tvm_lib.export_library(os.path.join(export_dir, TVM_ASSETS[0]))
        with open(os.path.join(export_dir, TVM_ASSETS[1]), "w", encoding="utf8") as fout:
            fout.write(self.tvm_graph)
        with open(os.path.join(export_dir, TVM_ASSETS[2]), "wb") as fout:
            fout.write(relay.save_param_dict(self.tvm_params))

    def load_tvm(self, export_dir):
        """Load tvm module from export directory"""
        self.export_dir = export_dir
        self.tvm_lib = load_module(os.path.join(export_dir, TVM_ASSETS[0]))
        with open(os.path.join(export_dir, TVM_ASSETS[1]), "r", encoding="utf8") as f:
            self.tvm_graph = f.read()
        with open(os.path.join(export_dir, TVM_ASSETS[2]), "rb") as f:
            self.tvm_params = relay.load_param_dict(f.read())

        self.tvm_module = graph_executor.create(self.tvm_graph, self.tvm_lib, device=self.dev)
        self.tvm_module.set_input(**self.tvm_params)
        return self.tvm_module

    def build_pytorch_module(self, num_inputs, num_outputs, input_infos=None):
        """Build pytorch module containing TVM Graph Module"""
        warnings.warn(
            " ".join(
                (
                    "This function will be removed at TVM version 0.11,",
                    "we suggest users to use `optimized_torch` for tuning Torch modules instead.",
                )
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        assert self.export_dir, "you must build_tvm or load_tvm before"
        input_infos = input_infos or self.input_infos
        assert input_infos
        assert len(input_infos) == num_inputs
        assets = [os.path.join(self.export_dir, i) for i in TVM_ASSETS]
        input_shapes = [i[1] for i in input_infos]

        def _tvm_dev_to_pt_dev(device):
            """convert tvm device to pytorch device string"""
            if tvm.runtime.Device.MASK2STR[device.device_type] == "cpu":
                return "cpu"
            if tvm.runtime.Device.MASK2STR[device.device_type] == "cuda":
                return f"cuda:{device.device_id}"
            raise ValueError(f"unsupported device for pt graph module: {device}")

        mod = GraphModule(num_inputs=num_inputs, num_outputs=num_outputs).to(
            _tvm_dev_to_pt_dev(self.dev)
        )
        mod.init(input_shapes, *assets)
        return mod


def compile(script_module, option):
    """
    example:
    option = {
        "input_infos": [
            ("x", (1, 3, 244, 244)),
        ],
        "default_dtype": "float16",
        "export_dir": "pytorch_compiled",
        "num_outputs": 1,
        "tuning_n_trials": 20,  # set zero to skip tuning
        "tuning_log_file": "tuning.log",
        "target": "llvm",
        "device": tvm.cpu(),
    }
    script_module = torch.jit.script(model)
    pytorch_tvm_module = compile(script_module, option)
    pytorch_tvm_module("model_tvm.pt")
    """
    warnings.warn(
        " ".join(
            (
                "This function will be removed at TVM version 0.11,",
                "we suggest users to use `optimized_torch` for tuning Torch modules instead.",
            )
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    input_infos = option["input_infos"]
    default_dtype = option.get("default_dtype", "float32")
    export_dir = option.get("export_dir", "pytorch_compiled")
    tuning_log_file = option.get("tuning_log_file", "tuning.log")
    tuning_n_trials = option.get("tuning_n_trials", 20)
    num_outputs = option.get("num_outputs", 1)
    target = option.get("target", "cuda")
    device = option.get("device", tvm.cuda(0))

    mod = PyTorchTVMModule(target=target, device=device)
    print("Converting...")

    mod.log_file = tuning_log_file
    mod.from_pytorch(script_module, input_infos, default_dtype)

    if tuning_n_trials > 0:
        print("Tuning...")
        mod.tune_tvm(log_file=tuning_log_file, n_trial=tuning_n_trials)

    print("Building...")
    mod.build_tvm(export_dir)
    pytorch_mod = mod.build_pytorch_module(num_inputs=len(input_infos), num_outputs=num_outputs)
    return pytorch_mod
