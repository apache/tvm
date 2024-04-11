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
# pylint: disable=unused-import
"""tvm.contrib.msc.framework.runtime.tvm.runner"""

import os
import time
from typing import Dict, List, Union, Any, Tuple
import numpy as np

import tvm
from tvm.contrib.msc.core.runtime import ModelRunner
from tvm.contrib.msc.core.tools import execute_step
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.framework.tvm.codegen import to_relax
from tvm.contrib.msc.framework.tvm import tools


class WrapRunnable(object):
    """Wrapped runnable for tools

    Parameters
    -------
    runner: ModelRunner
        The runner context
    runnable: tvm.relax.VirtualMachine
        The virtual machine.
    entry: str
        The entry funcname.
    """

    def __init__(self, runnable: tvm.relax.VirtualMachine, entry: str = "main"):
        self._runnable = runnable
        self._entry = entry

    def __call__(self, *inputs) -> List[tvm.nd.array]:
        execute_step("before_forward", *inputs)
        output = self._runnable[self._entry](*inputs)
        return execute_step("after_forward", output)


class TVMRunner(ModelRunner):
    """Runner of Relax"""

    def setup(self) -> dict:
        """Setup the runner

        Returns
        -------
        info: dict
            The setup info.
        """

        self._executable = None
        return super().setup()

    def _build_runnable(self, model: Any) -> Any:
        """Build runnable object

        Parameters
        -------
        model: Any
            The meta model.

        Returns
        -------
        runnable: Any
            The runnable
        """

        if self._training:
            model = tvm.relax.transform.DecomposeOpsForTraining()(model)
        else:
            model = tvm.relax.transform.DecomposeOpsForInference()(model)
        if "builder" in self._generate_config:
            builder, build_config = self._generate_config["builder"]
            runnable = builder(model, **build_config)
            self._logger.info(
                "Model({}) processed by customize builder {}({})".format(
                    self.framework, builder, build_config
                )
            )
        else:
            model = tvm.relax.transform.LegalizeOps()(model)
            if self._device.startswith("cpu"):
                target = tvm.target.Target("llvm")
                with tvm.transform.PassContext(opt_level=3):
                    self._executable = tvm.relax.build(model, target)
                    runnable = tvm.relax.VirtualMachine(self._executable, tvm.cpu())
            elif self._device.startswith("cuda"):
                target = tvm.target.Target("cuda")
                with target:
                    model = tvm.tir.transform.DefaultGPUSchedule()(model)
                with tvm.transform.PassContext(opt_level=3):
                    self._executable = tvm.relax.build(model, target)
                    runnable = tvm.relax.VirtualMachine(self._executable, tvm.cuda())
            else:
                raise NotImplementedError("Unsupported device " + str(self._device))
        return WrapRunnable(runnable)

    def _call_runnable(
        self, runnable: WrapRunnable, inputs: Dict[str, np.ndarray], device: str
    ) -> Union[List[np.ndarray], Dict[str, np.ndarray]]:
        """Call the runnable to get outputs

        Parameters
        -------
        runnable: tvm.relax.VirtualMachine
            The virtual machine.
        inputs: dict<str, data>
            The inputs in dict.
        device: str
            The device.

        Returns
        -------
        outputs: list<data>
            The outputs in list.
        """

        input_names = [i["name"] for i in self.get_inputs()]
        tvm_inputs = [
            msc_utils.cast_array(inputs[i], MSCFramework.TVM, device) for i in input_names
        ]
        return runnable(*tvm_inputs)

    def export_runnable(self, folder: msc_utils.MSCDirectory) -> dict:
        """Export the runnable

        Parameters
        -------
        folder: MSCDirectory
            The export folder.

        Returns
        -------
        info: dict
            The runnable info.
        """

        export_lib = folder.relpath("lib.so")
        self._executable.export_library(export_lib)
        return {
            "lib": export_lib,
            "device": self.device,
            "model_type": self.framework,
            "abstract": self.model_info,
        }

    @property
    def codegen_func(self):
        return to_relax

    @property
    def framework(self):
        return MSCFramework.TVM

    @classmethod
    def load_native(cls, model: Any, config: dict) -> Tuple[tvm.IRModule, str, bool]:
        """Load the native model

        Parameters
        -------
        model:
            The native model.
        config: dict
            The config for pipeline.

        Returns
        -------
        model: tvm.IRModule
            The loaded native model.
        device: str
            The device of the model.
        training: bool
            Whether the model is for training.
        """

        if isinstance(model, str) and os.path.isfile(model):
            with open(model, "r") as f:
                native_model = tvm.ir.load_json(f.read())
        elif isinstance(model, tvm.IRModule):
            native_model = model
        else:
            raise NotImplementedError(
                "Load native model {} with type {} is not supported".format(model, type(model))
            )
        if tvm.cuda().exist:
            device = "cuda"
        else:
            device = "cpu"
        return native_model, device, False

    @classmethod
    def run_native(
        cls,
        model: tvm.IRModule,
        inputs: Dict[str, np.ndarray],
        input_names: List[str],
        output_names: List[str],
        warm_up: int = 10,
        repeat: int = 0,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """Run the datas and get outputs

        Parameters
        -------
        model: tvm.IRModule
            The runnable model.
        inputs: dict<str, data>
            The inputs in dict.
        input_names: list<str>
            The input names.
        output_names: list<str>
            The outut names.
        warm_up: int
            The warm_up num for profile.
        repeat: int
            The repeat num for profile.

        Returns
        -------
        outputs: dict<str, np.array>
            The outputs in dict.
        avg_time: float
            The average time.
        """

        model = tvm.relax.transform.LegalizeOps()(model)
        if tvm.cuda().exist:
            target = tvm.target.Target("cuda")
            with target:
                model = tvm.tir.transform.DefaultGPUSchedule()(model)
            with tvm.transform.PassContext(opt_level=3):
                relax_exec = tvm.relax.build(model, target)
                runnable = tvm.relax.VirtualMachine(relax_exec, tvm.cuda())
            tvm_inputs = [tvm.nd.array(inputs[i], device=tvm.cuda()) for i in input_names]
        else:
            target = tvm.target.Target("llvm")
            with tvm.transform.PassContext(opt_level=3):
                relax_exec = tvm.relax.build(model, target)
                runnable = tvm.relax.VirtualMachine(relax_exec, tvm.cpu())
            tvm_inputs = [tvm.nd.array(inputs[i]) for i in input_names]

        def _run_once():
            return runnable["main"](*tvm_inputs)

        if repeat > 0:
            for _ in range(warm_up):
                _run_once()
            start = time.time()
            for _ in range(repeat):
                outputs = _run_once()
            avg_time = (time.time() - start) * 1000 / repeat
        else:
            outputs = _run_once()
            avg_time = -1
        if isinstance(outputs, tvm.runtime.NDArray):
            outputs = [outputs]
        assert len(output_names) == len(outputs), "Outputs mismatch, {} with {}".format(
            output_names, len(outputs)
        )
        outputs = {
            o_name: msc_utils.cast_array(o_data) for o_name, o_data in zip(output_names, outputs)
        }
        return outputs, avg_time

    @classmethod
    def update_config(cls, stage: str, config: dict, model: Any = None) -> dict:
        """Update the config for parse

        Parameters
        -------
        stage: str
            The stage to be updated
        config: dict
            The config for pipeline.
        model:
            The native model.

        Returns
        -------
        config: dict
            The updated config.
        """

        config = ModelRunner.update_config(stage, config, model)
        if stage not in config:
            return config
        if stage == MSCStage.PARSE:
            # pylint: disable=unused-argument
            def passby(mod, *args, **kwargs):
                return mod, None

            config["parse"]["parser"] = passby
        return config

    @classmethod
    def support_device(cls, device: str) -> bool:
        """Check if the device is enabled

        Returns
        -------
        enabled: bool
            Whether the device is enabled.
        """

        if device == "cpu":
            return True
        if device.startswith("cuda"):
            dev_id = int(device.split(":")[1]) if ":" in device else 0
            return tvm.cuda(dev_id).exist
        return False
