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
"""tvm.contrib.msc.framework.runtime.tvm.runner"""

from typing import Dict, List, Union
import numpy as np

import tvm
from tvm.contrib.msc.core.runtime import ModelRunner
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.framework.tvm.codegen import to_relax


class TVMRunner(ModelRunner):
    """Runner of Relax"""

    def _to_runnable(self, model: object, device: str, is_training: bool) -> object:
        """Build runnable object

        Parameters
        -------
        model: object
            The meta model.
        device: str
            The device for place model
        is_training: bool
            Whether to load model for training

        Returns
        -------
        runnable: object
            The runnable
        """

        if "builder" in self._load_config:
            builder, build_config = self._load_config["builder"]
            runnable = builder(model, **build_config)
            self._logger.info(
                "Model({}) processed by customize builder {}({})".format(
                    self.framework, builder, build_config
                )
            )
        else:
            model = tvm.relax.transform.LegalizeOps()(model)
            if device == "cpu":
                target = tvm.target.Target("llvm")
                with tvm.transform.PassContext(opt_level=3):
                    relax_exec = tvm.relax.build(model, target)
                    runnable = tvm.relax.VirtualMachine(relax_exec, tvm.cpu())
            elif device.startswith("cuda"):
                target = tvm.target.Target("cuda")
                with target:
                    model = tvm.tir.transform.DefaultGPUSchedule()(model)
                with tvm.transform.PassContext(opt_level=3):
                    relax_exec = tvm.relax.build(model, target)
                    runnable = tvm.relax.VirtualMachine(relax_exec, tvm.cuda())
            else:
                raise NotImplementedError("Unsupported device " + str(device))
        return runnable

    def _call_runnable(
        self, runnable: tvm.relax.VirtualMachine, inputs: Dict[str, np.ndarray], device: str
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

        model_inputs = self.get_inputs()
        if device == "cpu":
            tvm_inputs = [tvm.nd.array(inputs[i["name"]]) for i in model_inputs]
        elif device.startswith("cuda"):
            dev_id = int(device.split(":")[1]) if ":" in device else 0
            tvm_inputs = [
                tvm.nd.array(inputs[i["name"]], device=tvm.cuda(dev_id)) for i in model_inputs
            ]
        else:
            raise NotImplementedError("Unsupported device " + str(device))
        return runnable["main"](*tvm_inputs)

    def _device_enabled(self, device: str) -> bool:
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

    @property
    def codegen_func(self):
        return to_relax

    @property
    def framework(self):
        return MSCFramework.TVM
