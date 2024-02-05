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
"""tvm.contrib.msc.framework.torch.runtime.runner"""

import os
import time
from typing import Dict, List, Union, Tuple, Any
import numpy as np

import torch
import tvm
from tvm.contrib.msc.core.runtime import ModelRunner
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.framework.torch.frontend import from_torch
from tvm.contrib.msc.framework.torch.codegen import to_torch
from tvm.contrib.msc.framework.torch.frontend import set_weight_alias
from tvm.contrib.msc.framework.torch import tools


class TorchRunner(ModelRunner):
    """Runner of Torch"""

    def _translate(self, mod: tvm.IRModule) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Translate IRModule to MSCgraphs

        Parameters
        -------
        mod: tvm.IRModule
            The module to be translated.

        Returns
        -------
        graph_list: list<MSCGraph>
            The translated graphs
        weights_list: list<dict<str, tvm.nd.array>>
            The translated weights
        """
        graphs, weights = super()._translate(mod)
        return [set_weight_alias(graphs[0])], weights

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

        if self._device.startswith("cpu"):
            pass
        elif self._device.startswith("cuda"):
            model = model.to(torch.device(self._device))
        else:
            raise NotImplementedError("Unsupported device " + str(self._device))
        if self._training:
            model = model.train()
        else:
            model = model.eval()
        return model

    def _call_runnable(
        self, runnable: torch.nn.Module, inputs: Dict[str, np.ndarray], device: str
    ) -> Union[List[np.ndarray], Dict[str, np.ndarray]]:
        """Call the runnable to get outputs

        Parameters
        -------
        runnable: torch.nn.Module
            The runnable model.
        inputs: dict<str, data>
            The inputs in dict.
        device: str
            The device.

        Returns
        -------
        outputs: list<torch.Tensor>
            The outputs in list.
        """

        model_inputs = self.get_inputs()
        parameters = list(runnable.parameters())
        if parameters:
            in_dev = parameters[0].device
        elif device == "cpu":
            in_dev = torch.device(device)
        elif device.startswith("cuda"):
            in_dev = torch.device(device)
        else:
            raise NotImplementedError("Unsupported device " + str(device))
        torch_inputs = [torch.from_numpy(inputs[i["name"]]).to(in_dev) for i in model_inputs]
        return runnable(*torch_inputs)

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
            return torch.cuda.is_available()
        return False

    @property
    def codegen_func(self):
        return to_torch

    @property
    def framework(self):
        return MSCFramework.TORCH

    @classmethod
    def load_native(cls, model: Any) -> Tuple[torch.nn.Module, str, bool]:
        """Load the native model

        Parameters
        -------
        model:
            The native model.

        Returns
        -------
        model: torch.nn.Module
            The loaded native model.
        device: str
            The device of the model.
        training:
            Whether the model is for training.
        """

        if isinstance(model, dict) and "model" in model:
            native_model = msc_utils.load_callable(model["model"])
        elif isinstance(model, torch.nn.Module):
            native_model = model
        else:
            raise NotImplementedError(
                "Load native model {} with type {} is not supported".format(model, type(model))
            )
        parameters = list(model.parameters())
        if parameters:
            ref_device = parameters[0].device
            if ref_device.index:
                device = "{}:{}".format(ref_device.type, ref_device.index)
            else:
                device = ref_device.type
        else:
            device = "cpu"
        return native_model, device, model.training

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
            config["parse"]["parser"] = from_torch
            parse_config = config["parse"].get("parse_config", {})
            parse_config.update(
                {
                    "input_info": [
                        [i[1], "float" if len(i) < 2 else i[2]] for i in config["inputs"]
                    ],
                    "input_names": [i[0] for i in config["inputs"]],
                }
            )
            config["parse"]["parse_config"] = parse_config
        return config

    @classmethod
    def run_native(
        cls,
        model: torch.nn.Module,
        inputs: Dict[str, np.ndarray],
        input_names: List[str],
        output_names: List[str],
        warm_up: int = 10,
        repeat: int = 0,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """Run the datas and get outputs

        Parameters
        -------
        model: torch.nn.Module
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

        parameters = list(model.parameters())
        if parameters:
            device = parameters[0].device
        else:
            device = torch.device("cpu")
        torch_inputs = [torch.from_numpy(inputs[i_name]).to(device) for i_name in input_names]

        def _run_once():
            return model(*torch_inputs)

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
        if isinstance(outputs, torch.Tensor):
            assert len(output_names) == 1, "Expect 1 outputs, get " + str(output_names)
            return {output_names[0]: msc_utils.cast_array(outputs)}, avg_time
        assert len(output_names) == len(outputs), "Outputs mismatch, {} with {}".format(
            output_names, len(outputs)
        )
        outputs = {
            o_name: msc_utils.cast_array(o_data) for o_name, o_data in zip(output_names, outputs)
        }
        return outputs, avg_time

    @classmethod
    def dump_nativate(cls, model: torch.nn.Module, folder: msc_utils.MSCDirectory) -> str:
        """Dump the nativate model

        Parameters
        -------
        model: torch.nn.Module
            The runnable model.
        folder: MSCDirectory
            The export folder.

        Returns
        -------
        export_path: str
            The exported path
        """

        graph_model = torch.fx.symbolic_trace(model)
        exp_path = folder.create_dir("model")
        graph_model.to_folder(exp_path.path, "native_model")
        return {"model": exp_path.relpath("module.py") + ":native_model"}
