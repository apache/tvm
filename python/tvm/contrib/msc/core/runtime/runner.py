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
# pylint: disable=unused-argument
"""tvm.contrib.msc.core.runtime.runner"""

import logging
from typing import Dict, Optional, Any, List, Tuple, Union
import numpy as np

import tvm
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.frontend import from_relax
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.core import _ffi_api


class BaseRunner(object):
    """Basic runner of MSC

    Parameters
    ----------
    mod: IRModule
        The IRModule of relax.
    params: dict of <string:tvm.ndarray>
        The parameters of the IRModule.
    tools_config: dict
        The config of MSC Tools.
    translate_config: dict
        The config for translate IRModule to MSCGraph.
    codegen_config: dict
        The config for build MSCGraph to runnable model.
    name: str
        The name of the runner
    device: str
        The device of the model, cpu| gpu
    is_training: bool
        Whether use model in training
    logger: logging.Logger
        The logger
    """

    def __init__(
        self,
        mod: tvm.IRModule,
        tools_config: Optional[Dict[str, Any]] = None,
        translate_config: Optional[Dict[str, str]] = None,
        load_config: Optional[Dict[str, str]] = None,
        name: str = "main",
        device: str = "cpu",
        is_training: bool = False,
        logger: logging.Logger = None,
    ):
        self._mod = mod
        self._tools_config = tools_config or {}
        self._translate_config = translate_config or {}
        self._load_config = load_config or {}
        self._name = name
        self._device = device if self._device_enabled(device) else "cpu"
        self._is_training = is_training
        self._logger = logger or msc_utils.get_global_logger()
        self.setup()

    def setup(self):
        """Setup the runner"""
        self._graphs, self._weights = [], []
        self._model, self._model_info = None, {}

    def build(self, build_graph: bool = False) -> object:
        """Build the runnable object

        Parameters
        -------
        build_graph: bool
            Whether to build the MSCGraphs.

        Returns
        -------
        model: object
           The runnable object.
        """

        # Get or rebuild graphs
        if build_graph or not self._graphs:
            self._graphs, self._weights = self._translate()
            self._model_info = self._inspect_model()
            self._logger.info("Translate {} graphs from module".format(len(self._graphs)))

        # Save graphs for debug
        for graph in self._graphs:
            graph.visualize(msc_utils.get_debug_dir().relpath(graph.name + ".prototxt"))

        # Create tools
        if self._tools_config:
            raise NotImplementedError("Build runner with tools is not supported")

        # Load model
        model = self._generate_model()
        if "loader" in self._load_config:
            loader, load_config = self._load_config["loader"]
            model = loader(model, **load_config)
            self._logger.info(
                "Model({}) processed by customize loader {}({})".format(
                    self.framework, loader, load_config
                )
            )
        self._model = self._to_device(model, self._device, self._is_training)

        self._logger.info(
            "Model({}, is_training {}) loaded on device {}".format(
                self.framework, self._is_training, self._device
            )
        )
        return self._model

    def run(
        self, inputs: Union[List[np.ndarray], Dict[str, np.ndarray]], ret_type="dict"
    ) -> Union[List[np.ndarray], Dict[str, np.ndarray]]:
        """Run the model to get outputs

        Parameters
        -------
        inputs: list<data> or dict<str, data>
            The inputs in list or dict.
        ret_type: str
            The return type list| dict

        Returns
        -------
        outputs: dict<str, data>
            The outputs in dict.
        """

        model_inputs = self.get_inputs()
        model_outputs = self.get_outputs()
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == len(
                model_inputs
            ), "inputs({}) mismatch with model inputs {}".format(len(inputs), model_inputs)
            inputs = {info["name"]: data for info, data in zip(model_inputs, inputs)}
        assert isinstance(inputs, dict), "Expect inputs as list or dict, get {}({})".format(
            inputs, type(inputs)
        )
        assert all(
            isinstance(data, np.ndarray) for data in inputs.values()
        ), "Expected all inputs as np.ndarray"
        inputs = {i["name"]: inputs[i["name"]] for i in model_inputs}
        outputs = self._run_model(self._model, inputs, self._device)
        if ret_type == "dict":
            if isinstance(outputs, (list, tuple)):
                assert len(outputs) == len(
                    model_outputs
                ), "outputs({}) mismatch with model outputs {}".format(len(outputs), model_outputs)
                outputs = {info["name"]: data for info, data in zip(model_outputs, outputs)}
            if not isinstance(outputs, dict):
                assert len(model_outputs) == 1, "Expect model_outputs with len 1, get " + str(
                    model_outputs
                )
                outputs = {model_outputs[0]["name"]: outputs}
            outputs = {name: msc_utils.cast_array(data) for name, data in outputs.items()}
        elif ret_type == "list":
            if isinstance(outputs, dict):
                assert len(outputs) == len(
                    model_outputs
                ), "outputs({}) mismatch with model outputs {}".format(len(outputs), model_outputs)
                outputs = [outputs[o["name"]] for o in model_outputs]
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            outputs = [msc_utils.cast_array(data) for data in outputs]
        return outputs

    def get_inputs(self) -> List[Dict[str, str]]:
        """Get the inputs of the model

        Returns
        -------
        inputs: list<tensor_des>
            The inputs info.
        """

        return self._model_info["inputs"]

    def get_outputs(self) -> List[Dict[str, str]]:
        """Get the outputs of the model

        Returns
        -------
        outputs: list<tensor_des>
            The outputs info.
        """

        return self._model_info["outputs"]

    def get_model(self) -> object:
        """Get the model

        Returns
        -------
        model:
            The runnable model.
        """

        return self._model

    def _translate(self) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Translate IRModule to MSCgraphs

        Returns
        -------
        graph_list: list<MSCGraph>
            The translated graphs
        weights_list: list<dict<str, tvm.nd.array>>
            The translated weights
        """

        raise NotImplementedError("_translate is not implemented for " + str(self.__class__))

    def _generate_model(self) -> object:
        """Codegen the model according to framework

        Returns
        -------
        model: object
            The runnable model
        """

        raise NotImplementedError("_load is not implemented for " + str(self.__class__))

    def _to_device(self, model: object, device: str, is_training: bool) -> object:
        """Place model on device

        Parameters
        -------
        model: object
            The runnable model on cpu.
        device: str
            The device for place model
        is_training: bool
            Whether to load model for training

        Returns
        -------
        model: object
            The runnable model
        """

        raise NotImplementedError("_to_device is not implemented for " + str(self.__class__))

    def _inspect_model(self) -> dict:
        """Inspect the model

        Returns
        -------
        model_info: dict
            The inspected model info
        """

        raise NotImplementedError("_inspect_model is not implemented for " + str(self.__class__))

    def _run_model(
        self, model: object, inputs: Dict[str, np.ndarray], device: str
    ) -> Union[List[np.ndarray], Dict[str, np.ndarray]]:
        """Run the model to get outputs

        Parameters
        -------
        model:
            The runnable model.
        inputs: dict<str, data>
            The inputs in dict.
        device: str
            The device.

        Returns
        -------
        outputs: list<data> or dict<str, data>
            The outputs in list or dict.
        """

        raise NotImplementedError("_run_model is not implemented for " + str(self.__class__))

    def _device_enabled(self, device: str) -> bool:
        """Check if the device is enabled

        Returns
        -------
        enabled: bool
            Whether the device is enabled.
        """

        return True

    @property
    def codegen_func(self):
        raise NotImplementedError("codegen_func is not implemented for " + str(self.__class__))

    @property
    def framework(self):
        return MSCFramework.MSC


class ModelRunner(BaseRunner):
    """Model runner of MSC"""

    def _translate(self) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Translate IRModule to MSCgraphs

        Returns
        -------
        graph_list: list<MSCGraph>
            The translated graphs
        weights_list: list<dict<str, tvm.nd.array>>
            The translated weights
        """

        graph, weights = from_relax(
            self._mod,
            trans_config=self._translate_config.get("transform"),
            build_config=self._translate_config.get("build"),
            opt_config=self._translate_config.get("optimize"),
        )
        return [graph], [weights]

    def _generate_model(self) -> object:
        """Codegen the model according to framework

        Returns
        -------
        model: object
            The runnable model
        """

        return self.codegen_func(
            self._graphs[0],
            self._weights[0],
            codegen_config=self._load_config.get("codegen"),
            print_config=self._load_config.get("build"),
            build_folder=msc_utils.get_build_dir(),
        )

    def _inspect_model(self) -> dict:
        """Inspect the model

        Returns
        -------
        model_info: dict
            The inspected model info
        """

        return self._graphs[0].inspect()


class BYOCRunner(BaseRunner):
    """BYOC runner of MSC"""

    def setup(self):
        """Setup the runner"""

        super().setup()
        self._byoc_mod, self._byoc_graph = None, None
        self._graph_infos = {}

    def _translate(self) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Translate IRModule to MSCgraphs

        Returns
        -------
        graph_list: list<MSCGraph>
            The translated graphs
        weights_list: list<dict<str, tvm.nd.array>>
            The translated weights
        """

        self._byoc_mod, self._graph_infos = self.partition_func(
            self._mod,
            trans_config=self._translate_config.get("transform"),
            build_config=self._translate_config.get("build"),
        )
        graphs, weights = [], []
        for graph, sub_weights in self._graph_infos:
            graphs.append(graph)
            weights.append(sub_weights)
        self._byoc_graph = _ffi_api.BuildFromRelax(
            self._byoc_mod, "main", msc_utils.dump_dict(self._translate_config.get("build"))
        )
        self._byoc_graph.visualize(
            msc_utils.get_debug_dir().relpath(self._byoc_graph.name + ".prototxt")
        )
        return graphs, weights

    def _generate_model(self) -> tvm.IRModule:
        """Codegen the model according to framework

        Returns
        -------
        model: tvm.IRModule
            The relax module
        """

        return self.codegen_func(
            self._byoc_mod,
            self._graph_infos,
            codegen_config=self._load_config.get("codegen"),
            print_config=self._load_config.get("build"),
            build_folder=msc_utils.get_build_dir(),
            output_folder=msc_utils.get_output_dir(),
        )

    def _to_device(self, model: object, device: str, is_training: bool) -> object:
        """Place model on device

        Parameters
        -------
        model: object
            The runnable model on cpu.
        device: str
            The device for place model
        is_training: bool
            Whether to load model for training

        Returns
        -------
        model: object
            The runnable model
        """

        model = tvm.relax.transform.LegalizeOps()(model)
        if device == "cpu":
            target = tvm.target.Target("llvm")
            with tvm.transform.PassContext(opt_level=3):
                relax_exec = tvm.relax.build(model, target)
                runnable = tvm.relax.VirtualMachine(relax_exec, tvm.cpu())
        elif device == "gpu":
            target = tvm.target.Target("cuda")
            with target:
                model = tvm.tir.transform.DefaultGPUSchedule()(model)
            with tvm.transform.PassContext(opt_level=3):
                relax_exec = tvm.relax.build(model, target)
                runnable = tvm.relax.VirtualMachine(relax_exec, tvm.cuda())
        else:
            raise NotImplementedError("Unsupported device " + str(device))
        return runnable

    def _run_model(
        self, model: tvm.relax.VirtualMachine, inputs: Dict[str, np.ndarray], device: str
    ) -> Union[List[np.ndarray], Dict[str, np.ndarray]]:
        """Run the model to get outputs

        Parameters
        -------
        model: tvm.relax.VirtualMachine
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
        elif device == "gpu":
            tvm_inputs = [tvm.nd.array(inputs[i["name"]], device=tvm.cuda()) for i in model_inputs]
        else:
            raise NotImplementedError("Unsupported device " + str(device))
        return model["main"](*tvm_inputs)

    def _inspect_model(self) -> dict:
        """Inspect the model

        Returns
        -------
        model_info: dict
            The inspected model info
        """

        return self._byoc_graph.inspect()

    def _device_enabled(self, device: str) -> bool:
        """Check if the device is enabled

        Returns
        -------
        enabled: bool
            Whether the device is enabled.
        """

        if device == "cpu":
            return True
        if device == "gpu":
            return tvm.cuda().exist
        return False

    @property
    def partition_func(self):
        raise NotImplementedError("partition_func is not implemented for " + str(self.__class__))
