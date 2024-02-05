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
# pylint: disable=not-context-manager,unused-import
"""tvm.contrib.msc.framework.tensorflow.runtime.runner"""

import time
from typing import Dict, List, Union, Any, Tuple
import numpy as np

from tensorflow.python.client import device_lib
from tensorflow.python.ops import variables

import tvm
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.runtime import ModelRunner
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.framework.tensorflow.frontend import from_tensorflow
from tvm.contrib.msc.framework.tensorflow.codegen import to_tensorflow
from tvm.contrib.msc.framework.tensorflow import tf_v1
from tvm.contrib.msc.framework.tensorflow import tools


class WrapSession(tf_v1.Session):
    """Wrapped session for MSC"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inputs, self._outputs = None, None

    def set_bindings(self, inputs: List[Dict[str, str]], outputs: List[Dict[str, str]]):
        """Set inputs and outputs for session

        Parameters
        -------
        inputs: list
            The inputs info of the model.
        outputs: list
            The outputs info of the model.
        """

        self._inputs = inputs
        self._outputs = outputs

    def run(self, fetches, *args, **kwargs):  # pylint: disable=useless-parent-delegation
        return super().run(fetches, *args, **kwargs)


class TensorflowRunner(ModelRunner):
    """Runner of Tensorflow"""

    def setup(self) -> dict:
        """Setup the runner

        Returns
        -------
        info: dict
            The setup info.
        """

        self._tf_graph = None
        self._tf_outputs = None
        self._session = None
        return super().setup()

    def destory(self):
        """Destory runner"""

        self._session.close()
        self._tf_graph = None
        self._tf_outputs = None
        self._session = None
        super().destory()

    def _generate_model(
        self, graphs: List[MSCGraph], weights: Dict[str, tvm.nd.array]
    ) -> tf_v1.Graph:
        """Codegen the model according to framework

        Parameters
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: dict<str, tvm.nd.array>
            The weights.

        Returns
        -------
        model: tf_v1.Graph
            The runnable model
        """

        if self._tf_graph:
            del self._tf_graph
        self._tf_graph = tf_v1.Graph()
        with self._tf_graph.as_default():
            self._tf_outputs = super()._generate_model(graphs, weights)
        return self._tf_graph

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

        if self._session:
            self._session.close()
            del self._session
        self._session = WrapSession(graph=self._tf_graph)
        self._session.set_bindings(self.get_inputs(), self.get_outputs())
        with self._tf_graph.as_default():
            self._session.run(variables.global_variables_initializer())
        return self._session

    def _call_runnable(
        self, runnable: WrapSession, inputs: Dict[str, np.ndarray], device: str
    ) -> Union[List[np.ndarray], Dict[str, np.ndarray]]:
        """Call the runnable to get outputs

        Parameters
        -------
        runnable: WrapSession
            The wrapped session.
        inputs: dict<str, data>
            The inputs in dict.
        device: str
            The device.

        Returns
        -------
        outputs: list<data> or dict<str, data>
            The outputs in list or dict.
        """

        feed_dict = {i["name"] + ":0": inputs[i["name"]] for i in self.get_inputs()}
        return runnable.run(self._tf_outputs, feed_dict)

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
            device_protos = device_lib.list_local_devices()
            return any(dev.device_type == "GPU" for dev in device_protos)
        return False

    @property
    def codegen_func(self):
        return to_tensorflow

    @property
    def framework(self):
        return MSCFramework.TENSORFLOW

    @classmethod
    def load_native(cls, model: Any) -> Tuple[tf_v1.GraphDef, str, bool]:
        """Load the native model

        Parameters
        -------
        model:
            The native model.

        Returns
        -------
        model: tf_v1.GraphDef
            The loaded native model.
        device: str
            The device of the model.
        training:
            Whether the model is for training.
        """

        if isinstance(model, tf_v1.GraphDef):
            native_model = model
        else:
            raise NotImplementedError(
                "Load native model {} with type {} is not supported".format(model, type(model))
            )
        device_protos = device_lib.list_local_devices()
        if any(dev.device_type == "GPU" for dev in device_protos):
            device = "cuda"
        else:
            device = "cpu"
        return native_model, device, False

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
            config["parse"]["parser"] = from_tensorflow
            parse_config = config["parse"].get("parse_config", {})
            parse_config.update(
                {
                    "shape_dict": {i[0]: i[1] for i in config["inputs"]},
                    "outputs": config["outputs"],
                }
            )
            config["parse"]["parse_config"] = parse_config
        return config

    @classmethod
    def run_native(
        cls,
        model: tf_v1.GraphDef,
        inputs: Dict[str, np.ndarray],
        input_names: List[str],
        output_names: List[str],
        warm_up: int = 10,
        repeat: int = 0,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """Run the datas and get outputs

        Parameters
        -------
        model: tf_v1.GraphDef
            The graph def.
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

        feed_dict = {i_name + ":0": inputs[i_name] for i_name in input_names}
        with tf_v1.Graph().as_default():
            tf_v1.import_graph_def(model, name="")
            with tf_v1.Session() as sess:
                if repeat > 0:
                    for _ in range(warm_up):
                        outputs = sess.run(output_names, feed_dict)
                    start = time.time()
                    for _ in range(repeat):
                        outputs = sess.run(output_names, feed_dict)
                    avg_time = (time.time() - start) * 1000 / repeat
                else:
                    outputs = sess.run(output_names, feed_dict)
                    avg_time = -1
        outputs = dict(zip(output_names, outputs))
        return outputs, avg_time
