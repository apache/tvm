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
"""tvm.contrib.msc.framework.tvm.tools.track.tracker"""

from typing import List, Union

import tvm
from tvm.contrib.msc.core.tools.tool import ToolType, ToolStrategy
from tvm.contrib.msc.core.tools.track import BaseTracker
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


class TVMTrackerFactory(object):
    """Tracker factory for tvm"""

    def create(self, base_cls: BaseTracker) -> BaseTracker:
        """Create adaptive tracker

        Parameters
        ----------
        base_cls: BaseTracker
            The base tracker class

        Returns
        -------
        tracker_cls: BaseTracker
            The tracker class.
        """

        class Tracker(base_cls):
            """Adaptive tracker for tvm"""

            def _execute_before_build(self, block_builder: tvm.relax.BlockBuilder):
                """Execute before model build

                Parameters
                ----------
                block_builder: tvm.relax.BlockBuilder
                    The block builder.
                """

                self._block_builder = block_builder
                self._track_tensors, self._track_names = {}, []
                super()._execute_before_build(block_builder)

            def _execute_after_build(
                self, output: Union[tvm.relax.Var, List[tvm.relax.DataflowVar]]
            ) -> List[tvm.relax.Var]:
                """Execute after model build

                Parameters
                ----------
                output: var or list<var>
                    The output var of the model.

                Returns
                -------
                outputs: list<var>
                    The modified outputs var.
                """

                self._track_names = list(sorted(self._track_tensors.keys()))
                track_tensors = [self._track_tensors[o]["tensor"] for o in self._track_names]
                if isinstance(output, tvm.relax.Var):
                    return super()._execute_after_build([output] + track_tensors)
                return super()._execute_after_build(output + track_tensors)

            def _execute_after_forward(
                self, outputs: List[tvm.runtime.NDArray]
            ) -> Union[tvm.runtime.NDArray, List[tvm.runtime.NDArray]]:
                """Execute after model forward

                Parameters
                ----------
                outputs: list<np.ndarray>
                    The output datas.

                Returns
                -------
                output: np.ndarray or list<np.ndarray>
                    The modified output ndarray.
                """

                output_num = len(outputs) - len(self._track_names)
                for data, name in zip(outputs[output_num:], self._track_names):
                    consumer = self._track_tensors[name]["consumer"]
                    strategys = self._get_tensor_strategys(name, consumer)
                    producer = self.find_producer(name)
                    if producer == "nn.batch_norm":
                        data = data[0]
                    self._track_tensor(data, name, consumer, strategys)
                if output_num == 1:
                    return super()._execute_after_forward(outputs[0])
                return super()._execute_after_forward(outputs[:output_num])

            def _process_tensor(
                self,
                tensor: tvm.relax.DataflowVar,
                name: str,
                consumer: str,
                scope: str,
                strategys: List[ToolStrategy],
            ) -> tvm.relax.DataflowVar:
                """Process tensor

                Parameters
                -------
                tensor: Any
                    Tensor in framework
                name: str
                    The name of the tensor.
                consumer: str
                    The name of the consumer.
                scope: str
                    The scope mark teacher| student| null.
                strategys: list<ToolStrategy>
                    The strategys for the tensor.

                Returns
                -------
                tensor: Any
                    The processed tensor.
                """

                if self.is_weight(name):
                    self._track_tensor(self.get_data(name), name, consumer, strategys)
                if name not in self._track_tensors:
                    self._track_tensors[name] = {"consumer": consumer, "tensor": tensor}
                    self._track_names.append(name)
                return tensor

            @classmethod
            def framework(cls):
                return MSCFramework.TVM

        return Tracker


factory = TVMTrackerFactory()
tools = msc_utils.get_registered_tool_cls(MSCFramework.MSC, ToolType.TRACKER, tool_style="all")
for tool in tools.values():
    msc_utils.register_tool_cls(factory.create(tool))
