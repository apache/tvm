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
"""tvm.contrib.msc.framework.tensorrt.tools.track.tracker"""

from typing import Dict, List

from tvm.contrib.msc.core.tools.tool import ToolType, ToolStrategy
from tvm.contrib.msc.core.tools.track import BaseTracker
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


class TensorRTTrackerFactory(object):
    """Tracker factory for tensorrt"""

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
            """Adaptive tracker for tensorrt"""

            def _execute_before_build(self, codegen_context: dict) -> dict:
                """Execute before model build

                Parameters
                ----------
                codegen_context: dict
                    The context.

                Returns
                ----------
                codegen_context: dict
                    The processed context.
                """

                self._track_tensors = {}
                super()._execute_before_build(codegen_context)

            def _execute_before_forward(self, step_context: dict) -> dict:
                """Execute before model forward

                Parameters
                ----------
                step_context: dict
                    The context.

                Returns
                ----------
                step_context: dict
                    The processed context.
                """

                for name, data in step_context["datas"].items():
                    if name not in self._track_tensors:
                        continue
                    consumer = self._track_tensors[name]["consumer"]
                    strategys = self._get_tensor_strategys(name, consumer)
                    self._track_tensor(data.asnumpy(), name, consumer, strategys)
                return super()._execute_before_forward(step_context)

            def _execute_after_forward(self, step_context: dict) -> dict:
                """Execute after model forward

                Parameters
                ----------
                step_context: dict
                    The context.

                Returns
                ----------
                step_context: dict
                    The processed context.
                """

                for name, data in step_context["datas"].items():
                    if name not in self._track_tensors:
                        continue
                    consumer = self._track_tensors[name]["consumer"]
                    strategys = self._get_tensor_strategys(name, consumer)
                    self._track_tensor(data.asnumpy(), name, consumer, strategys)
                return super()._execute_after_forward(step_context)

            def _process_tensor(
                self,
                tensor_ctx: Dict[str, str],
                name: str,
                consumer: str,
                scope: str,
                strategys: List[ToolStrategy],
            ) -> Dict[str, str]:
                """Process tensor

                Parameters
                -------
                tensor_ctx: dict<str, str>
                    Tensor describe items.
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
                tensor_ctx: dict<str, str>
                    Tensor items with processed.
                """

                if self.is_weight(name):
                    return self._track_tensor(self.get_data(name), name, consumer, strategys)
                if name not in self._track_tensors:
                    self._track_tensors[name] = {
                        "consumer": consumer,
                    }
                    tensor_ctx["processed"].append(
                        "{}->markOutput(*{});".format(tensor_ctx["ctx"], tensor_ctx["tensor"])
                    )
                return tensor_ctx

            @classmethod
            def framework(cls):
                return MSCFramework.TENSORRT

        return Tracker


factory = TensorRTTrackerFactory()
tools = msc_utils.get_registered_tool_cls(MSCFramework.MSC, ToolType.TRACKER, tool_style="all")
for tool in tools.values():
    msc_utils.register_tool_cls(factory.create(tool))
