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
"""tvm.contrib.msc.framework.tvm.tools.quantize.quantizer"""

from typing import List, Union

import tvm
from tvm.contrib.msc.core.tools.tool import ToolType, ToolStrategy
from tvm.contrib.msc.core.tools.quantize import BaseQuantizer
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


class TVMQuantizerFactory(object):
    """Quantizer factory for tvm"""

    def create(self, base_cls: BaseQuantizer) -> BaseQuantizer:
        """Create adaptive quantizer

        Parameters
        ----------
        base_cls: BaseQuantizer
            The base quantizer class

        Returns
        -------
        quantizer_cls: BaseQuantizer
            The quantizer class.
        """

        @msc_utils.register_tool
        class Quantizer(base_cls):
            """Adaptive quantizer for tvm"""

            def _execute_before_build(self, block_builder: tvm.relax.BlockBuilder):
                """Execute before model build

                Parameters
                ----------
                block_builder: tvm.relax.BlockBuilder
                    The block builder.
                """

                self._block_builder = block_builder
                self._gather_tensors, self._gather_names = {}, []
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

                if self._calibrated:
                    return super()._execute_after_build(output)
                self._gather_names = list(sorted(self._gather_tensors.keys()))
                gather_tensors = [self._gather_tensors[o]["tensor"] for o in self._gather_names]
                if isinstance(output, tvm.relax.Var):
                    return super()._execute_after_build([output] + gather_tensors)
                return super()._execute_after_build(output + gather_tensors)

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

                if self._calibrated:
                    return super()._execute_after_forward(outputs)
                output_num = len(outputs) - len(self._gather_names)
                for data, name in zip(outputs[output_num:], self._gather_names):
                    info = self._gather_tensors[name]
                    for consumer in info["consumers"]:
                        strategys = self._get_tensor_strategys(name, consumer)
                        self._gather_tensor(data, name, consumer, strategys)
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

                if not self._calibrated:
                    if self.is_weight(name):
                        return self._gather_tensor(self.get_data(name), name, consumer, strategys)
                    if name not in self._gather_tensors:
                        self._gather_tensors[name] = {
                            "consumers": [consumer],
                            "tensor": tensor,
                        }
                        self._gather_names.append(name)
                    else:
                        self._gather_tensors[name]["consumers"].append(consumer)
                    return tensor
                return self._quantize_tensor(tensor, name, consumer, strategys)

            @classmethod
            def framework(cls):
                return MSCFramework.TVM

        return Quantizer


factory = TVMQuantizerFactory()
tools = msc_utils.get_registered_tool(MSCFramework.MSC, ToolType.QUANTIZER, tool_style="all")
for tool in tools.values():
    factory.create(tool)
