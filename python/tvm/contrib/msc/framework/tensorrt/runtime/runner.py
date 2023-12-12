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
"""tvm.contrib.msc.framework.tensorrt.runtime.runner"""

from typing import Any, List, Dict

import tvm
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.runtime import BYOCRunner
from tvm.contrib.msc.core.tools import ToolType
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.framework.tensorrt.frontend import (
    partition_for_tensorrt,
    transform_for_tensorrt,
)
from tvm.contrib.msc.framework.tensorrt.codegen import to_tensorrt
from tvm.contrib.msc.framework.tensorrt import tools


class TensorRTRunner(BYOCRunner):
    """Runner of tensorrt"""

    def setup(self) -> dict:
        """Setup the runner

        Returns
        -------
        info: dict
            The setup info.
        """

        if not self._device.startswith("cuda"):
            self._device = "cuda"
        return super().setup()

    def apply_tool(self, tool_type: str, data_loader: Any = None) -> dict:
        """Execute tool and get plan

        Parameters
        -------
        tool_type: str
            The tool type, should be in ToolType
        data_loader:
            The data loader
        """

        assert tool_type in self._tools, "Can not find tool " + str(tool_type)
        if tool_type == ToolType.QUANTIZER:
            quantizer = self.get_tool(ToolType.QUANTIZER)
            assert data_loader, "data_loader should be given to plan prune"
            for inputs in data_loader():
                self.run(inputs)
            self._generate_model()
            quantizer.calibrate()
            assert quantizer.calibrated, "Failed to calibrate the tenosrrt quantizer"
        return super().apply_tool(tool_type, data_loader)

    def _generate_model(
        self, graphs: List[MSCGraph] = None, weights: List[Dict[str, tvm.nd.array]] = None
    ) -> Any:
        """Codegen the model according to framework

        Parameters
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dict<str, tvm.nd.array>>
            The weights

        Returns
        -------
        model: Any
            The meta model
        """

        codegen = self._generate_config.get("codegen")
        if not isinstance(codegen, (list, tuple)):
            self._generate_config["codegen"] = [msc_utils.copy_dict(codegen)] * len(self._graphs)
        for tool in self.get_tools():
            self._generate_config = tool.config_generate(self._generate_config)

        return super()._generate_model(graphs, weights)

    @classmethod
    def target_transform(cls, mod: tvm.IRModule):
        """Transform the mod by target.

        Parameters
        ----------
        mod: IRModule
            The IRModule of relax.

        Returns
        -------
        mod: IRModule
            The IRModule of partitioned relax.
        """

        return transform_for_tensorrt(mod)

    @property
    def codegen_func(self):
        return to_tensorrt

    @property
    def partition_func(self):
        return partition_for_tensorrt

    @property
    def framework(self):
        return MSCFramework.TENSORRT
