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
"""tvm.contrib.msc.core.tools.distill.distiller"""

import os
from typing import List, Any, Dict, Tuple

import tvm
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.tools.tool import ToolType, BaseTool, ToolStrategy
from tvm.contrib.msc.core import utils as msc_utils


class BaseDistiller(BaseTool):
    """Base distiller for all"""

    def setup(self) -> dict:
        """Setup the tool

        Returns
        -------
        info: dict
            The setup info.
        """

        self._max_iter = self._options.get("max_iter", 1)
        self._save_step = self._options.get("save_step", 50)
        if "weights_folder" in self._options:
            self._weights_folder = msc_utils.msc_dir(self._options["weights_folder"])
        else:
            self._weights_folder = msc_utils.get_weights_dir().create_dir("Distill")
        self._weights_path = self._weights_folder.relpath("distill_{}.bin".format(self._max_iter))
        self._distilled = os.path.isfile(self._weights_path)
        return super().setup()

    def _reset(
        self, graphs: List[MSCGraph], weights: Dict[str, tvm.nd.array]
    ) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Reset the tool

        Parameters
        ----------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: dict<str, tvm.nd.array>
            The weights.

        Returns
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: dict<str, tvm.nd.array>
            The weights.
        """

        self._current_iter, self._total_loss = 0, 0
        if self._distilled:
            with open(self._weights_path, "rb") as f:
                distilled_weights = tvm.runtime.load_param_dict(f.read())
            weights.update({k: v for k, v in distilled_weights.items() if k in weights})
            msg = "Update {} distilled weights".format(len(distilled_weights))
            self._logger.info(self.tool_mark(msg))
        return super()._reset(graphs, weights)

    def build_model(self, teacher: Any, student: Any) -> Any:
        """Build the model with teacher and student

        Parameters
        ----------
        teacher: Any
            The teacher model
        student: Any
            The student model

        Returns
        -------
        model: Any
            The built model.
        """

        raise NotImplementedError("build_model is not implemented in BaseDistiller")

    def learn(self, loss: Any):
        """Learn after forward

        Parameters
        ----------
        loss: Any
            The loss after forward
        """

        if self.on_debug(3, in_forward=False):
            msg = "Start learn[{}]".format(self._current_iter)
            self._logger.debug(self.tool_mark(msg))
        self._total_loss += float(self._learn(loss))

    def _learn(self, loss: Any):
        """Learn after forward

        Parameters
        ----------
        loss: Any
            The loss after forward
        """

        raise NotImplementedError("_learn is not implemented in BaseDistiller")

    def distill(self) -> Dict[str, Any]:
        """Distill the knowledge

        Returns
        -------
        weights: dict<str, Any>
            The distilled weights.
        """

        weights = self._distill()
        if self._current_iter >= self._max_iter or (
            self._current_iter > 0 and self._current_iter % self._save_step == 0
        ):
            self._save_weights(weights)
        if self._current_iter >= self._max_iter:
            self._distilled = True
            self._plan = {n: msc_utils.inspect_array(d, False) for n, d in weights.items()}
        msg = "Distill[{}] loss({} batch) {}".format(
            self._current_iter, self._forward_cnt, self._total_loss
        )
        self._logger.info(self.tool_mark(msg))
        self._current_iter += 1
        self._total_loss, self._forward_cnt = 0, 0
        return weights

    def _distill(self) -> Dict[str, Any]:
        """Distill the knowledge

        Returns
        -------
        weights: dict<str, Any>
            The distilled weights.
        """

        raise NotImplementedError("_distill is not implemented in BaseDistiller")

    def _save_weights(self, weights: Dict[str, Any]):
        """Save the distilled weights

        Parameters
        ----------
        weights: dict<str, Any>
            The distilled weights.
        """

        weights = {n: tvm.nd.array(msc_utils.cast_array(d)) for n, d in weights.items()}
        weights_path = self._weights_folder.relpath("distill_{}.bin".format(self._current_iter))
        with open(weights_path, "wb") as f_params:
            f_params.write(tvm.runtime.save_param_dict(weights))
        if self._debug_level >= 2:
            msg = "Save weights[{}] to {}".format(self._current_iter, weights_path)
            self._logger.debug(self.tool_mark(msg))

    def _support_scope(self, scope: str) -> bool:
        """Check if the scope si supported

        Parameters
        -------
        scope: str
            The scope mark, should be null or ToolScope

        Returns
        -------
        vaild: bool
            Whether to process the tensor.
        """

        return True

    def _process_tensor(
        self, tensor: Any, name: str, consumer: str, scope: str, strategys: List[ToolStrategy]
    ) -> Any:
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

        if self._distilled:
            return tensor
        return self._distill_tensor(tensor, name, consumer, scope, strategys)

    def _distill_tensor(
        self, tensor: Any, name: str, consumer: str, scope: str, strategys: List[ToolStrategy]
    ) -> Any:
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

        if name not in self._plan:
            self._plan[name] = {}
        plan = {}
        for strategy in strategys:
            plan.update(strategy(self, tensor, name, consumer, scope))
        self._plan[name][scope] = plan
        return tensor

    @property
    def distilled(self):
        return self._distilled

    @classmethod
    def tool_type(cls):
        return ToolType.DISTILLER

    @classmethod
    def exportable(cls):
        return False


@msc_utils.register_tool
class DefaultDistiller(BaseDistiller):
    @classmethod
    def tool_style(cls):
        return "default"
