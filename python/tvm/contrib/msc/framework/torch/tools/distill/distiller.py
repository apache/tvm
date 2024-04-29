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
"""tvm.contrib.msc.framework.torch.tools.distill.distiller"""

from typing import Any, Dict

import torch
from torch import optim
from tvm.contrib.msc.core.tools.tool import ToolType
from tvm.contrib.msc.core.tools.distill import BaseDistiller
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


class TorchDistillerFactory(object):
    """Distiller factory for torch"""

    def create(self, base_cls: BaseDistiller) -> BaseDistiller:
        """Create adaptive distiller

        Parameters
        ----------
        base_cls: BaseDistiller
            The base distiller class

        Returns
        -------
        distiller_cls: BaseDistiller
            The distiller class.
        """

        @msc_utils.register_tool
        class Distiller(base_cls):
            """Adaptive distiller for torch"""

            def build_model(self, teacher: Any, student: Any) -> Any:
                """Build the model with teacher and student

                Parameters
                -------
                teacher: Any
                    The teacher model
                student: Any
                    The student model

                Returns
                -------
                model: Any
                    The built model.
                """

                optimizer = self._options.get("optimizer", "sgd")
                opt_config = {"lr": 0.0001, "weight_decay": 1e-4}
                opt_config.update(self._options.get("opt_config", {}))
                self._logger.debug(
                    "%s build model with optimizer %s(%s)",
                    self.tool_type().upper(),
                    optimizer,
                    opt_config,
                )
                if optimizer == "sgd":
                    self._optimizer = optim.SGD(student.parameters(), **opt_config)
                elif optimizer == "adam":
                    self._optimizer = optim.Adam(student.parameters(), **opt_config)
                else:
                    raise NotImplementedError("optimizer {} is not supported".format(optimizer))

                # Get loss function
                loss_strategy = self._strategys.get("loss")
                assert loss_strategy, "Can not find loss in strategys"

                def get_loss(teacher_outputs, student_outputs):
                    return loss_strategy(self, teacher_outputs, student_outputs)

                # Build model
                class DistillModel(torch.nn.Module):
                    """Common distill model class"""

                    def __init__(self):
                        super(DistillModel, self).__init__()
                        self.teacher = teacher
                        self.student = student

                    def forward(self, *inputs):
                        with torch.no_grad():
                            teacher_outputs = self.teacher.forward(*inputs)
                        student_outputs = self.student.forward(*inputs)
                        return get_loss(teacher_outputs, student_outputs)

                self._model = DistillModel()
                return self._model

            def _learn(self, loss: torch.Tensor):
                """Learn after forward

                Parameters
                -------
                loss: torch.Tensor
                    The loss after forward
                """

                loss.backward()
                self._optimizer.step()
                return loss

            def _distill(self) -> Dict[str, Any]:
                """Distill the knowledge

                Returns
                -------
                weights: dict<str, Any>
                    The distilled weights.
                """

                state_dict = self._model.student.state_dict()
                return {
                    n: state_dict.get(self.find_tensor(n).alias, d)
                    for n, d in self._weights.items()
                }

            @classmethod
            def framework(cls):
                return MSCFramework.TORCH

        return Distiller


factory = TorchDistillerFactory()
tools = msc_utils.get_registered_tool(MSCFramework.MSC, ToolType.DISTILLER, tool_style="all")
for tool in tools.values():
    factory.create(tool)
