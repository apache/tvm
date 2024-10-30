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
"""tvm.contrib.msc.framework.torch.runtime.jit_model"""

from typing import Any, List, Tuple, Dict
from functools import partial

import torch
from torch import fx
from torch import _dynamo as dynamo

from tvm.contrib.msc.core.runtime import BaseJIT
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils
from .runner import TorchRunner


class TorchJIT(BaseJIT):
    """JIT of Torch"""

    def _call_jit(self, inputs: Dict[str, Any]) -> Any:
        """Run the jit model

        Parameters
        ----------
        inputs:
            The inputs of model.
        """

        torch_inputs = [
            msc_utils.cast_array(inputs[i], MSCFramework.TORCH, self._device) for i in self._inputs
        ]
        return self._jit_model(*torch_inputs)

    def _build(self, model: Any) -> Any:
        """Build the jit model

        Parameters
        ----------
        model:
            The model.

        Returns
        -------
        jit_model:
            The jit model.
        """

        # pylint: disable=unused-argument
        def _compile(graph_module: fx.GraphModule, example_inputs):
            graph_module = graph_module.train() if self._training else graph_module.eval()
            name = "jit_" + str(len(self._runner_ctxs))
            self._runner_ctxs[name] = {"model": graph_module}
            return partial(self._redirect_run, runner_name=name)

        dynamo.reset()
        return torch.compile(self._model, backend=_compile)

    def _to_msc_inputs(self, runner_name: str, *args, **kwargs) -> List[Tuple[str, Any]]:
        """Change inputs to msc format

        Parameters
        ----------
        runner_name: str
            The runner name.
        args:
            The arguments.
        kwargs:
            The kwargs.

        Returns
        -------
        inputs:
            The msc format inputs.
        """

        assert not kwargs, "TorchJIT do not support kwargs"
        return [("input_" + str(i), d) for i, d in enumerate(args)]

    def _from_msc_outputs(self, runner_name: str, outputs: List[Tuple[str, Any]]) -> Any:
        """Change inputs from msc format

        Parameters
        ----------
        runner_name: str
            The runner name.
        outputs: list<(str, tensor)>
            The msc format outputs.

        Returns
        -------
        outputs:
            The framework outputs.
        """

        torch_outputs = [o[1] for o in outputs]
        unpack_outputs = self.get_runner_ctx(runner_name).get("unpack_outputs", True)
        if not unpack_outputs:
            return torch_outputs
        return torch_outputs[0] if len(torch_outputs) == 1 else torch_outputs

    def _run_ctx(self, runner_ctx: dict, inputs: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        """Forward by runner context

        Parameters
        ----------
        runner_ctx: dict
            The runner context
        inputs: list<(str, tensor)>
            The inputs.

        Returns
        -------
        outputs: list<(str, tensor)>
            The outputs.
        """

        if "runner" in runner_ctx:
            runner = runner_ctx["runner"]
            if runner.framework == MSCFramework.TORCH:
                outputs = runner.run({i[0]: i[1] for i in inputs}, ret_type="native")
            else:
                outputs = runner.run({i[0]: i[1] for i in inputs}, ret_type="list")
                outputs = [
                    msc_utils.cast_array(o, MSCFramework.TORCH, runner.device) for o in outputs
                ]
        else:
            torch_inputs = [i[1] for i in inputs]
            outputs = runner_ctx["model"](*torch_inputs)
            if isinstance(outputs, (list, tuple)) and len(outputs) == 1:
                runner_ctx["unpack_outputs"] = False
        if isinstance(outputs, (list, tuple)):
            return [("output_" + str(i), o) for i, o in enumerate(outputs)]
        return [("output", outputs)]

    @property
    def framework(self):
        return MSCFramework.TORCH

    @classmethod
    def load_native(cls, model: Any, config: dict) -> Tuple[torch.nn.Module, str, bool]:
        """Load the native model

        Parameters
        -------
        model:
            The native model.
        config: dict
            The config for pipeline.

        Returns
        -------
        model: torch.nn.Module
            The loaded native model.
        device: str
            The device of the model.
        training:
            Whether the model is for training.
        """

        return TorchRunner.load_native(model, config)

    @classmethod
    def dump_nativate(
        cls, model: torch.nn.Module, folder: msc_utils.MSCDirectory, dump_config: dict = None
    ) -> str:
        """Dump the nativate model

        Parameters
        -------
        model: torch.nn.Module
            The runnable model.
        folder: MSCDirectory
            The export folder.
        dump_config: dict
            The dump config.

        Returns
        -------
        export_path: str
            The exported path
        """

        dump_config = dump_config or {}
        assert dump_config.get("mode", "fx") == "fx", "TorchJIT only support dump nativate as fx"
        return TorchRunner.dump_nativate(model, folder, dump_config)

    @classmethod
    def support_device(cls, device: str) -> bool:
        """Check if the device is enabled

        Returns
        -------
        enabled: bool
            Whether the device is enabled.
        """

        return TorchRunner.support_device(device)
