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
"""tvm.contrib.msc.core.runtime.jit_model"""

import logging
from typing import Any, List, Tuple, Union, Dict

from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.core.tools import ToolType
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from .runner import BaseRunner


class BaseJIT(object):
    """Base Just-In-Time compile for msc

    Parameters
    ----------
    model:
        The model to be jit compile.
    inputs: list<str>
        The input names.
    outputs: list<str>
        The output names.
    device: str
        The device to build runnable.
    training: bool
        Whether compile model to trainable.
    hooks: dict
        The hooks for runners.
    logger: logging.Logger
        The logger
    """

    def __init__(
        self,
        model: Any,
        inputs: List[str],
        outputs: List[str],
        device: str = "cpu",
        training: bool = False,
        hooks: dict = None,
        logger: logging.Logger = None,
    ):
        self._model = model
        self._jit_model = model
        self._inputs = inputs
        self._outputs = outputs
        self._device = device if self.support_device(device) else "cpu"
        self._training, self._trained = training, training
        self._hooks = hooks or {}
        self._runner_ctxs = {}
        self._logger = logger or msc_utils.get_global_logger()
        self._logger.info(msc_utils.msg_block(self.jit_mark("SETUP"), self.setup()))

    def setup(self) -> dict:
        """Setup the jit

        Returns
        -------
        info: dict
            The setup info.
        """

        return {
            "inputs": self._inputs,
            "outputs": self._outputs,
            "device": self._device,
            "training": self._training,
            "hooks": self._hooks,
        }

    def run(
        self, inputs: Union[List[Any], Dict[str, Any]], ret_type="native"
    ) -> Union[List[Any], Dict[str, Any]]:
        """Run the jit to get outputs

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

        inputs = msc_utils.format_datas(inputs, self._inputs, style="dict")
        outputs = self._call_jit(inputs)
        if ret_type == "native":
            return outputs
        return msc_utils.format_datas(outputs, self._outputs, style=ret_type)

    def _call_jit(self, inputs: Dict[str, Any]) -> Any:
        """Run the jit model

        Parameters
        ----------
        inputs:
            The inputs of model.
        """

        raise NotImplementedError("_call_jit is not implemented in " + str(self.__class__))

    def set_runner(self, runner_name: str, runner: BaseRunner):
        """Set runner in runner ctx

        Parameters
        ----------
        runner_name: str
            The runner name.
        runner: BaseRunner
            The runner.
        """

        self.get_runner_ctx(runner_name)["runner"] = runner

    def build(self):
        """Build the jit model"""

        self._jit_model = self._build(self._model)

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

        raise NotImplementedError("_build is not implemented in " + str(self.__class__))

    def make_plan(self, tool_type: str, data_loader: Any = None) -> str:
        """Execute tool and get plan

        Parameters
        -------
        tool_type: str
            The tool type, should be in ToolType
        data_loader:
            The data loader.

        Returns
        -------
        plan_file: str
            The saved plan file.
        """

        tools = {n: r["runner"].get_tool(tool_type) for n, r in self._runner_ctxs.items()}

        def _finalize_tool(
            checker: callable, post_batch: callable = None, post_iter: callable = None
        ):
            while any(not checker(t) for t in tools.values()):
                assert data_loader, "data_loader should be given to make plan for " + tool_type
                for inputs in data_loader():
                    outputs = self.run(inputs, ret_type="native")
                    if post_batch:
                        for t in tools.values():
                            post_batch(t, outputs)
                    if all(checker(t) for t in tools.values()):
                        break
                if post_iter:
                    for t in tools.values():
                        post_iter(t)
            return {n: t.finalize() for n, t in tools.items()}

        if tool_type == ToolType.PRUNER:
            plans = _finalize_tool(lambda t: t.pruned)
        elif tool_type == ToolType.QUANTIZER:
            plans = _finalize_tool(lambda t: t.calibrated, post_iter=lambda t: t.calibrate())
        elif tool_type == ToolType.DISTILLER:
            plans = _finalize_tool(
                lambda t: t.distilled,
                post_batch=lambda t, outputs: t.learn(outputs),
                post_iter=lambda t: t.distill(),
            )
        elif tool_type == ToolType.TRACKER:
            plans = _finalize_tool(lambda t: t.tracked)
        else:
            plans = {n: t.finalize() for n, t in tools.items()}
        plans_info = ", ".join(["{}({})".format(n, len(p)) for n, p in plans.items()])
        self._logger.debug("Made %s plans for %s", plans_info, tool_type)

    def _redirect_run(self, *args, runner_name: str = "worker", **kwargs) -> Any:
        """Redirect forward of model

        Parameters
        ----------
        args:
            The arguments.
        runner_name: str
            The runner name.
        kwargs:
            The kwargs.

        Returns
        -------
        outputs:
            The outputs.
        """

        assert runner_name in self._runner_ctxs, "Failed to create runner " + runner_name
        inputs = self._to_msc_inputs(runner_name, *args, **kwargs)
        for hook in self._hooks.get("pre_forward", []):
            hook(runner_name, inputs)
        outputs = self._run_ctx(self.get_runner_ctx(runner_name), inputs)
        for hook in self._hooks.get("post_forward", []):
            outputs = hook(runner_name, outputs)
        return self._from_msc_outputs(runner_name, outputs)

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

        raise NotImplementedError("_to_msc_inputs is not implemented in " + str(self.__class__))

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

        raise NotImplementedError("_from_msc_outputs is not implemented in " + str(self.__class__))

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

        raise NotImplementedError("_run_ctx is not implemented in " + str(self.__class__))

    def get_runner_ctx(self, runner_name: str) -> dict:
        """Get the runner context

        Parameters
        ----------
        runner_name: str
            The runner name

        Returns
        -------
        runner_cts: dict
            The runner context.
        """

        assert runner_name in self._runner_ctxs, "Can not finc runner_context " + str(runner_name)
        return self._runner_ctxs[runner_name]

    def train(self):
        """Change status to train"""

        if not self._training:
            self._training = True
            for runner_ctx in self._runner_ctxs.values():
                if "runner" in runner_ctx:
                    runner_ctx["runner"].train()

    def eval(self):
        """Change status to eval"""

        if self._training:
            self._training, self._trained = False, True
            for runner_ctx in self._runner_ctxs.values():
                if "runner" in runner_ctx:
                    runner_ctx["runner"].eval()

    def jit_mark(self, msg: str):
        """Mark the message with jit info

        Parameters
        -------
        msg: str
            The message

        Returns
        -------
        msg: str
            The message with mark.
        """

        return "JIT({}) {}".format(self.framework, msg)

    @property
    def trained(self):
        return self._trained

    @property
    def jit_model(self):
        return self._jit_model

    @property
    def framework(self):
        return MSCFramework.MSC

    @classmethod
    def support_device(cls, device: str) -> bool:
        """Check if the device is enabled

        Returns
        -------
        enabled: bool
            Whether the device is enabled.
        """

        return True
