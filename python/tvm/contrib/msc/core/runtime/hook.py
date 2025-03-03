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
# pylint: disable=unused-argument, arguments-differ
"""tvm.contrib.msc.core.runtime.hook"""

from typing import Dict, List, Tuple, Union, Any

import tvm
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core import utils as msc_utils


class RunnerHook(object):
    """Hook for runner

    Parameters
    ----------
    config: dict
        The config of the func.
    """

    def __init__(self, config: dict):
        self._config = config

    def __str__(self):
        return "{}({})".format(self.name(), self._config)

    def apply(self, runner: object, *args, **kwargs) -> Any:
        """Apply the hook

        Parameters
        ----------
        runner:
            The runner context.
        args: list<Any>
            The arguments for run method.
        kwargs: dict<Any>
            The key word arguments for run method.

        Returns
        -------
        result:
           The results.
        """

        kwargs.update({k: v for k, v in self._config.items() if k not in kwargs})
        return self._apply(runner, *args, **kwargs)

    def _apply(self, runner: object, *args, **kwargs):
        """Apply the hook

        Parameters
        ----------
        runner:
            The runner context.
        args: list<Any>
            The arguments for run method.
        kwargs: dict<Any>
            The key word arguments for run method.

        Returns
        -------
        result:
           The results.
        """

        raise NotImplementedError("default_func is not supported in " + str(self.__class__))

    @classmethod
    def name(cls):
        return "base"


class CustomizedHook(RunnerHook):
    """Hook for customized func

    Parameters
    ----------
    func: callable/str
        The function.
    config: dict
        The config of the func.
    """

    def __init__(self, func: Union[str, callable], config: dict):
        super(CustomizedHook, self).__init__(config)
        self._func = msc_utils.load_callable(func)

    def __str__(self):
        return "{} {}({})".format(self.name(), self._func, self._config)

    def _apply(self, runner: object, *args, **kwargs):
        """Apply the hook

        Parameters
        ----------
        runner:
            The runner context.
        args: list<Any>
            The arguments for run method.
        kwargs: dict<Any>
            The key word arguments for run method.

        Returns
        -------
        result:
           The results.
        """

        return self._func(runner, *args, **kwargs)

    @classmethod
    def name(cls):
        return "customized"


@msc_utils.register_runner_hook
class UpdateWeightsHook(RunnerHook):
    """Hook for update weights"""

    def _apply(
        self,
        runner: object,
        graphs: List[MSCGraph],
        weights: Dict[str, tvm.nd.array],
        weights_path: str,
    ) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Apply the default funcion

        Parameters
        -------
        runner:
            The runner context.
        graphs: list<MSCGraph>
            The translated graphs
        weights: dict<str, tvm.nd.array>
            The translated weights.
        weights_path: str
            The weights path.

        Returns
        -------
        graphs: list<MSCGraph>
            The updated graphs
        weights: dict<str, tvm.nd.array>
            The updated weights.

        """

        with open(weights_path, "rb") as f:
            new_weights = tvm.runtime.load_param_dict(f.read())
        weights.update({k: v for k, v in new_weights.items() if k in weights})
        return graphs, weights

    @classmethod
    def name(cls):
        return "update_weights"


def load_runner_hook(config: dict) -> Any:
    """Load a registered hook

    Parameters
    ----------
    config: dict
        The config of the func.

    Returns
    -------
    hook: RunnerHook
        The hook
    """

    assert "hook" in config, "hook should be given to load hook"
    hook_ref = config["hook"]
    hook_config = {k: v for k, v in config.items() if k != "hook"}
    hook_cls = msc_utils.get_registered_runner_hook(hook_ref) if isinstance(hook_ref, str) else None
    if hook_cls:
        return hook_cls(hook_config)
    return CustomizedHook(hook_ref, hook_config)
