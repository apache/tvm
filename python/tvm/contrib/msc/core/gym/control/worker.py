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
"""tvm.contrib.msc.core.gym.control.worker"""

from typing import Any
from tvm.contrib.msc.core.gym.namespace import GYMObject, GYMAction
from tvm.contrib.msc.core import utils as msc_utils


class BaseGymWorker(object):
    """Basic worker for gym

    Parameters
    ----------
    name: str
        The worker name.
    workspace: MSCDirectory
        The worksapce.
    worker_id: int
        The worker_id.
    worker_cls: class
        The worker class.
    worker_config: dict
        The worker config.
    """

    def __init__(
        self,
        name: str,
        workspace: msc_utils.MSCDirectory,
        worker_id: int,
        worker_cls: Any,
        worker_config: dict,
    ):
        self._name = name
        self._worker_id = worker_id
        debug_level = worker_config.get("debug_level", 0)
        if "logger" not in worker_config:
            verbose = "debug" if debug_level > 0 else "info"
            worker_config["logger"] = msc_utils.create_file_logger(
                verbose, workspace.relpath("{}.{}_LOG".format(self.obj_type.upper(), worker_id))
            )
        if "workspace" not in worker_config:
            worker_config["workspace"] = workspace
        worker_config["name"] = name
        self._worker_impl = worker_cls(**worker_config)

    def __str__(self):
        return "<{}>: {}({})".format(self.obj_type, self._name, self._worker_id)

    def execute(self, act_type: str, **kwargs) -> Any:
        """Execute the worker

        Parameters
        ----------
        act_type: str
            The action type, should be one of GYMAction.
        kwargs: dict
            The kwargs for execute.

        Returns
        -------
        response: dict
            The execute result.
        """

        raise NotImplementedError("execute is not implemented in " + str(self.__class__))

    @property
    def obj_type(self):
        return GYMObject.BASE

    @property
    def name(self):
        return self._name

    @property
    def worker_id(self):
        return self._worker_id


class EnvGymWorker(BaseGymWorker):
    """Env worker for gym"""

    def execute(self, act_type: str, **kwargs) -> Any:
        """Execute the worker

        Parameters
        ----------
        act_type: str
            The action type, should be one of GYMAction.
        kwargs: dict
            The kwargs for execute.

        Returns
        -------
        response: dict
            The execute result.
        """

        response = {}
        if act_type == GYMAction.INIT:
            max_task, baseline = self._worker_impl.init()
            response.update({"max_task": max_task, "baseline": baseline})
        elif act_type == GYMAction.RESET:
            self._worker_impl.reset()
        elif act_type == GYMAction.GET_STATE:
            observation, action_space = self._worker_impl.get_state(kwargs["task_id"])
            response.update({"observation": observation, "action_space": action_space})
        elif act_type == GYMAction.STEP:
            rewards = self._worker_impl.step(**kwargs)
            response.update({"rewards": rewards})
        elif act_type == GYMAction.SUMMARY:
            plan = self._worker_impl.summary(**kwargs)
            response.update({"plan": plan})
        elif act_type == GYMAction.CLEANUP:
            self._worker_impl.destory()
        return response

    @property
    def obj_type(self):
        return GYMObject.ENV


class AgentGymWorker(BaseGymWorker):
    """Agent worker for gym"""

    def execute(self, act_type: str, **kwargs) -> Any:
        """Execute the worker

        Parameters
        ----------
        act_type: str
            The action type, should be one of GYMAction.
        kwargs: dict
            The kwargs for execute.

        Returns
        -------
        response: dict
            The execute result.
        """

        response = {}
        if act_type == GYMAction.INIT:
            self._worker_impl.init(**kwargs)
        elif act_type == GYMAction.RESET:
            self._worker_impl.reset()
        elif act_type == GYMAction.CHOOSE_ACTION:
            actions = self._worker_impl.choose_action(**kwargs)
            response.update({"actions": actions})
        elif act_type == GYMAction.STORE:
            next_task = self._worker_impl.store(**kwargs)
            response.update({"next_task": next_task})
        elif act_type == GYMAction.LEARN:
            actions, rewards = self._worker_impl.learn()
            response.update({"actions": actions, "rewards": rewards})
        elif act_type == GYMAction.CLEANUP:
            self._worker_impl.destory()
        return response

    @property
    def obj_type(self):
        return GYMObject.AGENT


class WorkerFactory(object):
    """The Factory for workers"""

    @classmethod
    def create(cls, name: str, workspace: msc_utils.MSCDirectory, config: dict) -> BaseGymWorker:
        """Create worker

        Parameters
        ----------
        name: str
            The name of worker, should be in type.
        workspace: MSCDirectory
            The worksapce.
        worker_id: int
            The worker_id.
        worker_cls: class
            The worker class.
        worker_config: dict
            The worker config.

        Returns
        -------
        worker: BaseGymWorker
            The create worker.
        """

        def _get_worker_cls(obj: str):
            worker_type = config.pop("role_type") if "role_type" in config else "default"
            worker_cls = msc_utils.get_registered_gym_object(obj, worker_type)
            assert worker_cls, "Can not find worker class for {}:{}".format(obj, worker_type)
            return worker_cls

        obj_type, worker_id = name.split(":")
        if obj_type == GYMObject.ENV:
            worker_cls = _get_worker_cls(obj_type)
            return EnvGymWorker(name, workspace, int(worker_id), worker_cls, config)
        if obj_type == GYMObject.AGENT:
            worker_cls = _get_worker_cls(obj_type)
            return AgentGymWorker(name, workspace, int(worker_id), worker_cls, config)
        raise TypeError("Worker for {} is not supported".format(obj_type))
