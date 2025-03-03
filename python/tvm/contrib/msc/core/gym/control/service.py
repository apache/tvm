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
"""tvm.contrib.msc.core.gym.control.service"""

import json
import time
import copy
from typing import Dict, Any, List, Tuple
from multiprocessing import Manager
from functools import partial, reduce
import queue
import numpy as np

from tvm.contrib.msc.core.gym.namespace import GYMObject, GYMAction
from tvm.contrib.msc.core import utils as msc_utils
from .worker import BaseGymWorker, WorkerFactory


def _send_message(msg_queue: queue.Queue, header: str, body: dict, header_type: str = "message"):
    """Send the message to queue

    Parameters
    ----------
    msg_queue: Queue
        The message queue.
    header: str
        The header of message.
    body: dict
        The message body.
    header_type: str
        The header type
    """

    msg_queue.put(json.dumps({header_type: header, "body": body}))


def _wait_message(
    msg_queue: queue.Queue,
    header: str,
    checker: callable = None,
    wait_time: int = 2,
    max_retry: int = -1,
    header_type: str = "message",
) -> dict:
    """Wait until valid message

    Parameters
    ----------
    msg_queue: Queue
        The message queue.
    header: str
        The header of message.
    checker: callable
        The checker for the message.
    wait_time: int
        The wait time between retry in second.
    max_retry: int
        The max retry time.
    header_type: str
        The header type

    Returns
    -------
    message: dict
        The message body
    """

    def _check_message(message: dict, checker: callable = None) -> bool:
        """Check the message

        Parameters
        ----------
        message: dict
            The message.
        checker: callable
            The checker for the message.

        Returns
        -------
        pass: bool
            Whether the message pass.
        """

        if "body" not in message:
            return False
        if checker and not checker(message["body"]):
            return False
        return True

    try_cnt = 0
    while True:
        if try_cnt >= max_retry > 0:
            break
        info = msg_queue.get()
        message = json.loads(info)
        if message.get(header_type, "") == header and _check_message(message, checker):
            return message["body"]
        try_cnt += 1
        msg_queue.put(info)
        time.sleep(wait_time)
    return None


send_request = partial(_send_message, header_type="request_header")
send_response = partial(_send_message, header_type="response_header")
wait_request = partial(_wait_message, header_type="request_header")
wait_response = partial(_wait_message, header_type="response_header")


class GatherMode(object):
    """Enum all gather mode"""

    PARALLEL = "parallel"
    REDUCE_SUM = "reduce_sum"
    REDUCE_MEAN = "reduce_mean"
    FIRST = "first"


class BaseService(object):
    """Basic service for gym

    Parameters
    ----------
    workspace: MSCDirectory
        The worksapce.
    env: dict
        The environment config.
    agent: dict
        The agent config
    tasks: list<str>
        The tasks on the node.
    world_size: int
        The world size.
    max_iter: int
        The max seatch iter.
    record_step: int
        The record step.
    verbose: str
        The verbose level
    """

    def __init__(
        self,
        workspace: msc_utils.MSCDirectory,
        env: Dict[str, Any],
        agent: Dict[str, Any],
        tasks: List[str] = None,
        dist_manager: Manager = None,
        world_size: int = 1,
        max_iter: int = 1,
        record_step: int = 5,
        debug_level: int = 0,
        verbose: str = None,
    ):
        self._workspace = workspace
        tasks = tasks or [GYMObject.ENV + ":0", GYMObject.AGENT + ":0"]
        verbose = verbose or "info"
        debug_level = int(verbose.split(":")[1]) if verbose.startswith("debug:") else 0
        self._logger = msc_utils.create_file_logger(verbose, self._workspace.relpath("SERVICE_LOG"))

        def _create_workers(config: dict, obj_type: str) -> List[BaseGymWorker]:
            if "debug_level" not in config:
                config["debug_level"] = debug_level
            if "logger" not in config:
                config["logger"] = self._logger
            return [
                WorkerFactory.create(t, workspace, config) for t in tasks if t.startswith(obj_type)
            ]

        self._env_workers = _create_workers(env, GYMObject.ENV)
        self._agent_workers = _create_workers(agent, GYMObject.AGENT)
        self._dist_manager = dist_manager
        self._world_size = world_size
        self._max_iter = max_iter
        self._record_step = record_step
        self._debug_level = debug_level
        self._logger.info(msc_utils.msg_block(self.service_mark("SETUP"), self.setup()))

    def setup(self) -> dict:
        """Setup the tool

        Returns
        -------
        info: dict
            The setup info.
        """

        if self._world_size > 1:
            assert self._dist_manager, "dist manager should be given for distributed service"
            self._request_queue = self._dist_manager.get_request_queue()
            self._response_queue = self._dist_manager.get_response_queue()
            self._world_id, self._env_world_ids, self._agent_world_ids = self._connect()
        else:
            self._request_queue = queue.Queue()
            self._response_queue = queue.Queue()
            self._world_id = 0
            self._env_world_ids = [w.worker_id for w in self._env_workers]
            self._agent_world_ids = [w.worker_id for w in self._agent_workers]
        return {
            "workspace": self._workspace,
            "world_id": self._world_id,
            "world_size": self._world_size,
            "env_worker_ids": self._get_worker_ids(GYMObject.ENV),
            "env_world_ids": self._env_world_ids,
            "agent_worker_ids": self._get_worker_ids(GYMObject.AGENT),
            "agent_world_ids": self._agent_world_ids,
            "max_iter": self._max_iter,
            "record_step": self._record_step,
            "debug_level": self._debug_level,
        }

    def init(self):
        self._logger.info("SERVICE Init")
        self._iter_id, self._done = 0, False
        self._max_task = 0
        self._task_id, self._states = 0, []
        self._iter_done = False
        self.execute(GYMObject.ENV, GYMAction.INIT)
        self.execute(GYMObject.AGENT, GYMAction.INIT)

    def reset(self):
        self._task_id, self._states = 0, []
        self._iter_done = False
        self._logger.info("SERVICE Reset %d/%d th iter", self._iter_id, self._max_iter)
        self.execute(GYMObject.ENV, GYMAction.RESET)
        self.execute(GYMObject.AGENT, GYMAction.RESET)

    def learn(self):
        self.execute(GYMObject.AGENT, GYMAction.LEARN)
        if self._iter_done:
            self._iter_id += 1
        if self._iter_id >= self._max_iter:
            self._done = True

    def summary(self):
        self._logger.info("SERVICE Summary after %d iters", self._max_iter)
        self.execute(GYMObject.ENV, GYMAction.SUMMARY)
        plan = self._states[-1]["response"]["plan"]
        self.execute(GYMObject.ENV, GYMAction.CLEANUP)
        self.execute(GYMObject.AGENT, GYMAction.CLEANUP)
        return plan

    def execute(self, obj_type: str, act_type: str):
        """Execute the service

        Parameters
        ----------
        obj_type: str
            The object type, should be one of GYMObject.
        act_type: str
            The action type, should be one of GYMAction.
        """

        self._states.append(
            {
                "task_id": self._task_id,
                "msg_key": self._to_msg_key(obj_type, act_type),
                "response": self._execute(obj_type, act_type),
            }
        )

    def _execute(self, obj_type: str, act_type: str) -> dict:
        """Execute the service

        Parameters
        ----------
        obj_type: str
            The object type, should be one of GYMObject.
        act_type: str
            The action type, should be one of GYMAction.

        Returns
        -------
        state: dict
            The state after the execute.
        """

        raise NotImplementedError("_execute is not implemented in BaseService")

    def _send_request(self, msg_key: str, body: dict):
        """Send request

        Parameters
        ----------
        msg_key: str
            The header of message.
        body: dict
            The message body.
        """

        send_request(self._request_queue, msg_key, body)

    def _send_response(self, msg_key: str, body: dict):
        """Send request

        Parameters
        ----------
        msg_key: str
            The header of message.
        body: dict
            The message body.
        """

        send_response(self._response_queue, msg_key, body)

    def _wait_request(
        self,
        msg_key: str,
        checker: callable = None,
        wait_time: int = 2,
        max_retry: int = -1,
    ) -> dict:
        """Wait request

        Parameters
        ----------
        msg_key: str
            The header of message.
        checker: callable
            The checker for the message.
        wait_time: int
            The wait time between retry in second.
        max_retry: int
            The max retry time.
        """

        return wait_request(self._request_queue, msg_key, checker, wait_time, max_retry)

    def _wait_response(
        self,
        msg_key: str,
        checker: callable = None,
        wait_time: int = 2,
        max_retry: int = -1,
    ) -> dict:
        """Wait response

        Parameters
        ----------
        msg_key: str
            The header of message.
        checker: callable
            The checker for the message.
        wait_time: int
            The wait time between retry in second.
        max_retry: int
            The max retry time.
        """

        return wait_request(self._response_queue, msg_key, checker, wait_time, max_retry)

    def _process_request(self, msg_key: str) -> dict:
        """Process the request according to msg_key

        Parameters
        ----------
        msg_key: str
            The header of message.

        Returns
        -------
        responses: dict
            The responses of wrokers.
        """

        obj_type, act_type = self._from_msg_key(msg_key)
        workers = {w.worker_id: w for w in self._get_workers(obj_type)}
        requests = self._wait_request(msg_key)
        if act_type in (GYMAction.INIT, GYMAction.RESET):
            mark = "Iter[{}/{}] {}.{}".format(self._iter_id, self._max_iter, obj_type, act_type)
        else:
            mark = "Iter[{}/{}] Task[{}/{}] {}.{}".format(
                self._iter_id, self._max_iter, self._task_id, self._max_task, obj_type, act_type
            )
        requests = {int(k): v for k, v in requests.items()}
        responses = {}
        for w_id, worker in workers.items():
            responses[w_id] = worker.execute(act_type, **requests[w_id])
        info = {
            "requests": {workers[w].name: r for w, r in requests.items()},
            "responses": {workers[w].name: r for w, r in responses.items()},
        }
        self._logger.info(msc_utils.msg_block(mark, info, symbol="="))
        return responses

    def _process_response(self, msg_key: str, response: dict):
        """Update reponse

        Parameters
        ----------
        msg_key: str
            The header of message.
        response: dict
            The response.

        Returns
        -------
        response: dict
            The updated response.
        """

        obj_type, act_type = self._from_msg_key(msg_key)
        if obj_type == GYMObject.ENV and act_type == GYMAction.INIT:
            self._max_task = response["max_task"]
        if obj_type == GYMObject.AGENT and act_type == GYMAction.STORE:
            self._task_id = response["next_task"]
            if self._task_id >= self._max_task:
                self._iter_done = True
        return response

    def _to_msg_key(self, obj_type: str, act_type: str) -> str:
        """Create message key base on types

        Parameters
        ----------
        obj_type: str
            The object type, should be one of GYMObject.
        act_type: str
            The action type, should be one of GYMAction.

        Returns
        -------
        key: str
            The message key.
        """

        return "{}-s-{}".format(obj_type, act_type)

    def _from_msg_key(self, msg_key: str) -> Tuple[str, str]:
        """Get obj_type and act_type from message key

        Parameters
        ----------
        msg_key: str
            The message key.

        Returns
        -------
        obj_type: str
            The object type, should be one of GYMObject.
        act_type: str
            The action type, should be one of GYMAction.
        """

        return msg_key.split("-s-")

    def _get_workers(self, obj_type: str) -> List[BaseGymWorker]:
        """Get workers according to obj_type

        Parameters
        ----------
        obj_type: str
            The object type, should be one of GYMObject.

        Returns
        -------
        workers: list<Worker>
            The workers.
        """

        if obj_type == GYMObject.ENV:
            return self._env_workers
        if obj_type == GYMObject.AGENT:
            return self._agent_workers
        return []

    def _get_worker_ids(self, obj_type: str) -> List[int]:
        """Get worker ids according to obj_type

        Parameters
        ----------
        obj_type: str
            The object type, should be one of GYMObject.

        Returns
        -------
        worker_ids: list<int>
            The worker ids.
        """

        return [w.worker_id for w in self._get_workers(obj_type)]

    def _get_world_ids(self, obj_type: str) -> List[int]:
        """Get world ids according to obj_type

        Parameters
        obj_type: str
            The object type, should be one of GYMObject.

        Returns
        -------
        world_ids: list<Worker>
            The world ids.
        """

        if obj_type == GYMObject.ENV:
            return self._env_world_ids
        if obj_type == GYMObject.AGENT:
            return self._agent_world_ids
        return []

    def service_mark(self, msg: Any) -> str:
        """Mark the message with service info

        Parameters
        -------
        msg: str
            The message

        Returns
        -------
        msg: str
            The message with mark.
        """

        return "SERIVCE({}) {}".format(self.service_type, msg)

    @property
    def done(self):
        return self._done

    @property
    def iter_done(self):
        return self._iter_done

    @property
    def service_type(self):
        return "base"


class MainService(BaseService):
    """Main service for gym"""

    def _connect(self):
        msg_key = self._to_msg_key(GYMObject.SERVICE, GYMAction.SETUP)
        env_world_ids = self._get_worker_ids(GYMObject.ENV)
        agent_world_ids = self._get_worker_ids(GYMObject.AGENT)
        # send world_id and get env/agent ids
        barrier = self._world_size - 1

        def _check_response(body):
            return all(k in body for k in ["env_worker_ids", "agent_worker_ids"])

        for i in range(barrier):
            self._send_request(msg_key, {"world_id": i + 1})
        while barrier > 0:
            info = self._wait_response(msg_key, _check_response)
            if info:
                env_world_ids.extend(info["env_world_ids"])
                agent_world_ids.extend(info["agent_world_ids"])
                barrier -= 1

        self._synchronize_feedback(
            msg_key, env_world_ids=env_world_ids, agent_world_ids=agent_world_ids
        )
        return 0, env_world_ids, agent_world_ids

    def _execute(self, obj_type: str, act_type: str) -> dict:
        """Execute the service

        Parameters
        ----------
        obj_type: str
            The object type, should be one of GYMObject.
        act_type: str
            The action type, should be one of GYMAction.

        Returns
        -------
        state: dict
            The state after the execute.
        """

        world_ids = self._get_worker_ids(obj_type)
        tasks = {i: self._create_task(obj_type, act_type, i) for i in world_ids}
        msg_key = self._to_msg_key(obj_type, act_type)
        response = self._synchronize_request(msg_key, tasks)
        response = self._process_response(msg_key, response)
        self._synchronize_feedback(msg_key, **response)
        return response

    def _synchronize_request(
        self,
        msg_key: str,
        requests: List[dict],
        checker: callable = None,
        wait_time: int = 2,
        max_retry: int = -1,
    ) -> dict:
        """Send requests to workers and gather response

        Parameters
        ----------
        msg_key: str
            The header of message.
        requests: list<dict>
            The requests
        checker: callable
            The checker for the response.
        wait_time: int
            The wait time between retry in second.
        max_retry: int
            The max retry time.

        Returns
        -------
        response: dict
            The gathered response.
        """

        responses = {}
        barrier = self._world_size
        for _ in range(barrier):
            self._send_request(msg_key, requests)
        responses.update(self._process_request(msg_key))
        barrier -= 1
        while barrier > 0:
            info = self._wait_response(msg_key, checker, wait_time, max_retry)
            if info:
                info = {int(k): v for k, v in info.items()}
                responses.update(info)
                barrier -= 1
        responses = [responses[i] for i in sorted(responses)]
        gathered_response = {}
        for key in responses[0]:
            if key in ("action", "reward"):
                gather_mode = GatherMode.PARALLEL
            else:
                gather_mode = GatherMode.FIRST
            gathered_response[key] = self._gather_values([r[key] for r in responses], gather_mode)
        return gathered_response

    def _synchronize_feedback(self, msg_key: str, **feedback: dict):
        """Broadcast feedback to workers

        Parameters
        ----------
        msg_key: str
            The header of message.
        feedback: dict
            The feedback body
        """

        def _check_feedback(body):
            return body.get("feedback_receive", False)

        barrier = self._world_size - 1
        for _ in range(barrier):
            self._send_request(msg_key, {"feedback_send": True, **feedback})
        while barrier > 0:
            info = self._wait_response(msg_key, _check_feedback)
            if info:
                barrier -= 1

    def _create_task(self, obj_type: str, act_type: str, worker_id: int) -> dict:
        """Create message key base on types

        Parameters
        ----------
        obj_type: str
            The object type, should be one of GYMObject.
        act_type: str
            The action type, should be one of GYMAction.
        worker_id: int
            The worker id.

        Returns
        -------
        config: dict
            The config for the worker.execute.
        """

        if not self._states:
            config = {}
        else:
            config = copy.deepcopy(self._states[-1]["response"])
        if obj_type == GYMObject.ENV and act_type == GYMAction.GET_STATE:
            config["task_id"] = self._task_id
        if obj_type == GYMObject.ENV and act_type == GYMAction.STEP:
            config["actions"] = self._map_values(config["actions"], obj_type, worker_id)
            config["task_id"] = self._task_id
        elif obj_type == GYMObject.AGENT and act_type in (GYMAction.CHOOSE_ACTION, GYMAction.STORE):
            config["task_id"] = self._task_id
        return config

    def _map_values(self, values: List[Any], obj_type: str, worker_id: int) -> List[Any]:
        """Map the values for worker

        Parameters
        ----------
        values: list
            The global values,
        obj_type: str
            The object type, should be one of GYMObject.
        worker_id: int
            The worker id.

        Returns
        -------
        values: list
            The values for the worker.
        """

        world_ids = self._get_world_ids(obj_type)
        tile_size = len(values) // len(world_ids)
        if len(values) % len(world_ids) != 0:
            tile_size += 1
        worker_idx = world_ids.index(worker_id)
        start = worker_idx * tile_size
        end = min((worker_idx + 1) * tile_size, len(values))
        return values[start:end]

    def _gather_values(self, values: List[Any], gather_mode: str) -> Any:
        """Gather the values

        Parameters
        ----------
        values: list
            The global values,
        gather_mode: str
            The gather mode should be in GatherMode.

        Returns
        -------
        value:
            The gathered value.
        """

        if gather_mode == GatherMode.FIRST or len(values) == 1:
            return values[0]
        if gather_mode == GatherMode.PARALLEL:
            return values
        if gather_mode in (GatherMode.REDUCE_MEAN, GatherMode.REDUCE_SUM):
            if all(msc_utils.MSCArray.is_array(v) for v in values):
                value_sum = np.array([msc_utils.cast_array(v) for v in values]).sum(axis=1)
            else:
                value_sum = reduce(lambda x, y: x + y, values)
            if gather_mode == GatherMode.REDUCE_SUM:
                return value_sum
            return value_sum / len(values)
        raise NotImplementedError("Gather mode {} is not supported")

    @property
    def service_type(self):
        return "main"


class NodeService(BaseService):
    """Normal service for gym"""

    def _connect(self):
        msg_key = self._to_msg_key(GYMObject.SERVICE, GYMAction.SETUP)
        env_worker_ids = self._get_worker_ids(GYMObject.ENV)
        agent_worker_ids = self._get_worker_ids(GYMObject.AGENT)

        def _check_request(body):
            return "world_id" in body

        info = self._wait_request(msg_key, _check_request)
        world_id = info["world_id"]
        self._send_response(
            msg_key, {"env_worker_ids": env_worker_ids, "agent_worker_ids": agent_worker_ids}
        )
        info = self._feedback(msg_key)
        return world_id, info["env_world_ids"], info["agent_world_ids"]

    def _feedback(self, msg_key: str) -> dict:
        """Send feed back to main service

        Parameters
        ----------
        msg_key: str
            The header of message.

        Returns
        -------
        response: dict
            The recived feedback.
        """

        def _check_feedback(body):
            return body.get("feedback_send", False)

        response = self._wait_request(msg_key, _check_feedback)
        self._send_response(msg_key, {"feedback_receive": True})
        response = self._process_response(msg_key, response)
        return response

    def _execute(self, obj_type: str, act_type: str) -> dict:
        """Execute the service

        Parameters
        ----------
        obj_type: str
            The object type, should be one of GYMObject.
        act_type: str
            The action type, should be one of GYMAction.

        Returns
        -------
        state: dict
            The state after the execute.
        """

        msg_key = self._to_msg_key(obj_type, act_type)
        info = self._process_request(msg_key)
        self._send_response(msg_key, info)
        return self._feedback(msg_key)

    @property
    def service_type(self):
        return "node"
