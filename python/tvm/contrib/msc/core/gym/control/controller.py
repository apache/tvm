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
"""tvm.contrib.msc.core.gym.control.controller"""

from typing import Dict, Any
from tvm.contrib.msc.core.gym.namespace import GYMObject, GYMAction
from tvm.contrib.msc.core import utils as msc_utils
from .service import MainService, NodeService


class BaseController(object):
    """Basic controller for optimize search

    Parameters
    ----------
    workspace: MSCDirectory
        The worksapce.
    config: dict
        The config for service.
    is_main: bool
        Whether the node is main node
    """

    def __init__(
        self,
        workspace: msc_utils.MSCDirectory,
        config: Dict[str, Any],
        is_main: bool = True,
    ):
        self._workspace = workspace
        service_cls = MainService if is_main else NodeService
        self._service = service_cls(self._workspace, **config)

    def run(self) -> dict:
        """Run the controller

        Returns
        -------
        report: dict
            The run report.
        """

        self._service.init()
        while not self._service.done:
            self._service.reset()
            while not self._service.iter_done:
                self._service.execute(GYMObject.ENV, GYMAction.GET_STATE)
                self._service.execute(GYMObject.AGENT, GYMAction.CHOOSE_ACTION)
                self._service.execute(GYMObject.ENV, GYMAction.STEP)
                self._service.execute(GYMObject.AGENT, GYMAction.STORE)
            self._service.learn()
        return self._service.summary()


def create_controller(stage: str, config: dict, extra_config: dict = None):
    """Update the gym config

    Parameters
    ----------
    stage: str
        The stage for gym, should be in MSCStage.
    config: dict
        The raw config.
    extra_config: dict
        The extra config

    Returns
    -------
    config: dict
        The update config.
    """

    config_type = config.pop("config_type") if "config_type" in config else "default"
    configer_cls = msc_utils.get_registered_gym_configer(config_type)
    assert configer_cls, "Can not find configer for " + str(config_type)
    config = configer_cls(stage).update(config)
    if extra_config:
        config = msc_utils.update_dict(config, extra_config)
    if "control_type" in config:
        control_type = config.pop("control_type")
    else:
        control_type = "default"
    controller_cls = msc_utils.get_registered_gym_controller(control_type)
    return controller_cls(msc_utils.get_gym_dir(), config)


@msc_utils.register_gym_controller
class DefaultController(BaseController):
    @classmethod
    def control_type(cls):
        return "default"
