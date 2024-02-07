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
"""tvm.contrib.msc.core.utils.register"""

from typing import Any, Optional
from .namespace import MSCFramework


class MSCRegistery:
    """The registery for MSC"""

    REGISTERY = {}
    MSC_FUNCS = "msc_funcs"
    MSC_TOOLS_CLS = "msc_tools_cls"
    MSC_TOOLS_METHOD = "msc_tools_method"
    GYM_CONFIGERS = "gym_configers"
    GYM_CONTROLLERS = "gym_controllers"
    GYM_AGENTS = "gym_agents"
    GYM_ENVS = "gym_envs"
    GYM_METHODS = "gym_agents_method"
    RUNNER_HOOKS = "runner_hooks"

    @classmethod
    def register(cls, key: str, value: Any):
        cls.REGISTERY[key] = value
        return value

    @classmethod
    def unregister(cls, key: str):
        if key in cls.REGISTERY:
            return cls.REGISTERY.pop(key)
        return None

    @classmethod
    def get(cls, key: str, default: Optional[Any] = None) -> Any:
        return cls.REGISTERY.get(key, default)

    @classmethod
    def contains(cls, key: str):
        return key in cls.REGISTERY

    @classmethod
    def reset(cls):
        cls.REGISTERY = {}


def register_func(name: str, func: callable, framework: str = MSCFramework.MSC):
    """Register a func for framework.

    Parameters
    ----------
    name: string
        The name for the func.
    func: callable
        The function to be registered.
    framework: string
        Should be from MSCFramework.
    """

    funcs = MSCRegistery.get(MSCRegistery.MSC_FUNCS, {})
    if framework not in funcs:
        funcs[framework] = {}
    funcs[framework][name] = func
    MSCRegistery.register(MSCRegistery.MSC_FUNCS, funcs)


def get_registered_func(name: str, framework: str = MSCFramework.MSC):
    """Get the registered func of framework.

    Parameters
    ----------
    name: string
        The name for the func.
    framework: string
        Should be from MSCFramework.

    Returns
    -------
    func: callable
        The registered function.
    """

    funcs = MSCRegistery.get(MSCRegistery.MSC_FUNCS, {})
    if framework not in funcs:
        return None
    return funcs[framework].get(name)


def register_tool_cls(tool_cls: Any):
    """Register a tool class.

    Parameters
    ----------
    tool_cls: class
        The tool class to be registered.
    """

    tools_cls = MSCRegistery.get(MSCRegistery.MSC_TOOLS_CLS, {})
    for key in ["framework", "tool_type", "tool_style"]:
        assert hasattr(tool_cls, key), "{} should be given to register tool class".format(key)
    if tool_cls.framework() not in tools_cls:
        tools_cls[tool_cls.framework()] = {}
    framework_tools = tools_cls[tool_cls.framework()]
    if tool_cls.tool_type() not in framework_tools:
        framework_tools[tool_cls.tool_type()] = {}
    tools = framework_tools[tool_cls.tool_type()]
    tools[tool_cls.tool_style()] = tool_cls
    MSCRegistery.register(MSCRegistery.MSC_TOOLS_CLS, tools_cls)


def get_registered_tool_cls(framework: str, tool_type: str, tool_style: str) -> Any:
    """Get the registered tool class.

    Parameters
    ----------
    framework: string
        Should be from MSCFramework.
    tool_type: string
        The type of the tool prune| quantize| distill| debug.
    tool_style: string
        The style of the tool.

    Returns
    -------
    tool_cls: class
        The registered tool class.
    """

    tools_cls = MSCRegistery.get(MSCRegistery.MSC_TOOLS_CLS, {})
    if tool_style == "all":
        return tools_cls.get(framework, {}).get(tool_type, {})
    return tools_cls.get(framework, {}).get(tool_type, {}).get(tool_style)


def register_tool_method(method_cls: Any, method_style: str = "default"):
    """Register a tool method.

    Parameters
    ----------
    method_cls: class
        The method class.
    method_style: string
        The style of the method.
    """

    tools_method = MSCRegistery.get(MSCRegistery.MSC_TOOLS_METHOD, {})
    for key in ["framework", "tool_type"]:
        assert hasattr(method_cls, key), "{} should be given to register tool method".format(key)
    if method_cls.framework() not in tools_method:
        tools_method[method_cls.framework()] = {}
    register_name = "{}.{}".format(method_cls.tool_type(), method_style)
    tools_method[method_cls.framework()][register_name] = method_cls
    MSCRegistery.register(MSCRegistery.MSC_TOOLS_METHOD, tools_method)


def get_registered_tool_method(
    framework: str, tool_type: str, method_style: str = "default"
) -> Any:
    """Get the registered tool method.

    Parameters
    ----------
    framework: string
        Should be from MSCFramework.
    tool_type: string
        The type of the tool prune| quantize| distill| debug.
    method_style: string
        The style of the method.

    Returns
    -------
    method_cls: class
        The method class.
    """

    tools_method = MSCRegistery.get(MSCRegistery.MSC_TOOLS_METHOD, {})
    register_name = "{}.{}".format(tool_type, method_style)
    return tools_method.get(framework, {}).get(register_name)


def register_gym_configer(configer: Any):
    """Register a gym configer.

    Parameters
    ----------
    configer: class
        The configer class.
    """

    configers = MSCRegistery.get(MSCRegistery.GYM_CONFIGERS, {})
    assert hasattr(configer, "config_type"), "config_type should be given to register configer"
    configers[configer.config_type()] = configer
    MSCRegistery.register(MSCRegistery.GYM_CONFIGERS, configers)


def get_registered_gym_configer(config_type: str) -> Any:
    """Get the registered configer.

    Parameters
    ----------
    config_type: string
        The type of configer.

    Returns
    -------
    configer: class
        The configer class.
    """

    configers = MSCRegistery.get(MSCRegistery.GYM_CONFIGERS, {})
    return configers.get(config_type)


def register_gym_controller(controller: Any):
    """Register a gym controller.

    Parameters
    ----------
    controller: class
        The controller class.
    """

    controllers = MSCRegistery.get(MSCRegistery.GYM_CONTROLLERS, {})
    assert hasattr(
        controller, "control_type"
    ), "control_type should be given to register controller"
    controllers[controller.control_type()] = controller
    MSCRegistery.register(MSCRegistery.GYM_CONTROLLERS, controllers)


def get_registered_gym_controller(control_type: str) -> Any:
    """Get the registered controller.

    Parameters
    ----------
    control_type: string
        The type of controller.

    Returns
    -------
    controller: class
        The controller class.
    """

    controllers = MSCRegistery.get(MSCRegistery.GYM_CONTROLLERS, {})
    return controllers.get(control_type)


def register_gym_agent(agent: Any):
    """Register a gym agent.

    Parameters
    ----------
    agent: class
        The agent class.
    """

    agents = MSCRegistery.get(MSCRegistery.GYM_AGENTS, {})
    assert hasattr(agent, "agent_type"), "agent_type should be given to register agent"
    agents[agent.agent_type()] = agent
    MSCRegistery.register(MSCRegistery.GYM_AGENTS, agents)


def get_registered_gym_agent(agent_type: str) -> Any:
    """Get the registered agent.

    Parameters
    ----------
    agent_type: string
        The type of agent.

    Returns
    -------
    agent: class
        The agent class.
    """

    agents = MSCRegistery.get(MSCRegistery.GYM_AGENTS, {})
    return agents.get(agent_type)


def register_gym_env(env: Any):
    """Register a gym env.

    Parameters
    ----------
    env: class
        The env class.
    """

    envs = MSCRegistery.get(MSCRegistery.GYM_ENVS, {})
    assert hasattr(env, "env_type"), "env_type should be given to register env"
    envs[env.env_type()] = env
    MSCRegistery.register(MSCRegistery.GYM_ENVS, envs)


def get_registered_gym_env(env_type: str) -> Any:
    """Get the registered env.

    Parameters
    ----------
    env_type: string
        The type of agent.

    Returns
    -------
    env: class
        The agent class.
    """

    envs = MSCRegistery.get(MSCRegistery.GYM_ENVS, {})
    return envs.get(env_type)


def register_gym_method(method: Any):
    """Register a gym method.

    Parameters
    ----------
    method: class
        The method class.
    """

    methods = MSCRegistery.get(MSCRegistery.GYM_METHODS, {})
    assert hasattr(method, "method_type"), "method_type should be given to register method"
    methods[method.method_type()] = method
    MSCRegistery.register(MSCRegistery.GYM_METHODS, methods)


def get_registered_gym_method(method_type: str) -> Any:
    """Get the registered gym method.

    Parameters
    ----------
    method_type: str
        The type of method.

    Returns
    -------
    method: class
        The method class.
    """

    methods = MSCRegistery.get(MSCRegistery.GYM_METHODS, {})
    return methods.get(method_type)


def register_runner_hook(hook: Any):
    """Register a runner hook.

    Parameters
    ----------
    hook: class
        The hook class.
    """

    hooks = MSCRegistery.get(MSCRegistery.RUNNER_HOOKS, {})
    assert hasattr(hook, "name"), "name should be given to register hook"
    hooks[hook.name()] = hook
    MSCRegistery.register(MSCRegistery.RUNNER_HOOKS, hooks)


def get_registered_runner_hook(name: str) -> Any:
    """Get the registered runner hook.

    Parameters
    ----------
    name: str
        The name hook.

    Returns
    -------
    method: class
        The method class.
    """

    hooks = MSCRegistery.get(MSCRegistery.RUNNER_HOOKS, {})
    return hooks.get(name)
