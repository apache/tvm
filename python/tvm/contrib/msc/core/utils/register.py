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
    TOOL_CLASSES = "tool_classes"
    TOOL_METHODS = "tool_methods"
    TOOL_CONFIGERS = "tool_configers"
    GYM_CONFIGERS = "gym_configers"
    GYM_CONTROLLERS = "gym_controllers"
    GYM_OBJECTS = "gym_objects"
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


def register_tool(tool: Any):
    """Register a tool class.

    Parameters
    ----------
    tool: class
        The tool class to be registered.
    """

    for key in ["framework", "tool_type", "tool_style"]:
        assert hasattr(tool, key), "{} should be given to register tool".format(key)
    tools_classes = MSCRegistery.get(MSCRegistery.TOOL_CLASSES, {})
    col = tools_classes.setdefault(tool.framework(), {}).setdefault(tool.tool_type(), {})
    col[tool.tool_style()] = tool
    MSCRegistery.register(MSCRegistery.TOOL_CLASSES, tools_classes)
    return tool


def get_registered_tool(framework: str, tool_type: str, tool_style: str) -> Any:
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
    tool: class
        The registered tool class.
    """

    tools_classes = MSCRegistery.get(MSCRegistery.TOOL_CLASSES, {})
    if tool_style == "all":
        return tools_classes.get(framework, {}).get(tool_type, {})
    return tools_classes.get(framework, {}).get(tool_type, {}).get(tool_style)


def register_tool_method(method: Any):
    """Register a tool method.

    Parameters
    ----------
    method: class
        The method class.
    """

    for key in ["framework", "tool_type", "method_style"]:
        assert hasattr(method, key), "{} should be given to register tool method".format(key)
    tool_methods = MSCRegistery.get(MSCRegistery.TOOL_METHODS, {})
    col = tool_methods.setdefault(method.framework(), {}).setdefault(method.tool_type(), {})
    col[method.method_style()] = method
    MSCRegistery.register(MSCRegistery.TOOL_METHODS, tool_methods)
    return method


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

    tool_methods = MSCRegistery.get(MSCRegistery.TOOL_METHODS, {})
    return tool_methods.get(framework, {}).get(tool_type, {}).get(method_style)


def register_tool_configer(configer: Any):
    """Register a tool configer.

    Parameters
    ----------
    configer: class
        The configer class.
    """

    for key in ["tool_type", "config_style"]:
        assert hasattr(configer, key), "{} should be given to register tool configer".format(key)
    tool_configers = MSCRegistery.get(MSCRegistery.TOOL_CONFIGERS, {})
    col = tool_configers.setdefault(configer.tool_type(), {})
    col[configer.config_style()] = configer
    MSCRegistery.register(MSCRegistery.TOOL_CONFIGERS, tool_configers)
    return configer


def get_registered_tool_configer(tool_type: str, config_style: str) -> Any:
    """Get the registered configer.

    Parameters
    ----------
    tool_type: string
        The type of tool.
    config_style: string
        The style of tool.

    Returns
    -------
    configer: class
        The configer class.
    """

    tool_configers = MSCRegistery.get(MSCRegistery.TOOL_CONFIGERS, {})
    return tool_configers.get(tool_type, {}).get(config_style)


def register_gym_configer(configer: Any):
    """Register a gym configer.

    Parameters
    ----------
    configer: class
        The configer class.
    """

    assert hasattr(configer, "config_type"), "config_type should be given to register configer"
    gym_configers = MSCRegistery.get(MSCRegistery.GYM_CONFIGERS, {})
    gym_configers[configer.config_type()] = configer
    MSCRegistery.register(MSCRegistery.GYM_CONFIGERS, gym_configers)
    return configer


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

    gym_configers = MSCRegistery.get(MSCRegistery.GYM_CONFIGERS, {})
    return gym_configers.get(config_type)


def register_gym_controller(controller: Any):
    """Register a gym controller.

    Parameters
    ----------
    controller: class
        The controller class.
    """

    assert hasattr(
        controller, "control_type"
    ), "control_type should be given to register controller"
    gym_controllers = MSCRegistery.get(MSCRegistery.GYM_CONTROLLERS, {})
    gym_controllers[controller.control_type()] = controller
    MSCRegistery.register(MSCRegistery.GYM_CONTROLLERS, gym_controllers)
    return controller


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

    gym_controllers = MSCRegistery.get(MSCRegistery.GYM_CONTROLLERS, {})
    return gym_controllers.get(control_type)


def register_gym_object(obj: Any):
    """Register a gym object.

    Parameters
    ----------
    obj: class
        The object class.
    """

    for key in ["role", "role_type"]:
        assert hasattr(obj, key), "{} should be given to register gym object".format(key)
    gym_objects = MSCRegistery.get(MSCRegistery.GYM_OBJECTS, {})
    col = gym_objects.setdefault(obj.role(), {})
    col[obj.role_type()] = obj
    MSCRegistery.register(MSCRegistery.GYM_OBJECTS, gym_objects)
    return obj


def get_registered_gym_object(role: str, role_type: str) -> Any:
    """Get the registered object.

    Parameters
    ----------
    role: string
        The role.
    role_type: string
        The type of the role.

    Returns
    -------
    object: class
        The object class.
    """

    gym_objects = MSCRegistery.get(MSCRegistery.GYM_OBJECTS, {})
    return gym_objects.get(role, {}).get(role_type)


def register_gym_method(method: Any):
    """Register a gym method.

    Parameters
    ----------
    method: class
        The method class.
    """

    for key in ["role", "method_type"]:
        assert hasattr(method, key), "{} should be given to register gym method".format(key)
    gym_methods = MSCRegistery.get(MSCRegistery.GYM_METHODS, {})
    col = gym_methods.setdefault(method.role(), {})
    col[method.method_type()] = method
    MSCRegistery.register(MSCRegistery.GYM_METHODS, gym_methods)
    return method


def get_registered_gym_method(role: str, method_type: str) -> Any:
    """Get the registered gym method.

    Parameters
    ----------
    role: str
        The role.
    method_type: str
        The type of method.

    Returns
    -------
    method: class
        The method class.
    """

    gym_methods = MSCRegistery.get(MSCRegistery.GYM_METHODS, {})
    return gym_methods.get(role, {}).get(method_type)


def register_runner_hook(hook: Any):
    """Register a runner hook.

    Parameters
    ----------
    hook: class
        The hook class.
    """

    assert hasattr(hook, "name"), "name should be given to register hook"
    hooks = MSCRegistery.get(MSCRegistery.RUNNER_HOOKS, {})
    hooks[hook.name()] = hook
    MSCRegistery.register(MSCRegistery.RUNNER_HOOKS, hooks)
    return hook


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
