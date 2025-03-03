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
"""tvm.contrib.msc.core.tools.execute"""

from functools import wraps
from typing import List, Iterable, Any, Dict

import tvm
from tvm.contrib.msc.core.utils.namespace import MSCMap, MSCKey
from tvm.contrib.msc.core import utils as msc_utils
from .tool import ToolType, BaseTool


def _get_tool_key(tool_type: str) -> str:
    """Get the key according to tool_type

    Parameters
    -------
    tool_type: str
        The type of the tool prune| quantize| distill...

    Returns
    -------
    tool_key: str
        The tool key.
    """

    if tool_type == ToolType.PRUNER:
        return MSCKey.PRUNERS
    if tool_type == ToolType.QUANTIZER:
        return MSCKey.QUANTIZERS
    if tool_type == ToolType.DISTILLER:
        return MSCKey.DISTILLERS
    if tool_type == ToolType.TRACKER:
        return MSCKey.TRACKERS
    raise TypeError("Unexpected tool type " + str(tool_type))


def add_tool(tool: BaseTool, tool_type: str, tag: str = "main"):
    """Add tool by type and tag

    Parameters
    -------
    tool: BaseTool
        The tool.
    tool_type: str
        The type of the tool prune| quantize| distill...
    tag: str
        The tag of the tool.
    """

    tool_key = _get_tool_key(tool_type)
    tools = MSCMap.get(tool_key, {})
    tools[tag] = tool
    MSCMap.set(tool_key, tools)
    return tool


def get_tool_cls(framework: str, tool_type: str, config: dict) -> BaseTool:
    """Get the tool class

    Parameters
    -------
    framework: str
        The framework for implement
    tool_type: str
        The type of the tool prune| quantize| distill...
    config: dict
        The config of tool.
    """

    tool_style = config.pop("tool_style") if "tool_style" in config else "default"
    tool_cls = msc_utils.get_registered_tool(framework, tool_type, tool_style)
    assert tool_cls, "Can not find tool class for {}:{} @ {}".format(
        tool_type, tool_style, framework
    )
    return tool_cls


def create_tool(framework: str, tool_type: str, tag: str = "main", **config) -> BaseTool:
    """Create tool by type, config and tag

    Parameters
    -------
    framework: str
        The framework for implement
    tool_type: str
        The type of the tool prune| quantize| distill...
    tag: str
        The tag of the tool.
    config: dict
        The config of tool.
    """

    tool_cls = get_tool_cls(framework, tool_type, config)
    return add_tool(tool_cls(tag, **config), tool_type, tag)


def get_tool(tool_type: str, tag: str = "main") -> BaseTool:
    """Get tool by type and tag

    Parameters
    -------
    tool_type: str
        The type of the tool prune| quantize| distill...
    tag: str
        The tag of the tool.

    Returns
    -------
    tool: BaseTool
        The saved tool.
    """

    tool_key = _get_tool_key(tool_type)
    tools = MSCMap.get(tool_key, {})
    return tools.get(tag)


def get_tools(tag: str = "main") -> Iterable[BaseTool]:
    """Get all saved tools by tag

    Parameters
    -------
    tag: str
        The tag of the tool.

    Returns
    -------
    tools: iterable<BaseTool>
        The saved tools.
    """

    for t_type in ToolType.all_types():
        tool = get_tool(t_type, tag)
        if tool:
            yield tool


def remove_tool(tool_type: str, tag: str = "main"):
    """Remove tool by type and tag

    Parameters
    -------
    tool_type: str
        The type of the tool prune| quantize| distill...
    tag: str
        The tag of the tool.
    """

    tool_key = _get_tool_key(tool_type)
    tools = MSCMap.get(tool_key, {})
    if tag in tools:
        tools.pop(tag)
        MSCMap.set(tool_key, tools)


def remove_tools(tag: str = "main"):
    """Remove all saved tools by tag

    Parameters
    -------
    tag: str
        The tag of the tool.

    Returns
    -------
    tools: iterable<BaseTool>
        The saved tools.
    """

    for t_type in ToolType.all_types():
        remove_tool(t_type, tag)


def process_tensor(tensor: Any, name: str, consumer: str, scope: str, tag: str = "main") -> Any:
    """Process tensor with tools

    Parameters
    -------
    tensor: Any
        Tensor in framework
    name: str
        The name of the tensor.
    consumer: str
        The name of the consumer.
    scope: str
        The scope mark teacher| student| null
    tag: str
        The tag of the tool.

    Returns
    -------
    tensor: Any
        The processed tensor.
    """

    for tool in get_tools(tag):
        tensor = tool.process_tensor(tensor, name, consumer, scope)
    return tensor


@tvm.register_func("msc_tool.codegen_tensor")
def codegen_tensor(
    tensor_ctx: Dict[str, str], name: str, consumer: str, scope: str, tag: str = "main"
) -> List[str]:
    """Codegen processed tensor describe with tools

    Parameters
    -------
    tensor_ctx: dict<str, str>
        Tensor describe items.
    name: str
        The name of the tensor.
    consumer: str
        The name of the consumer.
    scope: str
        The scope mark teacher| student| null
    tag: str
        The tag of the tool.

    Returns
    -------
    processed: list<str>
        The tensor describe for processed tensor.
    """

    tensor_ctx = {**dict(tensor_ctx), "processed": []}
    tensor_ctx = process_tensor(dict(tensor_ctx), name, consumer, scope, tag)
    return tensor_ctx["processed"]


def wrap_step(step: str, tag: str = "main") -> callable:
    """Wrapper for tool execution

    Parameters
    -------
    step: str
        The step for tool execution build| forward
    tag: str
        The tag of the tool.

    Returns
    -------
    decorate: callable
        The decorate.
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for tool in get_tools(tag):
                if step == "build":
                    tool.execute_before_build(*args, **kwargs)
                elif step == "forward":
                    tool.execute_before_forward(*args, **kwargs)
                else:
                    raise TypeError("Unexpected step " + str(step))
            output = func(*args, **kwargs)
            for tool in get_tools(tag):
                if step == "build":
                    output = tool.execute_after_build(output)
                elif step == "forward":
                    output = tool.execute_after_forward(output)
                else:
                    raise TypeError("Unexpected step " + str(step))
            return output

        return wrapper

    return decorate


def execute_step(step: str, *args, **kwargs):
    """Execute tools for a step

    Parameters
    -------
    step: str
        The step for tool execution build| forward
    args: list<Any>
        The arguments for model build.
    kwargs: dict<Any>
        The key word arguments for model build.
    """

    if step in ("before_build", "before_forward"):
        output = None
    else:
        assert (
            len(args) == 1 and not kwargs
        ), "after step only accept 1 argument, get args {}, kwargs {}".format(args, kwargs)
        output = args[0]
    tag = kwargs.pop("tag") if "tag" in kwargs else "main"
    for tool in get_tools(tag):
        if step == "before_build":
            tool.execute_before_build(*args, **kwargs)
        elif step == "before_forward":
            tool.execute_before_forward(*args, **kwargs)
        elif step == "after_build":
            output = tool.execute_after_build(output)
        elif step == "after_forward":
            output = tool.execute_after_forward(output)
        else:
            raise TypeError("Unexpected step " + str(step))
    return output


def _execute_step_with_context(
    step_ctx: Dict[str, Any], step: str, graph_name: str, tag: str = "main"
) -> Dict[str, Any]:
    """Execute step with contect

    Parameters
    -------
    step_ctx: dict<str, any>
        The step context.
    step: str
        The step for tool execution build| forward
    graph_name: str
        The graph name.
    tag: str
        The tag of the tool.

    Returns
    -------
    step_ctx: dict<str, any>
        The processed step context.
    """

    for tool in get_tools(tag):
        if step == "before_build":
            tool.execute_before_build(step_ctx, graph_name=graph_name)
        elif step == "before_forward":
            tool.execute_before_forward(step_ctx, graph_name=graph_name)
        elif step == "after_build":
            step_ctx = tool.execute_after_build(step_ctx)
        elif step == "after_forward":
            step_ctx = tool.execute_after_forward(step_ctx)
        else:
            raise TypeError("Unexpected step " + str(step))
    return step_ctx


@tvm.register_func("msc_tool.codegen_step")
def codegen_step(
    step_ctx: Dict[str, str], step: str, graph_name: str, tag: str = "main"
) -> List[str]:
    """Codegen step codes

    Parameters
    -------
    step_ctx: dict<str, str>
        The step describe items.
    step: str
        The step for tool execution build| forward
    graph_name: str
        The graph name.
    tag: str
        The tag of the tool.

    Returns
    -------
    processed: list<str>
        The tensor describe for processed tensor.
    """

    step_ctx = {**dict(step_ctx), "processed": []}
    step_ctx = _execute_step_with_context(step_ctx, step, graph_name, tag)
    return step_ctx["processed"]


@tvm.register_func("msc_tool.callback_step")
def callback_step(step_ctx: Dict[str, Any], step: str, graph_name: str = "main", tag: str = "main"):
    """Execute tools for a step

    Parameters
    -------
    step_ctx: dict<str, Any>
        The step context.
    step: str
        The step for tool execution build| forward
    graph_name: str
        The graph name.
    tag: str
        The tag of the tool.
    """

    _execute_step_with_context(step_ctx, step, graph_name, tag)
