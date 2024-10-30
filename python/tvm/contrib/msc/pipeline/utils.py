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
"""tvm.contrib.msc.pipeline.config"""

import copy
from typing import List, Union, Dict, Tuple

from tvm.contrib.msc.core.tools import ToolType
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.core import utils as msc_utils


def get_tool_stage(tool_type: str) -> str:
    """Map the stage according to tool_type

    Parameters
    ----------
    tool_type: str
        The tool type.

    Returns
    -------
    stage: str
        The stage.
    """

    if tool_type == ToolType.PRUNER:
        return MSCStage.PRUNE
    if tool_type == ToolType.QUANTIZER:
        return MSCStage.QUANTIZE
    if tool_type == ToolType.DISTILLER:
        return MSCStage.DISTILL
    if tool_type == ToolType.TRACKER:
        return MSCStage.TRACK
    return tool_type


def map_tools(tools: List[dict]) -> dict:
    """Map tools from list

    Parameters
    ----------
    tools: list<dict>
        The tools config,

    Returns
    -------
    tools: dict
        The tools map.
    """

    tools_map = {t["tool_type"]: t for t in tools}
    assert len(tools_map) == len(tools), "Duplicate tools: " + str([t["tool_type"] for t in tools])
    return tools_map


def support_tool(tool: dict, stage: str, run_type: str) -> bool:
    """Check if the tool is supported

    Parameters
    ----------
    tool: dict
        The tool config,
    stage: str
        The pipeline stage.
    run_type: str
        The runtime type.

    Returns
    -------
    supported: bool
        Whether the tool is supported.
    """

    run_type = tool.get("run_type", run_type)
    if stage == MSCStage.BASELINE:
        return tool["tool_type"] == ToolType.TRACKER
    return True


def config_tool(tool_type: str, raw_config: Union[dict, str]) -> dict:
    """Config the tool

    Parameters
    ----------
    tool_type: str
        The tool type,
    raw_config: str| dict
        The tool config or style.

    Returns
    -------
    config: dict
        The config for tool.
    """

    if isinstance(raw_config, dict):
        if "config_style" in raw_config:
            config_style = raw_config.pop("config_style")
        else:
            config_style = "default"
    else:
        config_style, raw_config = raw_config, None
    configer_cls = msc_utils.get_registered_tool_configer(tool_type, config_style)
    assert configer_cls, "Can not find configer for {}:{}".format(tool_type, config_style)
    return {"tool_type": tool_type, **configer_cls().config(raw_config)}


def create_config(
    inputs: List[dict],
    outputs: List[str],
    model_type: str,
    baseline_type: str = None,
    optimize_type: str = None,
    compile_type: str = None,
    dataset: Dict[str, dict] = None,
    tools: List[Tuple[str, Union[dict, str]]] = None,
    dynamic: bool = False,
    run_config: Dict[str, dict] = None,
    skip_config: Dict[str, str] = None,
    **extra_config,
) -> dict:
    """Create config for msc pipeline

    Parameters
    ----------
    inputs: list<dict>
        The inputs info,
    outputs: list<str>
        The output names.
    model_type: str
        The model type.
    baseline_type: str
        The baseline type.
    compile_type: str
        The compile type.
    optimize_type: str
        The optimize type.
    dataset: dict<str, dict>
        The datasets for compile pipeline.
    tools: list<str, str|dict>
        The tools config.
    dynamic: bool
        Whether to config dyanmic mode.
    skip_config: dict
        The skip config for compile.
    extra_config: dict
        The extra config.
    """

    all_stages = [MSCStage.BASELINE, MSCStage.OPTIMIZE, MSCStage.COMPILE]
    baseline_type = baseline_type or model_type
    optimize_type = optimize_type or baseline_type
    compile_type = compile_type or optimize_type
    tools = tools or []
    tools = [config_tool(t_type, t_config) for t_type, t_config in tools]
    extra_config = extra_config or {}
    # basic config
    config = {
        "model_type": model_type,
        "dynamic": dynamic,
        "inputs": inputs,
        "outputs": outputs,
        "dataset": dataset,
        "tools": tools,
        MSCStage.PREPARE: {"profile": {"benchmark": {"repeat": -1}}},
        MSCStage.BASELINE: {
            "run_type": baseline_type,
            "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": -1}},
        },
    }

    # config optimize
    opt_tools = [t for t in tools if support_tool(t, MSCStage.OPTIMIZE, optimize_type)]
    if opt_tools:
        config[MSCStage.OPTIMIZE] = {
            "run_type": optimize_type,
            "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": -1}},
        }

    # config compile
    config[MSCStage.COMPILE] = {
        "run_type": compile_type,
        "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": -1}},
    }

    # update run config
    if run_config:
        if "all" in run_config:
            all_config = run_config.pop("all")
            run_config.update({s: copy.deepcopy(all_config) for s in all_stages})
        for stage, r_config in run_config.items():
            extra_config.setdefault(stage, {}).setdefault("run_config", {}).update(r_config)

    # update config
    if extra_config:
        config = msc_utils.update_dict(config, extra_config)

    # skip stages
    if skip_config:
        if "all" in run_config:
            all_config = skip_config.pop("all")
            skip_config.update({s: copy.deepcopy(all_config) for s in all_stages})
        for stage, s_type in skip_config.items():
            if stage not in config:
                continue
            if s_type == "stage":
                config.pop(stage)
            elif s_type == "profile":
                config[stage].pop("profile")
            elif s_type == "check":
                config[stage]["profile"]["check"]["err_rate"] = -1
            elif s_type == "benchmark":
                config[stage]["profile"].pop("benchmark")
            else:
                raise TypeError("Unexpected skip type " + str(s_type))
    return config
