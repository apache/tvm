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

from typing import List, Union, Dict, Tuple

from tvm.contrib.msc.core.tools import ToolType
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.core import utils as msc_utils


def support_tool(tool: dict, stage: str, run_type: str) -> bool:
    """Check if the tool is supported

    Parameters
    ----------
    tool: dict
        The tool config,
    stage: str
        The compile stage.
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
    skip_config: dict
        The skip config for compile.
    extra_config: dict
        The extra config.
    """

    baseline_type = baseline_type or model_type
    optimize_type = optimize_type or baseline_type
    compile_type = compile_type or optimize_type
    tools = tools or []
    tools = [config_tool(t_type, t_config) for t_type, t_config in tools]
    # basic config
    config = {
        "model_type": model_type,
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

    # update config
    if extra_config:
        config = msc_utils.update_dict(config, extra_config)

    # skip stages
    skip_config = skip_config or {}
    for stage in [MSCStage.BASELINE, MSCStage.OPTIMIZE, MSCStage.COMPILE]:
        if stage not in config:
            continue
        for key in ["all", stage]:
            if key not in skip_config:
                continue
            if skip_config[key] == "stage":
                config.pop(stage)
            elif skip_config[key] == "profile":
                config[stage].pop("profile")
            elif skip_config[key] == "check":
                config[stage]["profile"].pop("check")
            elif skip_config[key] == "benchmark":
                config[stage]["profile"].pop("benchmark")
            else:
                raise TypeError("Unexpected skip type " + str(skip_config[key]))

    return config
