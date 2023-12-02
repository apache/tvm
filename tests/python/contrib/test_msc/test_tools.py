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

""" Test Tools in MSC. """

import os
import pytest

import torch

import tvm.testing
from tvm.contrib.msc.pipeline import MSCManager
from tvm.contrib.msc.core.tools import ToolType
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils

requires_tensorrt = pytest.mark.skipif(
    tvm.get_global_func("relax.ext.tensorrt", True) is None,
    reason="TENSORRT is not enabled",
)


def _get_config(
    model_type,
    compile_type,
    tools_config,
    inputs,
    outputs,
    atol=1e-2,
    rtol=1e-2,
    optimize_type=None,
):
    """Get msc config"""
    return {
        "model_type": model_type,
        "inputs": inputs,
        "outputs": outputs,
        "debug_level": 0,
        "dataset": {"loader": "from_random", "max_iter": 5},
        "prepare": {"profile": {"benchmark": {"repeat": 10}}},
        "baseline": {
            "run_type": model_type,
            "profile": {"check": {"atol": atol, "rtol": rtol}, "benchmark": {"repeat": 10}},
        },
        "optimize": {
            "run_type": optimize_type or model_type,
            "profile": {"check": {"atol": atol, "rtol": rtol}, "benchmark": {"repeat": 10}},
            **tools_config,
        },
        "compile": {
            "run_type": compile_type,
            "profile": {"check": {"atol": atol, "rtol": rtol}, "benchmark": {"repeat": 10}},
        },
    }


def get_tool_config(tool_type):
    config = {}
    if tool_type == ToolType.PRUNER:
        config = {
            "plan_file": "msc_pruner.json",
            "strategys": [{"method": "per_channel", "density": 0.8}],
        }
    return {tool_type: config}


def _get_torch_model(name, is_training=False):
    """Get model from torch vision"""
    # pylint: disable=import-outside-toplevel
    try:
        import torchvision

        model = getattr(torchvision.models, name)(pretrained=True)
        if is_training:
            model = model.train()
        else:
            model = model.eval()
        return model
    except:  # pylint: disable=bare-except
        print("please install torchvision package")
        return None


def _test_from_torch(
    compile_type,
    tools_config,
    expected_info,
    is_training=False,
    atol=1e-2,
    rtol=1e-2,
    optimize_type=None,
):
    torch_model = _get_torch_model("resnet50", is_training)
    if torch_model:
        if torch.cuda.is_available():
            torch_model = torch_model.to(torch.device("cuda:0"))
        config = _get_config(
            MSCFramework.TORCH,
            compile_type,
            tools_config,
            inputs=[["input_0", [1, 3, 224, 224], "float32"]],
            outputs=["output"],
            atol=atol,
            rtol=rtol,
            optimize_type=optimize_type,
        )
        manager = MSCManager(torch_model, config)
        report = manager.run_pipe()
        assert report["success"], "Failed to run pipe for torch -> {}".format(compile_type)
        for t_type, config in tools_config.items():
            assert os.path.isfile(
                msc_utils.get_config_dir().relpath(config["plan_file"])
            ), "Failed to find plan of " + str(t_type)
        model_info = manager.runner.model_info
        assert msc_utils.dict_equal(
            model_info, expected_info
        ), "Model info {} mismatch with expected {}".format(model_info, expected_info)
        manager.destory()


@pytest.mark.parametrize("tool_type", [ToolType.PRUNER])
def test_tvm_tools(tool_type):
    """Test tools for tvm"""

    model_info = {
        "inputs": [
            {"name": "input_0", "shape": [1, 3, 224, 224], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [{"name": "output", "shape": [1, 1000], "dtype": "float32", "layout": "NC"}],
        "nodes": {
            "total": 229,
            "input": 1,
            "nn.conv2d": 53,
            "nn.batch_norm": 53,
            "get_item": 53,
            "nn.relu": 49,
            "nn.max_pool2d": 1,
            "add": 16,
            "nn.adaptive_avg_pool2d": 1,
            "reshape": 1,
            "msc.linear_bias": 1,
        },
    }
    tool_config = get_tool_config(tool_type)
    _test_from_torch(MSCFramework.TVM, tool_config, model_info, is_training=True)


@requires_tensorrt
@pytest.mark.parametrize(
    "tool_type,use_native",
    [(ToolType.PRUNER, False)],
)
def test_tensorrt_tools(tool_type, use_native):
    """Test tools for tensorrt"""

    model_info = {
        "inputs": [
            {"name": "input_0", "shape": [1, 3, 224, 224], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [{"name": "output", "shape": [1, 1000], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "msc_tensorrt": 1},
    }
    tool_config = get_tool_config(tool_type)
    if tool_type == ToolType.QUANTIZER and use_native:
        tool_config[ToolType.QUANTIZER]["strategys"] = []
    optimize_type = MSCFramework.TENSORRT if use_native else None
    _test_from_torch(
        MSCFramework.TENSORRT,
        tool_config,
        model_info,
        is_training=False,
        optimize_type=optimize_type,
    )


if __name__ == "__main__":
    tvm.testing.main()
