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

""" Test Pipeline in MSC. """

import json
import pytest
import torch

import tvm.testing
from tvm.contrib.msc.pipeline import MSCManager, TorchDynamic
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils

requires_tensorrt = pytest.mark.skipif(
    tvm.get_global_func("relax.ext.tensorrt", True) is None,
    reason="TENSORRT is not enabled",
)


def _get_config(model_type, compile_type, inputs, outputs, dynamic=False, atol=1e-1, rtol=1e-1):
    """Get msc config"""

    path = "test_pipe_{}_{}_{}".format(model_type, compile_type, "dynamic" if dynamic else "static")
    return {
        "workspace": msc_utils.msc_dir(path, keep_history=False),
        "verbose": "critical",
        "model_type": model_type,
        "inputs": inputs,
        "outputs": outputs,
        "dataset": {"prepare": {"loader": "from_random", "max_iter": 5}},
        "prepare": {"profile": {"benchmark": {"repeat": 10}}},
        "baseline": {
            "run_type": model_type,
            "profile": {"check": {"atol": atol, "rtol": rtol}, "benchmark": {"repeat": 10}},
        },
        "compile": {
            "run_type": compile_type,
            "profile": {"check": {"atol": atol, "rtol": rtol}, "benchmark": {"repeat": 10}},
        },
    }


def _get_torch_model(name, training=False):
    """Get model from torch vision"""

    # pylint: disable=import-outside-toplevel
    try:
        import torchvision

        model = getattr(torchvision.models, name)()
        if training:
            model = model.train()
        else:
            model = model.eval()
        return model
    except:  # pylint: disable=bare-except
        print("please install torchvision package")
        return None


def _get_tf_graph():
    """Get graph from tensorflow"""

    # pylint: disable=import-outside-toplevel
    try:
        from tvm.contrib.msc.framework.tensorflow import tf_v1
        import tvm.relay.testing.tf as tf_testing

        tf_graph = tf_v1.Graph()
        with tf_graph.as_default():
            graph_def = tf_testing.get_workload(
                "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz",
                "mobilenet_v2_1.4_224_frozen.pb",
            )
            # Call the utility to import the graph definition into default graph.
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        return graph_def
    except:  # pylint: disable=bare-except
        print("please install tensorflow package")
        return None


def _check_pipeline(pipeline, expected_info, dynamic=False):
    """Check the pipeline results"""

    passed, err = True, ""
    if not pipeline.report["success"]:
        passed = False
        err = "Failed to run pipe for {} -> {}".format(pipeline.model_type, pipeline.compile_type)
    if not dynamic:
        model_info = pipeline.get_runtime().model_info
        if not msc_utils.dict_equal(model_info, expected_info):
            passed = False
            err = "Model info {} mismatch with expected {}".format(model_info, expected_info)
    pipeline.destory()
    if not passed:
        raise Exception("{}\nReport:{}".format(err, json.dumps(pipeline.report, indent=2)))


def _test_from_torch(
    compile_type, expected_info, training=False, dynamic=False, atol=1e-1, rtol=1e-1
):
    if dynamic and not hasattr(torch, "compile"):
        return

    torch_model = _get_torch_model("resnet50", training)
    if torch_model:
        if torch.cuda.is_available():
            torch_model = torch_model.to(torch.device("cuda:0"))
        config = _get_config(
            MSCFramework.TORCH,
            compile_type,
            inputs=[["input_0", [1, 3, 224, 224], "float32"]],
            outputs=["output"],
            dynamic=dynamic,
            atol=atol,
            rtol=rtol,
        )
        pipeline = TorchDynamic(torch_model, config) if dynamic else MSCManager(torch_model, config)
        pipeline.run_pipe()
        _check_pipeline(pipeline, expected_info, dynamic)


def _test_from_tf(compile_type, expected_info, atol=1e-2, rtol=1e-2):
    graphdef = _get_tf_graph()
    if graphdef:
        config = _get_config(
            MSCFramework.TENSORFLOW,
            compile_type,
            inputs=[["input", [1, 224, 224, 3], "float32"]],
            outputs=["MobilenetV2/Predictions/Reshape_1:0"],
            atol=atol,
            rtol=rtol,
        )
        config["compile"]["profile"]["check"]["err_rate"] = -1
        manager = MSCManager(graphdef, config)
        manager.run_pipe()
        _check_pipeline(manager, expected_info)


@pytest.mark.parametrize("dynamic", [False, True])
def test_tvm_pipeline(dynamic):
    """Test pipeline for tvm"""

    model_info = {
        "inputs": [
            {"name": "input_0", "shape": [1, 3, 224, 224], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [{"name": "output", "shape": [1, 1000], "dtype": "float32", "layout": "NW"}],
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
    _test_from_torch(MSCFramework.TVM, model_info, training=False, dynamic=dynamic)

    if not dynamic:
        model_info = {
            "inputs": [
                {"name": "input", "shape": [1, 224, 224, 3], "dtype": "float32", "layout": "NHWC"}
            ],
            "outputs": [
                {
                    "name": "MobilenetV2/Predictions/Reshape_1:0",
                    "shape": [1, 1001],
                    "dtype": "float32",
                    "layout": "NC",
                }
            ],
            "nodes": {
                "total": 138,
                "input": 1,
                "msc.conv2d_bias": 36,
                "clip": 35,
                "nn.conv2d": 17,
                "nn.batch_norm": 17,
                "get_item": 17,
                "add": 10,
                "nn.avg_pool2d": 1,
                "squeeze": 1,
                "reshape": 2,
                "nn.softmax": 1,
            },
        }
        _test_from_tf(MSCFramework.TVM, model_info)


@pytest.mark.parametrize("dynamic", [False, True])
def test_torch_pipeline(dynamic):
    """Test pipeline for torch"""

    model_info = {
        "inputs": [
            {"name": "input_0", "shape": [1, 3, 224, 224], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [{"name": "output", "shape": [1, 1000], "dtype": "float32", "layout": "NW"}],
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
    _test_from_torch(MSCFramework.TORCH, model_info, training=False, dynamic=dynamic)


def test_tensorflow_pipeline():
    """Test manager for tensorflow"""

    model_info = {
        "inputs": [
            {"name": "input", "shape": [1, 224, 224, 3], "dtype": "float32", "layout": "NHWC"}
        ],
        "outputs": [
            {
                "name": "MobilenetV2/Predictions/Reshape_1:0",
                "shape": [1, 1001],
                "dtype": "float32",
                "layout": "NC",
            }
        ],
        "nodes": {
            "total": 138,
            "input": 1,
            "msc.conv2d_bias": 36,
            "clip": 35,
            "nn.conv2d": 17,
            "nn.batch_norm": 17,
            "get_item": 17,
            "add": 10,
            "nn.avg_pool2d": 1,
            "squeeze": 1,
            "reshape": 2,
            "nn.softmax": 1,
        },
    }
    _test_from_tf(MSCFramework.TENSORFLOW, model_info)


@requires_tensorrt
@pytest.mark.parametrize("dynamic", [False, True])
def test_tensorrt_pipeline(dynamic):
    """Test pipeline for tensorrt"""

    model_info = {
        "inputs": [
            {"name": "input_0", "shape": [1, 3, 224, 224], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [{"name": "output", "shape": [1, 1000], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "msc_tensorrt": 1},
    }
    _test_from_torch(MSCFramework.TENSORRT, model_info, training=False, dynamic=dynamic)


if __name__ == "__main__":
    tvm.testing.main()
