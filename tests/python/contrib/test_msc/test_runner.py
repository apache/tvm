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

""" Test Runners in MSC. """

import pytest
import numpy as np

import torch
from torch import fx

import tvm.testing
from tvm.relax.frontend.torch import from_fx
from tvm.contrib.msc.framework.tvm.runtime import TVMRunner
from tvm.contrib.msc.core import utils as msc_utils

requires_tensorrt = pytest.mark.skipif(
    tvm.get_global_func("relax.ext.tensorrt", True) is None,
    reason="TENSORRT is not enabled",
)


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


def _test_from_torch(runner_cls, device, is_training=False, atol=1e-3, rtol=1e-3):
    """Test runner from torch model"""
    torch_model = _get_torch_model("resnet50", is_training)
    if torch_model:
        workspace = msc_utils.set_workspace()
        input_info = [([1, 3, 224, 224], "float32")]
        datas = [np.random.rand(*i[0]).astype(i[1]) for i in input_info]
        torch_datas = [torch.from_numpy(d) for d in datas]
        graph_model = fx.symbolic_trace(torch_model)
        with torch.no_grad():
            golden = torch_model(*torch_datas)
            mod = from_fx(graph_model, input_info)
        runner = runner_cls(mod, device=device, is_training=is_training)
        runner.build()
        outputs = runner.run(datas, ret_type="list")
        golden = [msc_utils.cast_array(golden)]
        for gol_r, out_r in zip(golden, outputs):
            tvm.testing.assert_allclose(gol_r, out_r, atol=atol, rtol=rtol)
        workspace.destory()


def test_tvm_runner_cpu():
    """Test runner for tvm on cpu"""

    _test_from_torch(TVMRunner, "cpu", is_training=True)


@tvm.testing.requires_gpu
def test_tvm_runner_gpu():
    """Test runner for tvm on gpu"""

    _test_from_torch(TVMRunner, "cuda", is_training=True)


if __name__ == "__main__":
    tvm.testing.main()
