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
# pylint: disable=import-self, invalid-name, unused-argument
"""Unit tests for various models and operators"""
import os
import platform
import sys

from packaging import version as package_version

import pytest
import numpy as np
import tvm
import tvm.testing
from tvm import relay
from tvm.contrib import graph_executor
from tvm.contrib.nvcc import have_fp16
from tvm.contrib import cudnn

import torch
from torch.nn import Module
from torch.nn import functional as F
import torchvision

sys.setrecursionlimit(10000)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def list_ops(expr):
    """list_ops"""

    class OpLister(tvm.relay.ExprVisitor):
        """OpLister inherits from ExprVisitor"""

        def visit_op(self, op):
            if op not in self.node_set:
                self.node_list.append(op)
            return super().visit_op(op)

        def list_nodes(self, expr):
            self.node_set = {}
            self.node_list = []
            self.visit(expr)
            return self.node_list

    return OpLister().list_nodes(expr)


def assert_shapes_match(tru, est):
    """Verfiy whether the shapes are equal"""
    if tru.shape != est.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(tru.shape, est.shape))


def load_torchvision(model_name):
    """Given a model name, returns a Torchvision model in eval mode as well
    as an example input."""
    with torch.no_grad():
        if model_name.startswith("inception"):
            height = width = 299
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            height = width = 224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        input_shape = [1, 3, height, width]
        input_data = torch.randn(input_shape).float()
        for channel in range(3):
            input_data[:, channel] -= mean[channel]
            input_data[:, channel] /= std[channel]

        if model_name.startswith("googlenet"):
            model = getattr(torchvision.models, model_name)(pretrained=True, aux_logits=True)
        else:
            model = getattr(torchvision.models, model_name)(pretrained=True)
        model = model.float().eval()
        return model, [input_data]


def load_pretrainedmodels(model_name):
    """Given a model name, returns a pretrainedmodels.pytorch model in eval
    mode as well as an example input."""
    # pylint: disable=import-outside-toplevel
    import pretrainedmodels  # https://github.com/Cadene/pretrained-models.pytorch

    model = getattr(pretrainedmodels, model_name)().float().eval()
    input_shape = [1, *model.input_size]
    input_data = torch.rand(input_shape).float() * 256
    for channel in range(3):
        input_data[:, channel] -= model.mean[channel]
        input_data[:, channel] /= model.std[channel]
    return model, [input_data]


def load_model(model_name):
    """Given a model name, returns a model as well as an example input."""
    if hasattr(torchvision.models, model_name):
        return load_torchvision(model_name)
    # pylint: disable=import-outside-toplevel
    try:
        import pretrainedmodels

        if hasattr(pretrainedmodels, model_name):
            return load_pretrainedmodels(model_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Please install pretrainedmodels.pytorch") from e
    raise RuntimeError("Model not supported")


def verify_model(
    model_name,
    input_data=None,
    custom_convert_map=None,
    rtol=1e-5,
    atol=1e-5,
    expected_ops=None,
    kind="graph",
    check_correctness=True,
    cpu_only=False,
):
    """Assert that the output of a compiled model matches with that of its
    baseline."""
    input_data = [] if input_data is None else input_data
    custom_convert_map = custom_convert_map or {}
    expected_ops = expected_ops or []
    if isinstance(model_name, str):
        baseline_model, baseline_input = load_model(model_name)
    elif isinstance(input_data, list):
        baseline_model = model_name
        baseline_input = input_data
    elif isinstance(input_data, torch.Tensor) or not input_data.shape:
        baseline_model = model_name
        baseline_input = [input_data]
    else:
        assert False, "Unexpected input format"
    if torch.cuda.is_available():
        if isinstance(baseline_model, torch.nn.Module):
            baseline_model = baseline_model.cuda()
        baseline_input = [inp.cuda() for inp in baseline_input]

    with torch.no_grad():
        baseline_outputs = baseline_model(*[input.clone() for input in baseline_input])

    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.cpu().numpy(),)

    trace = torch.jit.trace(baseline_model, [input.clone() for input in baseline_input])
    if isinstance(baseline_model, torch.nn.Module):
        trace = trace.float().eval()

        if torch.cuda.is_available():
            trace = trace.cuda()
        else:
            trace = trace.cpu()

    input_names = [f"input{idx}" for idx, _ in enumerate(baseline_input)]
    input_shapes = list(zip(input_names, [inp.shape for inp in baseline_input]))
    mod, params = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
    for arg in mod["main"].params[: len(input_names)]:
        assert arg.name_hint in input_names
    compiled_input = dict(zip(input_names, [inp.clone().cpu().numpy() for inp in baseline_input]))

    targets = ["llvm"]
    if not cpu_only:
        targets.append("cuda")

    with tvm.transform.PassContext(opt_level=3):
        for target in targets:
            if not tvm.runtime.enabled(target):
                continue
            dev = tvm.device(target, 0)
            exe = relay.create_executor(
                kind, mod=mod, params=params, device=dev, target=target
            ).evaluate()
            result = exe(**compiled_input)
            if not isinstance(result, list):
                result = [result]

            for i, baseline_output in enumerate(baseline_outputs):
                output = result[i].numpy()

                assert_shapes_match(baseline_output, output)
                if check_correctness:
                    tvm.testing.assert_allclose(baseline_output, output, rtol=rtol, atol=atol)

    if expected_ops:

        def visit(op):
            if isinstance(op, tvm.ir.op.Op):
                if op.name in expected_ops:
                    expected_ops.remove(op.name)

        tvm.relay.analysis.post_order_visit(mod["main"].body, visit)

        if expected_ops:
            msg = "TVM Relay do not contain expected ops {}"
            raise AssertionError(msg.format(expected_ops))

    del model_name
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verify_model_with_input(
    test_func,
    input_data,
    *,
    input_dict=None,
    custom_convert_map=None,
    rtol=1e-5,
    atol=1e-5,
    assert_shape_only=False,
):
    """Generic function to generate and compare Pytorch and TVM output"""
    input_dict = input_dict or {}
    custom_convert_map = custom_convert_map or {}
    baseline_outputs = test_func(*input_data)
    trace = torch.jit.trace(test_func, [input.clone() for input in input_data])
    input_names = [f"input{idx}" for idx, _ in enumerate(input_data)]
    input_shapes = list(zip(input_names, [inp.shape for inp in input_data]))
    mod, params = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
    with tvm.transform.PassContext(opt_level=3):
        for target in ["llvm", "cuda"]:
            if not tvm.runtime.enabled(target):
                continue
            dev = tvm.device(target, 0)
            lib = relay.build(mod, target=target, params=params)
            relay_model = graph_executor.GraphModule(lib["default"](dev))
            for name, value in input_dict.items():
                relay_model.set_input(name, value)
            relay_model.run()

            compiled_output = relay_model.get_output(0).numpy()
            assert_shapes_match(baseline_outputs, compiled_output)
            if assert_shape_only is False:
                tvm.testing.assert_allclose(baseline_outputs, compiled_output, rtol=rtol, atol=atol)


# Single operator tests
@tvm.testing.uses_gpu
def test_forward_pixel_shuffle():
    """test_forward_pixel_shuffle"""
    torch.set_grad_enabled(False)
    input_shape = [1, 144, 16, 16]

    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.PixelShuffle(2).float().eval(), input_data=input_data)
    verify_model(torch.nn.PixelShuffle(3).float().eval(), input_data=input_data)
    verify_model(torch.nn.PixelShuffle(4).float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_add():
    """test_forward_add"""
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Add1(Module):
        def forward(self, *args):
            return args[0] + args[0]

    class Add2(Module):
        def forward(self, *args):
            return args[0] + 1

    class Add3(Module):
        def forward(self, *args):
            ones = torch.ones(input_shape, dtype=torch.float)
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] + ones

    class Add4(Module):
        def forward(self, *args):
            ones = torch.ones([], dtype=torch.float)
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] + ones

    input_data = torch.rand(input_shape).float()
    verify_model(Add1().float().eval(), input_data=input_data)
    verify_model(Add2().float().eval(), input_data=input_data)
    verify_model(Add3().float().eval(), input_data=input_data)
    verify_model(Add4().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_subtract():
    """test_forward_subtract"""
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Subtract1(Module):
        def forward(self, *args):
            return args[0] - args[0]

    class Subtract2(Module):
        def forward(self, *args):
            return args[0] - 1

    class Subtract3(Module):
        def forward(self, *args):
            ones = torch.ones(input_shape)
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] - ones

    class Subtract4(Module):
        def forward(self, *args):
            ones = torch.ones([])
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] - ones

    input_data = torch.rand(input_shape).float()
    verify_model(Subtract1().float().eval(), input_data=input_data)
    verify_model(Subtract2().float().eval(), input_data=input_data)
    verify_model(Subtract3().float().eval(), input_data=input_data)
    verify_model(Subtract4().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_multiply():
    """test_forward_multiply"""
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Multiply1(Module):
        def forward(self, *args):
            return args[0] * args[0]

    class Multiply2(Module):
        def forward(self, *args):
            return args[0] * 1.0

    class Multiply3(Module):
        def forward(self, *args):
            ones = torch.ones(input_shape)
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] * ones

    class Multiply4(Module):
        def forward(self, *args):
            ones = torch.ones([])
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] * ones

    input_data = torch.rand(input_shape).float()
    verify_model(Multiply1().float().eval(), input_data=input_data)
    verify_model(Multiply2().float().eval(), input_data=input_data)
    verify_model(Multiply3().float().eval(), input_data=input_data)
    verify_model(Multiply4().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_min_max():
    """test_min_max"""

    class Max(Module):
        def forward(self, inp):
            return torch.max(inp)

    class Min(Module):
        def forward(self, inp):
            return torch.min(inp)

    class Max2(Module):
        def forward(self, inp):
            out, _ = torch.max(inp, 1, keepdim=True)
            return out

    class Min2(Module):
        def forward(self, inp):
            out, _ = torch.min(inp, 0, keepdim=False)
            return out

    class Max3(Module):
        def forward(self, lhs, rhs):
            return torch.max(lhs, rhs)

    class Min3(Module):
        def forward(self, lhs, rhs):
            return torch.min(lhs, rhs)

    class Max4(Module):
        def forward(self, inp):
            out = torch.amax(inp, (1, 2), keepdim=True)
            return out

    class Min4(Module):
        def forward(self, inp):
            out = torch.amin(inp, (0, 3), keepdim=False)
            return out

    input_data = [torch.rand((10, 10, 10, 10)), torch.rand((10, 10, 10, 10))]

    verify_model(Max(), input_data=input_data[0])
    verify_model(Min(), input_data=input_data[0])
    verify_model(Max2(), input_data=input_data[0])
    verify_model(Min2(), input_data=input_data[0])
    verify_model(Max3(), input_data=input_data)
    verify_model(Min3(), input_data=input_data)
    verify_model(Max4(), input_data=input_data[0])
    verify_model(Min4(), input_data=input_data[0])


@tvm.testing.uses_gpu
def test_minimum_maximum():
    """test_minimum_maximum"""

    class Maximum(Module):
        def forward(self, lhs, rhs):
            return torch.maximum(lhs, rhs)

    class Minimum(Module):
        def forward(self, lhs, rhs):
            return torch.minimum(lhs, rhs)

    input_data = [torch.rand((10, 10, 10, 10)), torch.rand((10, 10, 10, 10))]

    verify_model(Maximum(), input_data=input_data)
    verify_model(Minimum(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_reciprocal():
    """test_forward_reciprocal"""
    torch.set_grad_enabled(False)
    input_shape = [2, 1, 10, 1, 10]

    class Reciprocal1(Module):
        def forward(self, *args):
            return args[0].reciprocal()

    input_data = torch.rand(input_shape).float()
    verify_model(Reciprocal1().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_repeat():
    """test_forward_repeat"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3]

    class Repeat1(Module):
        def forward(self, *args):
            return args[0].repeat(1, 1)

    class Repeat2(Module):
        def forward(self, *args):
            return args[0].repeat(4, 2)

    class Repeat3(Module):
        def forward(self, *args):
            return args[0].repeat(4, 2, 1)

    input_data = torch.rand(input_shape).float()
    verify_model(Repeat1().float().eval(), input_data=input_data)
    verify_model(Repeat2().float().eval(), input_data=input_data)
    verify_model(Repeat3().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_repeat_interleave():
    """test_forward_repeat_interleave"""
    torch.set_grad_enabled(False)
    input_shape = [2, 2, 3]

    class RepeatInterleave1(Module):
        def forward(self, *args):
            return args[0].repeat_interleave(2)

    class RepeatInterleave2(Module):
        def forward(self, *args):
            return args[0].repeat_interleave(3, dim=0)

    class RepeatInterleave3(Module):
        def forward(self, *args):
            return args[0].repeat_interleave(2, dim=1)

    class RepeatInterleave4(Module):
        def forward(self, *args):
            return args[0].repeat_interleave(4, dim=2)

    input_data = torch.rand(input_shape).float()
    verify_model(RepeatInterleave1().float().eval(), input_data=input_data)
    verify_model(RepeatInterleave2().float().eval(), input_data=input_data)
    verify_model(RepeatInterleave3().float().eval(), input_data=input_data)
    verify_model(RepeatInterleave4().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_unsqueeze():
    """test_forward_unsqueeze"""
    torch.set_grad_enabled(False)
    input_shape = [10, 10]

    class Unsqueeze1(Module):
        def forward(self, *args):
            return args[0].unsqueeze(2)

    class Unsqueeze2(Module):
        def forward(self, *args):
            _ = args[0].unsqueeze_(2)
            # Check whether operations after inplace unsqueeze works as expected
            y = args[0].squeeze(2)
            return torch.add(y, y)

    input_data = torch.rand(input_shape).float()
    verify_model(Unsqueeze1().float().eval(), input_data=input_data)
    verify_model(Unsqueeze2().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_squeeze():
    """test_forward_squeeze"""
    torch.set_grad_enabled(False)
    input_shape = [2, 1, 10, 1, 10]

    class Squeeze1(Module):
        def forward(self, *args):
            return args[0].squeeze()

    class Squeeze2(Module):
        def forward(self, *args):
            return args[0].squeeze(1)

    input_data = torch.rand(input_shape).float()
    verify_model(Squeeze1().float().eval(), input_data=input_data)
    verify_model(Squeeze2().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_arange():
    """test_forward_arange"""
    torch.set_grad_enabled(False)

    class Arange1(Module):
        def forward(self, *args):
            return torch.arange(5)

    class Arange2(Module):
        def forward(self, *args):
            return torch.arange(2.5)

    class Arange3(Module):
        def forward(self, *args):
            return torch.arange(1, 4)

    class Arange4(Module):
        def forward(self, *args):
            return torch.arange(1, 2.5, 0.5)

    class Arange5(Module):
        def forward(self, *args):
            return torch.arange(1, 2, 1, dtype=torch.int32)

    class Arange6(Module):
        def forward(self, *args):
            return torch.arange(start=1, end=6, step=2)

    class Arange7(Module):
        def forward(self, *args):
            return torch.arange(1, 4, dtype=torch.float32)

    class Arange8(Module):
        def forward(self, *args):
            return torch.arange(1, 2, 1, dtype=torch.int16)

    class Arange9(Module):
        def forward(self, *args):
            end = torch.add(torch.tensor(4), 1)
            return torch.arange(end) + torch.ones((5,), dtype=torch.int64)

    class Arange10(Module):
        def forward(self, *args):
            end = torch.add(torch.tensor(4.0), torch.tensor(1.0))
            return torch.arange(end) + torch.ones((5,), dtype=torch.float)

    class Arange11(Module):
        def forward(self, *args):
            start = torch.add(torch.tensor(1), 1)
            end = torch.add(torch.tensor(4), 1)
            step = torch.add(torch.tensor(2), 1)
            out = torch.arange(start, end, step)
            return out + torch.ones((3,), dtype=torch.int64)

    class Arange12(Module):
        def forward(self, *args):
            start = torch.add(torch.tensor(1), 1)
            end = torch.add(torch.tensor(4), 1)
            step = torch.add(torch.tensor(2.5), torch.tensor(4.1))
            out = torch.arange(start, end, step)
            return out + torch.ones((3,), dtype=torch.float)

    verify_model(Arange1().float().eval())
    verify_model(Arange2().float().eval())
    verify_model(Arange3().float().eval())
    verify_model(Arange4().float().eval())
    verify_model(Arange5().float().eval())
    verify_model(Arange6().float().eval())
    verify_model(Arange7().float().eval())
    verify_model(Arange8().float().eval())
    verify_model(Arange9().float().eval())
    verify_model(Arange10().float().eval())
    verify_model(Arange11().float().eval())
    verify_model(Arange12().float().eval())


@tvm.testing.uses_gpu
def test_forward_mesh_grid():
    """test_forward_mesh_grid"""
    torch.set_grad_enabled(False)

    class MeshGrid1(Module):
        def forward(self, *args):
            x = torch.tensor([1, 2, 3])
            y = torch.tensor([4, 5, 6])
            grid_x, grid_y = torch.meshgrid([x, y])
            return grid_x, grid_y

    class MeshGrid2(Module):
        def forward(self, *args):
            x = torch.tensor([1, 2, 3], dtype=torch.float32)
            y = torch.add(torch.tensor(5, dtype=torch.float32), 1)
            grid_x, grid_y = torch.meshgrid([x, y])
            return grid_x, grid_y

    verify_model(MeshGrid1().float().eval())
    verify_model(MeshGrid2().float().eval())


@tvm.testing.uses_gpu
def test_forward_abs():
    """test_forward_abs"""
    torch.set_grad_enabled(False)
    input_shape = [2, 1, 10, 1, 10]

    class Abs1(Module):
        def forward(self, *args):
            return args[0].abs()

    input_data = torch.rand(input_shape).float()
    verify_model(Abs1().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_concatenate():
    """test_forward_concatenate"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Concatenate1(Module):
        def forward(self, *args):
            return torch.cat([args[0][:, 0].unsqueeze(1), args[0][:, 1].unsqueeze(1)], 1)

    class Concatenate2(Module):
        def forward(self, *args):
            a = (args[0][:, :, 0] + 2) * 7
            b = (args[0][:, :, 1] + 3) * 11
            c = (args[0][:, :, 2] + 5) * 13
            return torch.cat([t.unsqueeze(2) for t in [a, b, c]], 2)

    input_data = torch.rand(input_shape).float()
    verify_model(Concatenate1().float().eval(), input_data=input_data)
    verify_model(Concatenate2().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_relu():
    """test_forward_relu"""
    torch.set_grad_enabled(False)
    input_shape = [10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.ReLU().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_relu6():
    """test_forward_relu6"""
    torch.set_grad_enabled(False)
    input_shape = [10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.ReLU6().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_prelu():
    """test_forward_prelu"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.PReLU(num_parameters=3).eval(), input_data=input_data)
    # Test when input channel > 1 and num parameters = 1
    verify_model(torch.nn.PReLU(num_parameters=1).eval(), input_data=input_data)
    # Test when input dims < 2
    verify_model(torch.nn.PReLU(num_parameters=1).eval(), input_data=torch.randn(2))


@tvm.testing.uses_gpu
def test_forward_leakyrelu():
    """test_forward_leakyrelu"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.LeakyReLU().eval(), input_data=input_data)
    verify_model(torch.nn.LeakyReLU(negative_slope=0.05).eval(), input_data=input_data)
    verify_model(torch.nn.LeakyReLU(negative_slope=1.0, inplace=True).eval(), input_data=input_data)
    verify_model(
        torch.nn.LeakyReLU(negative_slope=1.25, inplace=True).eval(), input_data=input_data
    )


@tvm.testing.uses_gpu
def test_forward_elu():
    """test_forward_elu"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.randn(input_shape).float()
    verify_model(torch.nn.ELU().eval(), input_data=input_data)
    verify_model(torch.nn.ELU(alpha=0.3).eval(), input_data=input_data)
    verify_model(torch.nn.ELU(alpha=1.0).eval(), input_data=input_data)
    verify_model(torch.nn.ELU(alpha=1.3).eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_celu():
    """test_forward_celu"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.CELU().eval(), input_data=input_data)
    verify_model(torch.nn.CELU(alpha=0.3).eval(), input_data=input_data)
    verify_model(torch.nn.CELU(alpha=1.0).eval(), input_data=input_data)
    verify_model(torch.nn.CELU(alpha=1.3).eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_gelu():
    """test_forward_gelu"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.GELU().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_selu():
    """test_forward_selu"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.SELU().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_silu():
    """test_forward_silu"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.SiLU().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_glu():
    """test_forward_glu"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.GLU().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_softplus():
    """test_forward_softplus"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.Softplus().eval(), input_data=input_data)
    verify_model(torch.nn.Softplus(beta=1.5, threshold=20).eval(), input_data=input_data)
    verify_model(torch.nn.Softplus(beta=5, threshold=10).eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_softsign():
    """test_forward_softsign"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.Softsign().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_log_sigmoid():
    """test_forward_log_sigmoid"""
    torch.set_grad_enabled(False)
    input_shape = [10, 10]
    input_data = torch.rand(input_shape).float()
    input_data_overflow = torch.tensor([-300.0, -100.0]).float()
    verify_model(torch.nn.LogSigmoid().eval(), input_data=input_data)
    verify_model(torch.nn.LogSigmoid().eval(), input_data=input_data_overflow)


@tvm.testing.uses_gpu
def test_forward_adaptive_avgpool():
    """test_forward_adaptive_avgpool"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.AdaptiveAvgPool2d([1, 1]).eval(), input_data=input_data)
    verify_model(torch.nn.AdaptiveAvgPool2d([10, 10]).eval(), input_data=input_data)

    input_data = torch.rand([1, 3, 10]).float()
    verify_model(torch.nn.AdaptiveAvgPool1d([1]).eval(), input_data=input_data)
    verify_model(torch.nn.AdaptiveAvgPool1d([5]).eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_adaptive_maxpool():
    """test_forward_adaptive_maxpool"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.AdaptiveMaxPool2d([1, 1]).eval(), input_data=input_data)
    verify_model(torch.nn.AdaptiveMaxPool2d([10, 10]).eval(), input_data=input_data)

    input_data = torch.rand([1, 3, 10]).float()
    verify_model(torch.nn.AdaptiveMaxPool1d([1]).eval(), input_data=input_data)
    verify_model(torch.nn.AdaptiveMaxPool1d([5]).eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_maxpool2d():
    """test_forward_maxpool2d"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()

    verify_model(torch.nn.MaxPool2d(kernel_size=[1, 1]).eval(), input_data)
    verify_model(torch.nn.MaxPool2d(kernel_size=[2, 2], dilation=[2, 3]).eval(), input_data)
    verify_model(torch.nn.MaxPool2d(kernel_size=[10, 10]).eval(), input_data)
    verify_model(torch.nn.MaxPool2d(kernel_size=[4, 4], padding=2, stride=2).eval(), input_data)

    # A functional variant (default strides = None case)
    class MaxPool2D(Module):
        def forward(self, *args):
            return torch.nn.functional.max_pool2d(args[0], kernel_size=[10, 10])

    verify_model(MaxPool2D(), input_data=input_data)

    class MaxPool2DWithIndices(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[1, 1], return_indices=True)

        def forward(self, *args):
            output, _ = self.pool(args[0])
            return output

    class MaxPool2DWithIntStrides(Module):
        def forward(self, *args):
            # Makes kernel_size and strides a Relay expr to test converting back to int
            x_shape = args[0].shape
            # kernel_size = [torch.tensor(x_shape[1]).int(), torch.tensor(x_shape[1]).int()]
            strides = [torch.tensor(x_shape[0]).int(), torch.tensor(x_shape[0]).int()]
            return torch.nn.functional.max_pool2d(args[0], kernel_size=[4, 4], stride=strides)

    verify_model(MaxPool2DWithIndices().float().eval(), input_data=input_data)
    verify_model(MaxPool2DWithIntStrides().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_maxpool1d():
    """test_forward_maxpool1d"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10]
    input_data = torch.rand(input_shape).float()

    verify_model(torch.nn.MaxPool1d(kernel_size=1).eval(), input_data)
    verify_model(torch.nn.MaxPool1d(kernel_size=2, dilation=[1]).eval(), input_data)
    verify_model(torch.nn.MaxPool1d(kernel_size=10).eval(), input_data)
    verify_model(torch.nn.MaxPool1d(kernel_size=4, padding=2, stride=2).eval(), input_data)

    # A functional variant (default strides = None case)
    class MaxPool1D(Module):
        def forward(self, *args):
            return torch.nn.functional.max_pool1d(args[0], kernel_size=10)

    verify_model(MaxPool1D(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_maxpool3d():
    """test_forward_maxpool3d"""
    torch.set_grad_enabled(False)
    for input_shape in [(1, 3, 10, 10, 10), (3, 10, 10, 10)]:
        input_data = torch.rand(input_shape).float()

        verify_model(torch.nn.MaxPool3d(kernel_size=[1, 1, 1]).eval(), input_data)
        verify_model(
            torch.nn.MaxPool3d(kernel_size=[2, 2, 2], dilation=[1, 2, 3]).eval(), input_data
        )
        verify_model(torch.nn.MaxPool3d(kernel_size=[10, 10, 10]).eval(), input_data)
        verify_model(
            torch.nn.MaxPool3d(kernel_size=[4, 4, 4], padding=2, stride=2).eval(), input_data
        )

    # A functional variant (default strides = None case)
    class MaxPool3D(Module):
        def forward(self, *args):
            return torch.nn.functional.max_pool3d(args[0], kernel_size=[10, 10, 10])

    verify_model(MaxPool3D(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_split():
    """test_forward_split"""
    torch.set_grad_enabled(False)
    input_shape = [4, 10]

    class Split(Module):
        def __init__(self, split_size_or_sections, dim):
            super().__init__()
            self.split_size_or_sections = split_size_or_sections
            self.dim = dim

        def forward(self, *args):
            return torch.split(args[0], self.split_size_or_sections, self.dim)

    input_data = torch.rand(input_shape).float()
    verify_model(Split(2, 0).float().eval(), input_data=input_data)
    verify_model(Split(3, 1).float().eval(), input_data=input_data)
    verify_model(Split(4, 1).float().eval(), input_data=input_data)
    verify_model(Split([2, 3, 5], 1).float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_tensor_split():
    """test_forward_tensor_split"""
    torch.set_grad_enabled(False)
    input_shape = [4, 10]

    class Tensor_Split(Module):
        def __init__(self, split_size_or_sections, dim):
            super().__init__()
            self.split_size_or_sections = split_size_or_sections
            self.dim = dim

        def forward(self, *args):
            return torch.tensor_split(args[0], self.split_size_or_sections, self.dim)

    input_data = torch.rand(input_shape).float()
    verify_model(Tensor_Split(2, 0).float().eval(), input_data=input_data)
    verify_model(Tensor_Split(torch.tensor(3), 1).float().eval(), input_data=input_data)
    verify_model(Tensor_Split([2, 3, 5], 1).float().eval(), input_data=input_data)
    verify_model(Tensor_Split((2, 3, 5), 1).float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_avgpool1d():
    """test_forward_avgpool1d"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10]

    class AvgPool1D2(Module):
        def forward(self, *args):
            return torch.nn.functional.avg_pool1d(args[0], kernel_size=[10])

    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.AvgPool1d(kernel_size=[10]).eval(), input_data=input_data)
    verify_model(AvgPool1D2().float().eval(), input_data=input_data)
    verify_model(
        torch.nn.AvgPool1d(kernel_size=[5], stride=2, padding=2).eval(), input_data=input_data
    )


@tvm.testing.uses_gpu
def test_forward_avgpool2d():
    """test_forward_avgpool2d"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class AvgPool2D2(Module):
        def forward(self, *args):
            return torch.nn.functional.avg_pool2d(args[0], kernel_size=[10, 10])

    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.AvgPool2d(kernel_size=[10, 10]).eval(), input_data=input_data)
    verify_model(AvgPool2D2().float().eval(), input_data=input_data)
    verify_model(
        torch.nn.AvgPool2d(kernel_size=5, stride=2, padding=2).eval(), input_data=input_data
    )

    input_shape = [1, 1, 1, 9]
    input_data = torch.rand(input_shape).float()
    verify_model(
        torch.nn.AvgPool2d(
            kernel_size=[1, 2], stride=[1, 2], ceil_mode=True, count_include_pad=True
        ).eval(),
        input_data=input_data,
    )


@tvm.testing.uses_gpu
def test_forward_avgpool3d():
    """test_forward_avgpool3d"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10, 10]

    class AvgPool3D1(Module):
        def forward(self, *args):
            return torch.nn.functional.avg_pool3d(args[0], kernel_size=[10, 10, 10])

    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.AvgPool3d(kernel_size=[10, 10, 10]).eval(), input_data=input_data)
    verify_model(AvgPool3D1().float().eval(), input_data=input_data)
    verify_model(
        torch.nn.AvgPool3d(kernel_size=5, stride=2, padding=2).eval(), input_data=input_data
    )


@tvm.testing.uses_gpu
def test_forward_hardtanh():
    """test_forward_hardtanh"""
    torch.set_grad_enabled(False)
    input_shape = [10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.Hardtanh().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_conv():
    """test_forward_conv"""
    torch.set_grad_enabled(False)
    conv1d_input_shape = [1, 3, 10]
    conv2d_input_shape = [1, 3, 10, 10]

    class Conv2D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)
            self.softmax = torch.nn.Softmax()

        def forward(self, *args):
            return self.softmax(self.conv(args[0]))

    class Conv2D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=False)
            self.softmax = torch.nn.Softmax()

        def forward(self, *args):
            return self.softmax(self.conv(args[0]))

    class Conv2D3(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, groups=3, bias=False)
            self.softmax = torch.nn.Softmax()

        def forward(self, *args):
            return self.softmax(self.conv(args[0]))

    class Conv1D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 6, 7)
            self.softmax = torch.nn.Softmax()

        def forward(self, *args):
            return self.softmax(self.conv(args[0]))

    class Conv1D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 6, 7, bias=False)
            self.softmax = torch.nn.Softmax()

        def forward(self, *args):
            return self.softmax(self.conv(args[0]))

    class Conv1D3(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 6, 7, groups=3, bias=False)
            self.softmax = torch.nn.Softmax()

        def forward(self, *args):
            return self.softmax(self.conv(args[0]))

    conv2d_input_data = torch.rand(conv2d_input_shape).float()
    verify_model(Conv2D1().float().eval(), input_data=conv2d_input_data)
    verify_model(Conv2D2().float().eval(), input_data=conv2d_input_data)
    # depth wise conv with channel mult 2
    verify_model(Conv2D3().float().eval(), input_data=conv2d_input_data)
    # group conv
    verify_model(
        torch.nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), groups=2).eval(),
        input_data=torch.randn((1, 8, 16, 16)),
    )

    conv1d_input_data = torch.rand(conv1d_input_shape).float()
    verify_model(Conv1D1().float().eval(), input_data=conv1d_input_data)
    verify_model(Conv1D2().float().eval(), input_data=conv1d_input_data)
    verify_model(Conv1D3().float().eval(), input_data=conv1d_input_data)


@tvm.testing.uses_gpu
@pytest.mark.parametrize("in_channels", [3], ids=lambda x: "in_channels=" + str(x))
@pytest.mark.parametrize("out_channels", [5], ids=lambda x: "out_channels=" + str(x))
@pytest.mark.parametrize("kernel_size", [3], ids=lambda x: "kernel_size=" + str(x))
@pytest.mark.parametrize("output_padding", [0, 1, 2], ids=lambda x: "output_padding=" + str(x))
@pytest.mark.parametrize("groups", [1], ids=lambda x: "groups=" + str(x))
@pytest.mark.parametrize("bias", [True, False], ids=lambda x: "bias=" + str(x))
def test_forward_conv_transpose(
    in_channels, out_channels, kernel_size, output_padding, bias, groups
):
    """test_forward_conv_transpose"""
    # Note we do not test with groups  > 1 because that is not supported
    # in tvm for conv transpose operations

    # Output padding must be smaller than either stride or dilation so we
    # opt to make the stride 1 + output padding
    stride = output_padding + 1

    # Conv 3D Transpose Tests
    conv3d_input_shape = [1, in_channels, 16, 16, 16]
    conv3d_input_data = torch.rand(conv3d_input_shape).float()
    conv3d_transpose = torch.nn.ConvTranspose3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
    ).eval()
    verify_model(conv3d_transpose, conv3d_input_data)

    # Conv 2D Transpose Tests
    conv2d_input_shape = [1, in_channels, 128, 256]
    conv2d_input_data = torch.rand(conv2d_input_shape).float()
    conv2d_transpose = torch.nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
    ).eval()
    verify_model(conv2d_transpose, conv2d_input_data)

    # # Conv 1D Transpose Tests
    conv1d_input_shape = [1, in_channels, 10]
    conv1d_input_data = torch.rand(conv1d_input_shape).float()
    conv1d_transpose = torch.nn.ConvTranspose1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
    ).eval()
    verify_model(conv1d_transpose, conv1d_input_data)


@tvm.testing.uses_gpu
def test_forward_conv2d_transpose_group():
    """test_forward_conv2d_transpose_group"""
    # https://github.com/apache/tvm/issues/10223

    class ModulatedConvTranspose2D(torch.nn.Module):
        """ModulatedConvTranspose2D module"""

        def forward(self, x, w, s):
            """forward"""
            B, C, H, W = x.shape
            I, O, KH, KW = w.shape

            # weight is different for each input in batch (this is why we want grouped conv
            # transpose)
            w = w.unsqueeze(0) * s.reshape(B, 1, 1, 1, 1)
            w = w.reshape(B * I, O, KH, KW)
            x = x.reshape(1, B * C, H, W)
            x = torch.nn.functional.conv_transpose2d(
                x, w, stride=(2, 2), padding=(1, 1), output_padding=(1, 1), groups=B
            )
            return x.reshape(B, O, H * 2, W * 2)

    b, c, h, w, k = 4, 512, 8, 16, 3
    inputs = torch.rand(b, c, h, w)
    weights = torch.rand(c, c // 2, k, k)
    styles = torch.rand(b)

    # cuda not supported for group > 1 conv2d_transpose
    targets = ["llvm"]

    if cudnn.exists():
        targets.append("cuda -libs=cudnn")

    verify_trace_model(ModulatedConvTranspose2D().eval(), [inputs, weights, styles], targets)


def test_forward_deform_conv():
    """test_forward_deform_conv"""
    torch.set_grad_enabled(False)

    def test_run(
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        offset_groups,
        kh,
        kw,
        groups,
    ):
        input_shape = [batch_size, in_channels, in_height, in_width]
        offset_shape = [batch_size, 2 * offset_groups * kh * kw, out_height, out_width]
        weight_shape = [out_channels, in_channels // groups, kh, kw]
        input_data = torch.rand(input_shape)
        offset_data = torch.rand(offset_shape)
        weight_data = torch.rand(weight_shape)

        class DeformConv2D(Module):
            def forward(self, *args):
                return torchvision.ops.deform_conv2d(args[0], args[1], args[2])

        verify_model(
            DeformConv2D().float().eval(),
            input_data=[input_data, offset_data, weight_data],
            rtol=1e-4,
            atol=1e-4,
        )

    batch_size = 4
    in_channels, out_channels = 4, 6
    in_height, in_width = 10, 10
    out_height, out_width = 8, 8
    offset_groups = 2
    kh, kw = 3, 3
    groups = 1

    test_run(
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        offset_groups,
        kh,
        kw,
        groups,
    )

    batch_size = 5
    in_channels, out_channels = 4, 6
    in_height, in_width = 10, 10
    out_height, out_width = 8, 8
    offset_groups = 1
    kh, kw = 3, 3
    groups = 1

    test_run(
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        offset_groups,
        kh,
        kw,
        groups,
    )


@tvm.testing.uses_gpu
def test_forward_threshold():
    """test_forward_threshold"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.Threshold(0, 0).float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_contiguous():
    """test_forward_contiguous"""
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Contiguous1(Module):
        def forward(self, *args):
            return args[0].contiguous()

    input_data = torch.rand(input_shape).float()
    verify_model(Contiguous1().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_batchnorm():
    """test_forward_batchnorm"""

    def init_weight(m):
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.normal_(m.bias)

    inp_2d = torch.rand((1, 16, 10, 10))
    inp_3d = torch.rand((1, 16, 10, 10, 10))

    for bn, inp in [(torch.nn.BatchNorm2d(16), inp_2d), (torch.nn.BatchNorm3d(16), inp_3d)]:
        init_weight(bn.eval())
        verify_model(bn.eval(), input_data=inp)


@tvm.testing.uses_gpu
def test_forward_instancenorm():
    """test_forward_instancenorm"""
    inp_2d = torch.rand((1, 16, 10, 10))
    inp_3d = torch.rand((1, 16, 10, 10, 10))

    for ins_norm, inp in [
        (torch.nn.InstanceNorm2d(16), inp_2d),
        (torch.nn.InstanceNorm3d(16), inp_3d),
    ]:
        verify_model(ins_norm.eval(), input_data=inp)


@tvm.testing.uses_gpu
def test_forward_layernorm():
    """test_forward_layernorm"""

    def init_weight(m):
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.normal_(m.bias, 0.02)

    inp_2d = torch.rand((1, 16, 10, 10))
    inp_3d = torch.rand((1, 16, 10, 10, 10))
    for ln, inp in [(torch.nn.LayerNorm(10), inp_2d), (torch.nn.LayerNorm(10), inp_3d)]:
        init_weight(ln.eval())
        verify_model(ln.eval(), input_data=inp)


@tvm.testing.uses_gpu
def test_forward_groupnorm():
    """test_forward_groupnorm"""
    input_shape = [10, 6, 5, 5]
    input_data = torch.rand(input_shape).float()

    # Separate 6 channels into 3 groups
    verify_model(torch.nn.GroupNorm(3, 6).eval(), input_data=input_data)

    # Put all 6 channels into a single group (equivalent with LayerNorm)
    verify_model(torch.nn.GroupNorm(1, 6).eval(), input_data=input_data)

    # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
    verify_model(torch.nn.GroupNorm(6, 6).eval(), input_data=input_data)

    input_shape = [1, 10, 4, 7]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.GroupNorm(1, 10).eval(), input_data=input_data)
    verify_model(torch.nn.GroupNorm(2, 10).eval(), input_data=input_data)
    verify_model(torch.nn.GroupNorm(5, 10).eval(), input_data=input_data)
    verify_model(torch.nn.GroupNorm(10, 10).eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_reshape():
    """test_forward_reshape"""
    torch.set_grad_enabled(False)
    input_shape = [2, 1, 10, 1, 10]
    new_shape = [2, 1, 10, 10]

    class Reshape1(Module):
        def forward(self, *args):
            return args[0].reshape(new_shape)

    class Reshape2(Module):
        def forward(self, *args):
            return args[0].reshape([-1])

    class Reshape3(torch.nn.Module):
        def forward(self, x):
            x_shape = x.shape
            return x.reshape((x_shape[0] * x_shape[1], x_shape[2]))

    input_data = torch.rand(input_shape).float()
    verify_model(Reshape1(), input_data=input_data)
    verify_model(Reshape2(), input_data=input_data)
    verify_model(Reshape3(), input_data=torch.randn(2, 3, 4))


@tvm.testing.uses_gpu
def test_forward_reshape_as():
    """test_forward_reshape_as"""

    def test_func(input_tensor, other_tensor):
        return input_tensor.reshape_as(other_tensor)

    input_data = [torch.rand([2, 1, 10, 1, 10]), torch.rand([2, 1, 10, 10])]

    verify_model_with_input(test_func, input_data, input_dict={"input0": input_data[0]})


@tvm.testing.uses_gpu
def test_flatten():
    """test_flatten"""

    def _test_flatten(start_dim, end_dim):
        return lambda inp: torch.flatten(inp, start_dim, end_dim)

    inp = torch.rand((3, 5, 2, 2))

    # [3, 5, 2, 2] -> [60]
    verify_model(_test_flatten(0, -1), inp)
    verify_model(_test_flatten(0, 3), inp)
    verify_model(_test_flatten(-4, 3), inp)
    verify_model(_test_flatten(-4, -1), inp)

    # [3, 5, 2, 2] -> [3, 5, 2, 2]
    verify_model(_test_flatten(3, -1), inp)
    verify_model(_test_flatten(-1, -1), inp)
    verify_model(_test_flatten(0, -4), inp)
    verify_model(_test_flatten(-4, -4), inp)

    # [3, 5, 2, 2] -> [3, 10, 2]
    verify_model(_test_flatten(1, 2), inp)
    verify_model(_test_flatten(1, -2), inp)
    verify_model(_test_flatten(-3, 2), inp)
    verify_model(_test_flatten(-3, -2), inp)


@tvm.testing.uses_gpu
def test_forward_transpose():
    """test_forward_transpose"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Transpose1(Module):
        def forward(self, *args):
            return args[0].transpose(2, 3)

    class Transpose2(Module):
        def forward(self, *args):
            return args[0].transpose(-2, -1)

    class Transpose3(Module):
        def forward(self, *args):
            return args[0].permute(0, 2, 3, 1)

    input_data = torch.rand(input_shape).float()
    verify_model(Transpose1().float().eval(), input_data=input_data)
    verify_model(Transpose2().float().eval(), input_data=input_data)
    verify_model(Transpose3().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_numpy_T():
    """test_forward_numpy_T"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    def test_fn(x):
        return x.T

    input_data = torch.rand(input_shape).float()
    verify_model(test_fn, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_size():
    """test_forward_size"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3]

    class Size1(Module):
        def forward(self, *args):
            return float(args[0].size(0)) * args[0]

    input_data = torch.rand(input_shape).float()
    verify_model(Size1().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_type_as():
    """test_type_as"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3]

    def _create_module(dtype):
        class TypeAs(Module):
            def forward(self, *args):
                expected_type_tensor = torch.zeros(1, 3, dtype=dtype)
                return args[0].type_as(expected_type_tensor)

        return TypeAs()

    input_data = torch.randn(input_shape).float()
    verify_model(_create_module(torch.float64), input_data=input_data)
    verify_model(_create_module(torch.float32), input_data=input_data)
    verify_model(_create_module(torch.int64), input_data=input_data)
    verify_model(_create_module(torch.int32), input_data=input_data)
    verify_model(_create_module(torch.int16), input_data=input_data)
    verify_model(_create_module(torch.int8), input_data=input_data)

    if torch.cuda.is_available():
        check_fp16 = False
        try:
            # Only check half precision on supported hardwares.
            if have_fp16(tvm.cuda(0).compute_version):
                check_fp16 = True
        # pylint: disable=broad-except
        except Exception:
            # If GPU is not enabled in TVM, skip the fp16 test.
            pass

        # Temporary disable fp16 test
        check_fp16 = False

        if check_fp16:
            verify_model(_create_module(torch.float16), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_view():
    """test_forward_view"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class View1(Module):
        def forward(self, *args):
            return args[0].view((1, 3 * 10 * 10))

    class View2(Module):
        def forward(self, *args):
            return args[0].view(args[0].shape[0], -1)

    class View3(Module):
        def forward(self, *args):
            d1 = torch.tensor(3) * torch.tensor(10) * torch.tensor(10)
            return args[0].view(args[0].shape[0], d1)

    input_data = torch.rand(input_shape).float()
    verify_model(View1().float().eval(), input_data=input_data)
    verify_model(View2().float().eval(), input_data=input_data)
    verify_model(View3().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_select():
    """test_forward_select"""
    torch.set_grad_enabled(False)
    input_shape = [5, 3, 10, 10]

    class Select1(Module):
        def forward(self, *args):
            return args[0].select(1, 1)

    class IndexedSelect(Module):
        def __init__(self, inp, dim):
            super().__init__()
            self.inp = inp
            self.dim = dim
            if torch.cuda.is_available():
                self.inp = self.inp.cuda()

        def forward(self, index):
            return torch.index_select(self.inp, self.dim, index)

    input_data = torch.rand(input_shape).float()
    verify_model(Select1().float().eval(), input_data=input_data)

    # test negative indexing
    verify_model(lambda x: x[-1], input_data=input_data)

    x = torch.randn(3, 4)
    indices = torch.tensor([0, 2])
    verify_model(IndexedSelect(x, 0).eval(), input_data=indices)
    verify_model(IndexedSelect(x, 1).eval(), input_data=indices)


@tvm.testing.uses_gpu
def test_forward_clone():
    """test_forward_clone"""
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Clone1(Module):
        def forward(self, *args):
            return args[0].clone()

    input_data = torch.rand(input_shape).float()
    verify_model(Clone1().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_gather():
    """test_forward_gather"""
    torch.set_grad_enabled(False)

    class Gather1(Module):
        def forward(self, *args):
            return torch.gather(args[0], 0, args[1])

    class Gather2(Module):
        def forward(self, *args):
            return torch.gather(args[0], 1, args[1])

    class Gather3(Module):
        def forward(self, *args):
            return torch.gather(args[0], 2, args[1])

    input_data = torch.rand((4,)).float()
    index = torch.tensor([1])
    verify_model(Gather1().float().eval(), input_data=[input_data, index])

    input_data = torch.rand((2, 2)).float()
    index = torch.tensor([[1, 0], [0, 1]])
    verify_model(Gather1().float().eval(), input_data=[input_data, index])

    input_data = torch.tensor([[1, 2], [3, 4]])
    index = torch.tensor([[0, 0], [1, 0]])
    verify_model(Gather2().float().eval(), input_data=[input_data, index])

    input_data = torch.rand((2, 2)).float()
    index = torch.tensor([[1, 0], [0, 1]])
    verify_model(Gather2().float().eval(), input_data=[input_data, index])

    input_data = torch.rand((3, 3, 3)).float()
    index = torch.tensor(
        [
            [[1, 0, 0], [1, 0, 1], [0, 1, 1]],
            [[1, 1, 1], [1, 2, 1], [1, 0, 1]],
            [[1, 2, 1], [1, 2, 1], [1, 2, 1]],
        ]
    )
    verify_model(Gather3().float().eval(), input_data=[input_data, index])


@tvm.testing.uses_gpu
def test_forward_logsoftmax():
    """test_forward_logsoftmax"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class LogSoftmax1(Module):
        def forward(self, *args):
            return torch.nn.LogSoftmax(dim=1)(args[0][0, 0])

    input_data = torch.rand(input_shape).float()
    verify_model(LogSoftmax1().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_norm():
    """test_forward_norm"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Norm1(Module):
        def forward(self, *args):
            return torch.norm(args[0], p=float("inf"), dim=None, keepdim=False)

    class Norm2(Module):
        def forward(self, *args):
            return torch.norm(args[0], p=float("-inf"), dim=None, keepdim=False)

    class Norm3(Module):
        def forward(self, *args):
            return torch.norm(args[0], p=float("-inf"), dim=None, keepdim=True)

    class Norm4(Module):
        def forward(self, *args):
            return torch.norm(args[0], p=float("inf"), dim=(1, 2), keepdim=False)

    class Norm5(Module):
        def forward(self, *args):
            return torch.norm(args[0], p=float("inf"), dim=(1), keepdim=True)

    class Norm6(Module):
        def forward(self, *args):
            return torch.norm(args[0], p=float(0.5), dim=(1), keepdim=True)

    class Norm7(Module):
        def forward(self, *args):
            return torch.norm(args[0], p=float(1), dim=None, keepdim=False)

    class Norm8(Module):
        def forward(self, *args):
            return torch.norm(args[0], p=float(2.0), dim=(1), keepdim=True)

    class Norm9(Module):
        def forward(self, *args):
            return torch.norm(args[0], p=float(-0.5), dim=(1, 2), keepdim=True)

    class Norm10(Module):
        def forward(self, *args):
            return torch.norm(args[0], p=float(-2), dim=(1), keepdim=False)

    input_data = torch.rand(input_shape).float()
    verify_model(Norm1().float().eval(), input_data=input_data)
    verify_model(Norm2().float().eval(), input_data=input_data)
    verify_model(Norm3().float().eval(), input_data=input_data)
    verify_model(Norm4().float().eval(), input_data=input_data)
    verify_model(Norm5().float().eval(), input_data=input_data)
    verify_model(Norm6().float().eval(), input_data=input_data)
    verify_model(Norm7().float().eval(), input_data=input_data)
    verify_model(Norm8().float().eval(), input_data=input_data)
    verify_model(Norm9().float().eval(), input_data=input_data)
    verify_model(Norm10().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_frobenius_norm():
    """test_forward_frobenius_norm"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class FroNorm1(Module):
        def forward(self, *args):
            return torch.norm(args[0])

    class FroNorm2(Module):
        def forward(self, *args):
            return torch.norm(args[0], p="fro", dim=None, keepdim=True)

    class FroNorm3(Module):
        def forward(self, *args):
            return torch.norm(args[0], p="fro", dim=(1), keepdim=True)

    class FroNorm4(Module):
        def forward(self, *args):
            return torch.norm(args[0], dim=None, keepdim=False)

    input_data = torch.rand(input_shape).float()
    verify_model(FroNorm1().float().eval(), input_data=input_data)
    verify_model(FroNorm2().float().eval(), input_data=input_data)
    verify_model(FroNorm3().float().eval(), input_data=input_data)
    verify_model(FroNorm4().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_sigmoid():
    """test_forward_sigmoid"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.Sigmoid().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_dense():
    """test_forward_dense"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Dense1(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=True)

        def forward(self, *args):
            return self.linear(args[0][0, 0])

    class Dense2(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=False)

        def forward(self, *args):
            return self.linear(args[0][0, 0])

    input_data = torch.rand(input_shape).float()
    verify_model(Dense1().float().eval(), input_data=input_data)
    verify_model(Dense2().float().eval(), input_data=input_data)

    trace = torch.jit.trace(Dense1(), [input_data])
    mod, _ = relay.frontend.from_pytorch(
        trace,
        [("input", input_shape)],
    )
    assert not any(list(op.name == "multiply" for op in list_ops(mod["main"])))


@tvm.testing.uses_gpu
def test_forward_linear():
    """test_forward_linear"""
    torch.set_grad_enabled(False)

    class Linear(Module):
        def forward(self, inputs, weight, bias):
            return F.linear(inputs, weight, bias)

    class LinearNoBias(Module):
        def forward(self, inputs, weight):
            return F.linear(inputs, weight)

    class LinearNested(torch.nn.Module):
        def forward(self, x, y, z):
            return F.linear(x, F.linear(y, z))

    input2d = torch.rand([2, 2]).float()
    input3d = torch.rand([4, 3, 2]).float()
    weight1d = torch.rand([2]).float()
    weight2d = torch.rand([2, 2]).float()
    weight3x2 = torch.rand([3, 2]).float()
    bias1d = torch.rand([2]).float()
    bias2d = torch.rand([2, 2]).float()
    # 2D input, 2D weight, 1D bias
    verify_model(Linear(), input_data=[input2d, weight2d, bias1d])
    # 2D input, 2D weight, 2D bias
    verify_model(Linear(), input_data=[input2d, weight2d, bias2d])
    # 2D input, 2D weight, no bias
    verify_model(LinearNoBias(), input_data=[input2d, weight2d])
    verify_model(LinearNoBias(), input_data=[input2d, weight3x2])
    # 2D input, 1D weight, 1D bias is not supported by torch.linear()
    # 2D input, 1D weight, no bias
    verify_model(LinearNoBias(), input_data=[input2d, weight1d])
    # 3D input, 2D weight, no bias
    verify_model(LinearNoBias(), input_data=[input3d, weight3x2])
    # 3D input, 2D weight, 1D bias
    verify_model(Linear(), input_data=[input3d, weight2d, bias1d])

    verify_model(LinearNested(), input_data=[torch.randn(10, 10) for _ in range(3)])

    # TODO: Add the following cases when matmul(1D, _) is supported by TVM
    # 1D input, 2D weight, 1D bias
    # 1D input, 2D weight, no bias
    # 1D input, 1D weight, scalar bias
    # 1D input, 1D weight, no bias


@tvm.testing.uses_gpu
def test_forward_dropout():
    """test_forward_dropout"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.Dropout(p=0.5).eval(), input_data=input_data[0, 0])
    verify_model(torch.nn.Dropout2d(p=0.5).eval(), input_data=input_data[0])
    verify_model(torch.nn.Dropout3d(p=0.5).eval(), input_data=input_data)
    verify_model(torch.nn.AlphaDropout(p=0.5).eval(), input_data=input_data[0, 0])


@tvm.testing.uses_gpu
def test_forward_slice():
    """test_forward_slice"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Slice1(Module):
        def forward(self, *args):
            return args[0][:, :, :, :3]

    class Slice2(Module):
        def forward(self, *args):
            return args[0][0, :, :-3, :]

    class Slice3(Module):
        def forward(self, *args):
            x0 = torch.tensor(2) - torch.tensor(1)
            x1 = torch.tensor(3) + torch.tensor(1)
            return args[0][:, x0:, 1:x1, :]

    class SliceWithStride(torch.nn.Module):
        def forward(self, x):
            return x[..., 0::2] + x[..., 1::2]

    class SliceWithStride2(torch.nn.Module):
        def forward(self, x):
            return x[0::2, 0::2] + x[1::2, 1::2]

    class DynamicLengthSlice(torch.nn.Module):
        def forward(self, values, length):
            return values[0:length]

    input_data = torch.rand(input_shape).float()
    verify_model(Slice1(), input_data=input_data)
    verify_model(Slice2(), input_data=input_data)
    verify_model(Slice3(), input_data=input_data)
    verify_model(SliceWithStride(), input_data=torch.randn(1, 4))
    verify_model(SliceWithStride2(), input_data=torch.randn(4, 4))

    inp = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    slice_len = torch.tensor(2)
    targets = ["llvm", "cuda"]
    verify_trace_model(DynamicLengthSlice(), [inp, slice_len], targets)


@tvm.testing.uses_gpu
def test_forward_narrow():
    """test_forward_narrow"""
    torch.set_grad_enabled(False)
    input_shape = [3, 3]

    class Narrow1(Module):
        def forward(self, *args):
            return torch.narrow(args[0], 0, 0, 2)

    class Narrow2(Module):
        def forward(self, *args):
            return torch.narrow(args[0], 1, 1, 2)

    class Narrow3(Module):
        def forward(self, *args):
            begin = torch.tensor(2) - torch.tensor(1)
            length = torch.tensor(1) * torch.tensor(2)
            return torch.narrow(args[0], 1, begin, length)

    input_data = torch.rand(input_shape).float()
    verify_model(Narrow1(), input_data=input_data)
    verify_model(Narrow2(), input_data=input_data)
    verify_model(Narrow3(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_mean():
    """test_forward_mean"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Mean1(Module):
        def forward(self, *args):
            return args[0].mean(2)

    input_data = torch.rand(input_shape).float()
    verify_model(Mean1().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_expand():
    """test_forward_expand"""
    torch.set_grad_enabled(False)

    class Expand1(Module):
        def forward(self, *args):
            return args[0].expand((3, -1, -1, -1))

    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(Expand1().float().eval(), input_data=input_data)

    class Expand2(Module):
        def forward(self, *args):
            return args[0].expand((3, 3, 3, 1))

    input_shape = [3, 1]
    input_data = torch.rand(input_shape).float()
    verify_model(Expand2().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_broadcast_tensors():
    """test_forward_broadcast_tensors"""
    torch.set_grad_enabled(False)

    class BroadCastTensors1(Module):
        def forward(self, x, y):
            return torch.broadcast_tensors(x, y)

    x = torch.arange(3).view(1, 1, 3)
    y = torch.arange(2).view(1, 2, 1)
    verify_model(BroadCastTensors1().float().eval(), input_data=[x, y])

    class BroadCastTensors2(Module):
        def forward(self, x, y, z):
            return torch.broadcast_tensors(x, y, z)

    x = torch.arange(3).view(1, 1, 3)
    y = torch.arange(2).view(1, 2, 1)
    z = torch.arange(4).view(4, 1, 1)
    verify_model(BroadCastTensors2().float().eval(), input_data=[x, y, z])


@tvm.testing.uses_gpu
def test_forward_pow():
    """test_forward_pow"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Pow1(Module):
        def forward(self, *args):
            return args[0] ** 2

    input_data = torch.rand(input_shape).float()
    verify_model(Pow1().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_chunk():
    """test_forward_chunk"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 14, 14]

    class Chunk1(Module):
        def forward(self, *args):
            chunks = args[0].chunk(7, 2)
            return torch.cat(chunks, 2)

    input_data = torch.rand(input_shape).float()
    verify_model(Chunk1().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_upsample():
    """test_upsample"""

    class Upsample(Module):
        def __init__(self, size=None, scale=None, mode="nearest", align_corners=None):
            super().__init__()
            self.size = size
            self.scale = scale
            self.mode = mode
            self.align_corners = align_corners

        def forward(self, x):
            return torch.nn.functional.interpolate(
                x,
                size=self.size,
                scale_factor=self.scale,
                mode=self.mode,
                align_corners=self.align_corners,
            )

    inp = torch.rand((1, 3, 32, 32))
    verify_model(Upsample(size=(64, 64), mode="nearest"), inp)
    verify_model(Upsample(scale=2, mode="nearest"), inp)
    verify_model(Upsample(size=(50, 50), mode="nearest"), inp)
    verify_model(Upsample(size=(64, 64), mode="bilinear", align_corners=True), inp)
    verify_model(Upsample(scale=2, mode="bilinear", align_corners=True), inp)
    verify_model(Upsample(size=(50, 50), mode="bilinear", align_corners=True), inp)
    verify_model(Upsample(size=(64, 64), mode="bicubic", align_corners=True), inp)
    verify_model(Upsample(scale=2, mode="bicubic", align_corners=True), inp)
    verify_model(Upsample(size=(50, 50), mode="bicubic", align_corners=True), inp)


@tvm.testing.uses_gpu
def test_to():
    """test for aten::to(...)"""

    class ToCPU(Module):
        def forward(self, x):
            return x.to("cpu")

    class ToFloat(Module):
        def forward(self, x):
            return x.float()

    class ToInt(Module):
        def forward(self, x):
            return x.int()

    class ToLong(Module):
        def forward(self, x):
            return x.long()

    class ToDouble(Module):
        def forward(self, x):
            return x.double()

    class ToFloat16(Module):
        def forward(self, x):
            return x.to(torch.float16)

    verify_model(ToCPU().eval(), torch.rand((1, 3, 32, 32)))
    verify_model(ToFloat().eval(), torch.zeros((1, 3, 32, 32), dtype=torch.int))
    verify_model(ToFloat().eval(), torch.tensor(2, dtype=torch.int))
    verify_model(ToInt().eval(), torch.zeros((1, 3, 32, 32)))
    verify_model(ToInt().eval(), torch.tensor(0.8))
    verify_model(ToLong().eval(), torch.tensor(0.8))
    verify_model(ToDouble().eval(), torch.tensor(0.8))
    verify_model(ToFloat16().eval(), torch.tensor(2, dtype=torch.float32))
    verify_model(ToFloat16().eval(), torch.zeros((1, 3, 32, 32), dtype=torch.int))


@tvm.testing.uses_gpu
def test_adaptive_pool3d():
    """test_adaptive_pool3d"""
    for ishape in [(1, 32, 16, 16, 16), (1, 32, 9, 15, 15), (1, 32, 13, 7, 7)]:
        inp = torch.rand(ishape)
        verify_model(torch.nn.AdaptiveMaxPool3d((1, 1, 1)).eval(), inp)
        verify_model(torch.nn.AdaptiveMaxPool3d((2, 2, 2)).eval(), inp)
        verify_model(torch.nn.AdaptiveAvgPool3d((1, 1, 1)).eval(), inp)
        verify_model(torch.nn.AdaptiveAvgPool3d((2, 2, 2)).eval(), inp)
        verify_model(torch.nn.AdaptiveAvgPool3d((4, 8, 8)).eval(), inp)
        verify_model(torch.nn.AdaptiveMaxPool3d((7, 8, 9)).eval(), inp)


@tvm.testing.uses_gpu
def test_forward_functional_pad():
    """test_forward_functional_pad"""
    torch.set_grad_enabled(False)
    pad = (0, 0)

    class Pad1(Module):
        def forward(self, *args):
            return torch.nn.functional.pad(args[0], pad, "constant", 0)

    input_data = torch.rand((3, 3, 4, 2))
    pad = (1, 1)
    verify_model(Pad1().float().eval(), input_data=input_data)

    pad = (1, 1, 2, 2)
    verify_model(Pad1().float().eval(), input_data=input_data)

    pad = (0, 1, 2, 1, 3, 3)
    verify_model(Pad1().float().eval(), input_data=input_data)

    class Pad2(Module):
        def forward(self, *args):
            return torch.nn.functional.pad(args[0], pad, "constant", 1)

    input_data = torch.rand((3, 3, 4, 2))
    pad = (1, 1)
    verify_model(Pad2().float().eval(), input_data=input_data)

    pad = (1, 1, 2, 2)
    verify_model(Pad2().float().eval(), input_data=input_data)

    pad = (0, 1, 2, 1, 3, 3)
    verify_model(Pad2().float().eval(), input_data=input_data)

    class Pad3(Module):
        def forward(self, *args):
            return torch.nn.functional.pad(args[0], pad, "constant", 1.0)

    input_data = torch.rand((3, 3, 4, 2))
    pad = (1, 1)
    verify_model(Pad3().float().eval(), input_data=input_data)

    pad = (1, 1, 2, 2)
    verify_model(Pad3().float().eval(), input_data=input_data)

    pad = (0, 1, 2, 1, 3, 3)
    verify_model(Pad3().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_zero_pad2d():
    """test_forward_zero_pad2d"""
    inp = torch.rand((1, 1, 3, 3))
    verify_model(torch.nn.ZeroPad2d(2).eval(), inp)
    verify_model(torch.nn.ZeroPad2d((1, 1, 2, 0)).eval(), inp)


@tvm.testing.uses_gpu
def test_forward_constant_pad1d():
    """test_forward_constant_pad1d"""
    inp = torch.rand((1, 2, 4))
    verify_model(torch.nn.ConstantPad1d(2, 3.5).eval(), inp)

    inp = torch.rand((1, 2, 3))
    verify_model(torch.nn.ConstantPad1d((3, 1), 3.5).eval(), inp)


@tvm.testing.uses_gpu
def test_forward_constant_pad2d():
    """test_forward_constant_pad2d"""
    inp = torch.rand((1, 2, 2, 2))
    verify_model(torch.nn.ConstantPad2d(2, 3.5).eval(), inp)
    verify_model(torch.nn.ConstantPad2d((3, 0, 2, 1), 3.5).eval(), inp)


@tvm.testing.uses_gpu
def test_forward_constant_pad3d():
    """test_forward_constant_pad3d"""
    inp = torch.rand((1, 3, 2, 2, 2))
    verify_model(torch.nn.ConstantPad3d(3, 3.5).eval(), inp)
    verify_model(torch.nn.ConstantPad3d((3, 4, 5, 6, 0, 1), 3.5).eval(), inp)


@tvm.testing.uses_gpu
def test_forward_reflection_pad1d():
    """test_forward_reflection_pad1d"""
    inp = torch.rand((1, 2, 4))
    verify_model(torch.nn.ReflectionPad1d(2).eval(), inp)
    verify_model(torch.nn.ReflectionPad1d((3, 1)).eval(), inp)

    inp = torch.rand((2, 4, 5))
    verify_model(torch.nn.ReflectionPad1d((2, 3)).eval(), inp)


@tvm.testing.uses_gpu
def test_forward_reflection_pad2d():
    """test_forward_reflection_pad2d"""
    inp = torch.rand((1, 1, 3, 3))
    verify_model(torch.nn.ReflectionPad2d(2).eval(), inp)
    verify_model(torch.nn.ReflectionPad2d((1, 1, 2, 0)).eval(), inp)

    inp = torch.rand((2, 4, 5, 6))
    verify_model(torch.nn.ReflectionPad2d((1, 3, 2, 4)).eval(), inp)


@tvm.testing.uses_gpu
def test_forward_replication_pad1d():
    """test_forward_replication_pad1d"""
    inp = torch.rand((1, 2, 4))
    verify_model(torch.nn.ReplicationPad1d(2).eval(), inp)
    verify_model(torch.nn.ReplicationPad1d((3, 1)).eval(), inp)

    inp = torch.rand((2, 4, 5))
    verify_model(torch.nn.ReplicationPad1d((2, 3)).eval(), inp)


@tvm.testing.uses_gpu
def test_forward_replication_pad2d():
    """test_forward_replication_pad2d"""
    inp = torch.rand((1, 1, 3, 3))
    verify_model(torch.nn.ReplicationPad2d(2).eval(), inp)
    verify_model(torch.nn.ReplicationPad2d((1, 1, 2, 0)).eval(), inp)

    inp = torch.rand((2, 4, 5, 6))
    verify_model(torch.nn.ReplicationPad2d((1, 3, 2, 4)).eval(), inp)


@tvm.testing.uses_gpu
def test_forward_replication_pad3d():
    """test_forward_replication_pad3d"""
    inp = torch.rand((1, 1, 3, 3, 3))
    verify_model(torch.nn.ReplicationPad3d(3).eval(), inp)
    verify_model(torch.nn.ReplicationPad3d((1, 1, 2, 2, 1, 1)).eval(), inp)

    inp = torch.rand((7, 5, 4, 5, 6))
    verify_model(torch.nn.ReplicationPad3d((2, 3, 2, 5, 1, 4)).eval(), inp)


@tvm.testing.uses_gpu
def test_forward_upsample3d():
    """test_forward_upsample3d"""
    inp = torch.arange(1, 9, dtype=torch.float32).view(1, 1, 2, 2, 2)
    verify_model(torch.nn.Upsample(scale_factor=2, mode="nearest").eval(), inp)
    verify_model(torch.nn.Upsample(scale_factor=2, mode="trilinear").eval(), inp)
    verify_model(
        torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True).eval(), inp
    )


def test_forward_nms():
    """dynamic Non-Maximum Suppression"""
    torch.set_grad_enabled(False)

    class NonMaxSupression(Module):
        def __init__(self, iou_thres):
            super().__init__()
            self.iou_threshold = iou_thres

        def forward(self, *args):
            return torchvision.ops.nms(args[0], args[1], self.iou_threshold)

    # Generate random input data
    def _gen_rand_inputs(num_boxes):
        box_len = 4
        boxes = torch.rand(num_boxes, box_len, dtype=torch.float) * 0.5
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        scores = np.linspace(0, 1, num=num_boxes).astype("float32")
        np.random.shuffle(scores)
        return boxes, torch.from_numpy(scores)

    targets = ["llvm", "cuda"]

    for num_boxes, iou_thres in [(10, 0.3), (100, 0.5), (500, 0.9)]:
        in_boxes, in_scores = _gen_rand_inputs(num_boxes)
        verify_trace_model(NonMaxSupression(iou_thres), [in_boxes, in_scores], targets)


def test_forward_roi_align():
    """ROI align"""
    torch.set_grad_enabled(False)

    class ROIAlign(Module):
        def __init__(self, output_sizes, spatial_scale=1.0, sampling_ratio=-1):
            super().__init__()
            self.spatial_scale = spatial_scale
            self.sampling_ratio = sampling_ratio
            self.output_sizes = output_sizes

        def forward(self, *args):
            return torchvision.ops.roi_align(
                args[0],
                args[1],
                self.output_sizes,
                self.spatial_scale,
                self.sampling_ratio,
            )

    in_data = torch.Tensor(np.random.uniform(size=(1, 8, 100, 100)))
    in_boxes = torch.Tensor(np.random.uniform(0.0, 100.0, size=(35, 4)))
    in_batch = torch.zeros((35, 1), dtype=torch.float)
    in_boxes = torch.cat([in_batch, in_boxes], dim=1)

    verify_model(ROIAlign(7), [in_data, in_boxes])
    verify_model(ROIAlign((10, 10), 0.7, 5), [in_data, in_boxes])
    verify_model(ROIAlign(15, 0.9, 3), [in_data, in_boxes])


@tvm.testing.uses_gpu
def test_conv3d():
    """test_conv3d"""
    for ishape in [(1, 32, 16, 16, 16), (1, 32, 9, 15, 15), (1, 32, 13, 7, 7)]:
        inp = torch.rand(ishape)
        verify_model(torch.nn.Conv3d(32, 16, (3, 3, 3), padding=(1, 1, 1)).eval(), inp)
        verify_model(torch.nn.Conv3d(32, 16, (5, 5, 5), padding=(2, 2, 2)).eval(), inp)
        verify_model(torch.nn.Conv3d(32, 16, kernel_size=1).eval(), inp)
        # downsample
        verify_model(torch.nn.Conv3d(32, 16, kernel_size=1, stride=2).eval(), inp)


@tvm.testing.uses_gpu
def test_conv3d_transpose():
    """test_conv3d_transpose"""
    for ishape in [(1, 8, 10, 5, 10), (1, 8, 5, 8, 8), (1, 8, 13, 7, 7)]:
        inp = torch.rand(ishape)
        verify_model(
            torch.nn.ConvTranspose3d(
                in_channels=8, out_channels=33, kernel_size=3, stride=2
            ).eval(),
            inp,
        )
        verify_model(
            torch.nn.ConvTranspose3d(
                in_channels=8,
                out_channels=20,
                kernel_size=(3, 5, 2),
                stride=(2, 1, 1),
                padding=(0, 4, 2),
            ).eval(),
            inp,
        )
        verify_model(
            torch.nn.ConvTranspose3d(in_channels=8, out_channels=20, kernel_size=1).eval(), inp
        )
        verify_model(
            torch.nn.ConvTranspose3d(in_channels=8, out_channels=5, kernel_size=1, stride=2).eval(),
            inp,
        )


# Model tests
@tvm.testing.uses_gpu
def test_resnet18():
    """test_resnet18"""
    torch.set_grad_enabled(False)
    verify_model("resnet18", atol=1e-4, rtol=1e-4)


@tvm.testing.uses_gpu
def test_squeezenet1_0():
    """test_squeezenet1_0"""
    torch.set_grad_enabled(False)
    verify_model("squeezenet1_0", atol=1e-4, rtol=1e-4)


@tvm.testing.uses_gpu
def test_squeezenet1_1():
    """test_squeezenet1_1"""
    torch.set_grad_enabled(False)
    verify_model("squeezenet1_1", atol=1e-4, rtol=1e-4)


@tvm.testing.uses_gpu
def test_densenet121():
    """test_densenet121"""
    torch.set_grad_enabled(False)
    verify_model("densenet121", atol=1e-4, rtol=1e-4)


@tvm.testing.uses_gpu
def test_inception_v3():
    """test_inception_v3"""
    torch.set_grad_enabled(False)
    verify_model("inception_v3", atol=1e-4, rtol=1e-4)


@tvm.testing.uses_gpu
def test_googlenet():
    """test_googlenet"""
    torch.set_grad_enabled(False)
    verify_model("googlenet", atol=1e-4, rtol=1e-4)


@tvm.testing.uses_gpu
def test_mnasnet0_5():
    """test_mnasnet0_5"""
    torch.set_grad_enabled(False)
    verify_model("mnasnet0_5", atol=1e-4, rtol=1e-4)


@tvm.testing.uses_gpu
def test_mobilenet_v2():
    """test_mobilenet_v2"""
    torch.set_grad_enabled(False)
    verify_model("mobilenet_v2", atol=1e-4, rtol=1e-4)


# pylint: disable=pointless-string-statement
"""
#TODO: Fix VGG and AlexNet issues (probably due to pooling)
@tvm.testing.uses_gpu
def test_alexnet():
    torch.set_grad_enabled(False)
    verify_model("alexnet")

@tvm.testing.uses_gpu
def test_vgg11():
    torch.set_grad_enabled(False)
    verify_model("vgg11")

@tvm.testing.uses_gpu
def test_vgg11_bn():
    torch.set_grad_enabled(False)
    verify_model("vgg11_bn")
"""


@tvm.testing.uses_gpu
def test_custom_conversion_map():
    """test_custom_conversion_map"""

    def get_roi_align():
        pool_size = 5
        n_channels = 2 * (pool_size**2)
        x = torch.rand(2, n_channels, 10, 10)
        rois = torch.tensor(
            [
                [0, 0, 0, 9, 9],  # format is (xyxy)
                [0, 0, 5, 4, 9],
                [0, 5, 5, 9, 9],
                [1, 0, 0, 9, 9],
            ],
            dtype=torch.float,
        )
        roi_align = torchvision.ops.RoIAlign(pool_size, spatial_scale=1, sampling_ratio=-1)
        return roi_align.eval(), [x, rois]

    def convert_roi_align():
        def _impl(inputs, input_types):
            spatial_scale = inputs[2]
            pooled_size = (inputs[3], inputs[4])
            sampling_ratio = inputs[5]
            return relay.op.vision.roi_align(
                inputs[0], inputs[1], pooled_size, spatial_scale, sampling_ratio
            )

        return _impl

    custom_map = {"torchvision::roi_align": convert_roi_align()}
    model, inputs = get_roi_align()

    verify_model(model, inputs, custom_map)


@tvm.testing.uses_gpu
def test_segmentation_models():
    """test_segmentation_models"""

    class SegmentationModelWrapper(Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inp):
            out = self.model(inp)
            return out["out"]

    fcn = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
    deeplab = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)

    inp = [torch.rand((1, 3, 300, 300), dtype=torch.float)]

    verify_model(SegmentationModelWrapper(fcn.eval()), inp, atol=1e-4, rtol=1e-4)
    verify_model(SegmentationModelWrapper(deeplab.eval()), inp, atol=1e-4, rtol=1e-4)


@tvm.testing.uses_gpu
def test_3d_models():
    """test_3d_models"""
    input_shape = (1, 3, 4, 56, 56)
    resnet3d = torchvision.models.video.r3d_18(pretrained=True).eval()
    verify_model(resnet3d, [torch.rand(input_shape)], atol=1e-4, rtol=1e-4)


def _get_default_vm_targets():
    """Get default vm targets"""
    return ["llvm", "cuda"]


def verify_script_model(pt_model, ishapes, targets, idtype=None):
    """verify_script_model"""
    script_module = torch.jit.script(pt_model)

    verify_model_vm(script_module, ishapes, idtype=idtype, targets=targets)


def verify_trace_model(pt_model, idata, targets):
    """verify_trace_model"""
    traced_model = torch.jit.trace(pt_model, idata)
    ishapes = [data.shape for data in idata]
    verify_model_vm(traced_model, ishapes, idata=idata, targets=targets)


def convert_pt_to_tvm_type(idtype):
    """Accepts a pytorch dtype and returns string TVM dtype."""
    # TVM does not support PyTorch complex dtypes
    if idtype == torch.float64:
        curr_dtype = "float64"
    elif idtype == torch.float32:
        curr_dtype = "float32"
    elif idtype == torch.float16:
        curr_dtype = "float16"
    elif idtype == torch.bfloat16:
        curr_dtype = "bfloat16"
    elif idtype == torch.int64:
        curr_dtype = "int64"
    elif idtype == torch.int32:
        curr_dtype = "int32"
    elif idtype == torch.int16:
        curr_dtype = "int16"
    elif idtype == torch.int8:
        curr_dtype = "int8"
    elif idtype == torch.uint8:
        curr_dtype = "uint8"
    elif idtype == torch.bool:
        curr_dtype = "bool"
    else:
        raise NotImplementedError(f"Unsupported dtype: {idtype}")
    return curr_dtype


def verify_model_vm(input_model, ishapes, idtype=None, idata=None, targets=None):
    """verify_model_vm"""
    targets = targets or ["llvm"]
    if not idtype:
        idtype = torch.float

    input_names = [f"i{idx}" for idx, _ in enumerate(ishapes)]
    tvm_dtype = convert_pt_to_tvm_type(idtype)
    input_dtypes = [tvm_dtype] * len(input_names)
    input_shapes = list(zip(input_names, list(zip(ishapes, input_dtypes))))

    if idata:
        input_data = idata
    # If no input_data provided, generate random data of specified dtype
    else:
        if idtype == torch.bool:
            input_data = [
                torch.Tensor.bool(torch.randint(low=0, high=2, size=shape)) for shape in ishapes
            ]
        # Torch dtype can be float, complex, int, or Bool. Complex not supported,
        # so if not float or Bool, dtype must be int!
        elif not idtype.is_floating_point:
            input_data = [
                torch.randint(low=0, high=10, size=shape, dtype=idtype) for shape in ishapes
            ]
        else:
            input_data = [torch.randn(shape, dtype=idtype) for shape in ishapes]

    # Compile via VM
    mod, params = relay.frontend.from_pytorch(input_model, input_shapes)

    for tgt in targets:
        if not tvm.testing.device_enabled(tgt):
            continue
        print("Running on target", tgt)

        dev = tvm.device(tgt, 0)

        evaluator = relay.create_executor("vm", mod=mod, device=dev, target=tgt).evaluate()

        # Inference
        for name, inp in zip(input_names, input_data):
            params[name] = inp.numpy()
        vm_res = evaluator(**params)

        # Baseline result
        with torch.no_grad():
            pt_result = input_model(*input_data)

        # Verify the accuracy
        if isinstance(pt_result, tuple):
            # handle multiple outputs
            for i, pt_result in enumerate(pt_result):
                tvm_res = vm_res[i].numpy()
                tvm.testing.assert_allclose(tvm_res, pt_result.numpy(), rtol=1e-5, atol=1e-5)
        elif not isinstance(pt_result, torch.Tensor):
            tvm_res = vm_res.numpy().item()
            assert pt_result == tvm_res
        else:
            tvm.testing.assert_allclose(vm_res.numpy(), pt_result.numpy(), rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_control_flow():
    """test_control_flow"""

    class SimpleIf(torch.nn.Module):
        """SimpleIf module"""

        def __init__(self, N, M):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(N, M))

        def forward(self, inp):
            if inp.sum() > 0.0:
                output = self.weight + inp
            else:
                output = self.weight - inp
            return output

    class NestedIf(torch.nn.Module):
        """NestedIf module"""

        def __init__(self, N, M):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(N, M))

        def forward(self, inp):
            """forward"""
            if inp.sum() > 0.0:
                if inp.mean() > 0.0:
                    output = self.weight + inp
                else:
                    output = self.weight - inp
            else:
                if inp.mean() >= 0.0:
                    output = self.weight * inp
                else:
                    output = self.weight / inp

            return output

    class ScalarLoop(torch.nn.Module):
        """ScalarLoop module"""

        def forward(self, inp):
            """forward"""
            a = 0
            for i in range(inp.size(0)):
                b = i * i
                b = b + 1
                a += b
            if a != 0:
                a += 1
            else:
                a += 2
            return a

    class SimpleLoop(torch.nn.Module):
        def forward(self, inp):
            a = inp
            for _ in range(inp.size(0)):
                b = a * 2.0
                c = a + b
                a += c
            return a

    class LoopWithIf(torch.nn.Module):
        """LoopWithIf module"""

        def forward(self, inp):
            a = inp
            for _ in range(inp.size(0)):
                b = a * 2.0
                b = a + b
                if b.sum() > 0.0:
                    a += b
                else:
                    a -= b
            return a

    class NestedLoop(torch.nn.Module):
        def forward(self, inp):
            a = inp
            for i in range(inp.size(0)):
                b = a * float(i)
                for j in range(inp.size(1)):
                    a += b * float(j)
            return a

    class SimpleScalarWhileLoop(torch.nn.Module):
        """SimpleScalarWhileLoop module"""

        def forward(self, inp):
            """forward"""
            a = 1
            i = 0
            while i <= inp.size(0):
                a += i
                i += 2
            i = 0
            # also test constant init cond
            while i < 10:
                a += i
                i += 3
            return a

    class SimpleWhileLoop(torch.nn.Module):
        def forward(self, inp):
            a = inp
            i = 0
            while i < inp.size(0):
                a += a * float(i) * 2.0
                i += 1
            return a

    models = [
        SimpleIf(10, 20),
        NestedIf(10, 20),
        ScalarLoop(),
        SimpleLoop(),
        LoopWithIf(),
        SimpleScalarWhileLoop(),
        SimpleWhileLoop(),
        NestedLoop(),
    ]

    for pt_model in models:
        verify_script_model(pt_model.eval(), [(10, 20)], _get_default_vm_targets())


@tvm.testing.uses_gpu
def test_simple_rnn():
    """test_simple_rnn"""
    # The mixed tracing and scripting example from
    # https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#mixing-scripting-and-tracing
    class DecisionGate(torch.nn.Module):
        def forward(self, x):
            if x.sum() > 0:
                return x
            else:
                return -x

    class Cell(torch.nn.Module):
        def __init__(self, dg):
            super().__init__()
            self.dg = dg
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x, h):
            new_h = torch.tanh(self.dg(self.linear(x)) + h)
            return new_h, new_h

    class RNNLoop(torch.nn.Module):
        """Pytorch RNNLoop module"""

        def __init__(self):
            super().__init__()
            x = torch.rand(10, 4, dtype=torch.float)
            h = torch.rand(10, 4, dtype=torch.float)
            self.cell = torch.jit.trace(Cell(DecisionGate()), (x, h))

        def forward(self, xs):
            h = torch.zeros(10, 4, dtype=torch.float)
            y = torch.zeros(10, 4, dtype=torch.float)
            for i in range(xs.size(0)):
                y, h = self.cell(xs[i], h)
            return y

    verify_script_model(RNNLoop().eval(), [(10, 10, 4)], _get_default_vm_targets())


@tvm.testing.uses_gpu
def test_forward_reduce_sum():
    """test_forward_reduce_sum"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class ReduceSum1(Module):
        def forward(self, *args):
            return args[0].sum(1)

    class ReduceSum2(Module):
        def forward(self, *args):
            return args[0].sum(dim=1, keepdim=False)

    class ReduceSum3(Module):
        def forward(self, *args):
            return args[0].sum(dim=2, keepdim=True)

    class ReduceSum4(Module):
        def forward(self, *args):
            return args[0].sum(dim=(2, 3), keepdim=True)

    class ReduceSum5(Module):
        def forward(self, *args):
            return args[0].sum(dim=(2, 3), keepdim=False)

    input_data = torch.rand(input_shape).float()
    verify_model(ReduceSum1().float().eval(), input_data=input_data)
    verify_model(ReduceSum2().float().eval(), input_data=input_data)
    verify_model(ReduceSum3().float().eval(), input_data=input_data)
    verify_model(ReduceSum4().float().eval(), input_data=input_data)
    verify_model(ReduceSum5().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_reduce_prod():
    """test_forward_reduce_prod"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class ReduceProd1(Module):
        def forward(self, *args):
            return args[0].prod(1)

    class ReduceProd2(Module):
        def forward(self, *args):
            return args[0].prod(dim=1, keepdim=False)

    class ReduceProd3(Module):
        def forward(self, *args):
            return args[0].prod(dim=2, keepdim=True)

    input_data = torch.rand(input_shape).float()
    verify_model(ReduceProd1().float().eval(), input_data=input_data)
    verify_model(ReduceProd2().float().eval(), input_data=input_data)
    verify_model(ReduceProd3().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_argmin():
    """test_forward_argmin"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class ArgMin1(Module):
        def forward(self, *args):
            return args[0].argmin(1)

    class ArgMin2(Module):
        def forward(self, *args):
            return args[0].argmin(dim=1, keepdim=False)

    class ArgMin3(Module):
        def forward(self, *args):
            return args[0].argmin(dim=2, keepdim=True)

    input_data = torch.rand(input_shape).float()
    verify_model(ArgMin1().float().eval(), input_data=input_data)
    verify_model(ArgMin2().float().eval(), input_data=input_data)
    verify_model(ArgMin3().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_argmax():
    """test_forward_argmax"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class ArgMax1(Module):
        def forward(self, *args):
            return args[0].argmax(1)

    class ArgMax2(Module):
        def forward(self, *args):
            return args[0].argmax(dim=1, keepdim=False)

    class ArgMax3(Module):
        def forward(self, *args):
            return args[0].argmax(dim=2, keepdim=True)

    input_data = torch.rand(input_shape).float()
    verify_model(ArgMax1().float().eval(), input_data=input_data)
    verify_model(ArgMax2().float().eval(), input_data=input_data)
    verify_model(ArgMax3().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_std():
    """test_forward_std"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Std1(Module):
        def forward(self, *args):
            return args[0].std(1, unbiased=False)

    class Std2(Module):
        def forward(self, *args):
            return args[0].std(dim=1, keepdim=False, unbiased=False)

    class Std3(Module):
        def forward(self, *args):
            return args[0].std(dim=2, keepdim=True, unbiased=False)

    class Std4(Module):
        def forward(self, *args):
            return args[0].std(dim=(2, 3), keepdim=True, unbiased=False)

    class Std5(Module):
        def forward(self, *args):
            return args[0].std(dim=(2, 3), keepdim=False, unbiased=False)

    class Std6(Module):
        def forward(self, *args):
            return args[0].std(unbiased=False)

    class Std7(Module):
        def forward(self, *args):
            return args[0].std(dim=1, keepdim=False, unbiased=True)

    class Std8(Module):
        def forward(self, *args):
            return args[0].std(dim=(2, 3), keepdim=True, unbiased=True)

    class Std9(Module):
        def forward(self, *args):
            return args[0].std(unbiased=True)

    input_data = torch.rand(input_shape).float()
    verify_model(Std1().float().eval(), input_data=input_data)
    verify_model(Std2().float().eval(), input_data=input_data)
    verify_model(Std3().float().eval(), input_data=input_data)
    verify_model(Std4().float().eval(), input_data=input_data)
    verify_model(Std5().float().eval(), input_data=input_data)
    verify_model(Std6().float().eval(), input_data=input_data)
    verify_model(Std7().float().eval(), input_data=input_data)
    verify_model(Std8().float().eval(), input_data=input_data)
    verify_model(Std9().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_var_mean():
    """test_forward_var_mean"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class VarMean1(Module):
        def forward(self, *args):
            return torch.var_mean(args[0], 1, unbiased=False)

    class VarMean2(Module):
        def forward(self, *args):
            return torch.var_mean(args[0], dim=1, keepdim=False, unbiased=False)

    class VarMean3(Module):
        def forward(self, *args):
            return torch.var_mean(args[0], dim=2, keepdim=True, unbiased=False)

    class VarMean4(Module):
        def forward(self, *args):
            return torch.var_mean(args[0], dim=(2, 3), keepdim=True, unbiased=False)

    class VarMean5(Module):
        def forward(self, *args):
            return torch.var_mean(args[0], dim=(2, 3), keepdim=False, unbiased=False)

    class VarMean6(Module):
        def forward(self, *args):
            return torch.var_mean(args[0], unbiased=False)

    class VarMean7(Module):
        def forward(self, *args):
            return torch.var_mean(args[0], dim=1, keepdim=False, unbiased=True)

    class VarMean8(Module):
        def forward(self, *args):
            return torch.var_mean(args[0], dim=(2, 3), keepdim=True, unbiased=True)

    class VarMean9(Module):
        def forward(self, *args):
            return torch.var_mean(args[0], unbiased=True)

    input_data = torch.rand(input_shape).float()
    verify_model(VarMean1().float().eval(), input_data=input_data)
    verify_model(VarMean2().float().eval(), input_data=input_data)
    verify_model(VarMean3().float().eval(), input_data=input_data)
    verify_model(VarMean4().float().eval(), input_data=input_data)
    verify_model(VarMean5().float().eval(), input_data=input_data)
    verify_model(VarMean6().float().eval(), input_data=input_data)
    verify_model(VarMean7().float().eval(), input_data=input_data)
    verify_model(VarMean8().float().eval(), input_data=input_data)
    verify_model(VarMean9().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_variance():
    """test_forward_variance"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Variance1(Module):
        def forward(self, *args):
            return args[0].var(1, unbiased=False)

    class Variance2(Module):
        def forward(self, *args):
            return args[0].var(dim=1, keepdim=False, unbiased=False)

    class Variance3(Module):
        def forward(self, *args):
            return args[0].var(dim=2, keepdim=True, unbiased=False)

    class Variance4(Module):
        def forward(self, *args):
            return args[0].var(dim=(2, 3), keepdim=True, unbiased=False)

    class Variance5(Module):
        def forward(self, *args):
            return args[0].var(dim=(2, 3), keepdim=False, unbiased=False)

    class Variance6(Module):
        def forward(self, *args):
            return args[0].var(unbiased=False)

    class Variance7(Module):
        def forward(self, *args):
            return args[0].var(dim=1, keepdim=False, unbiased=True)

    class Variance8(Module):
        def forward(self, *args):
            return args[0].var(dim=(2, 3), keepdim=True, unbiased=True)

    class Variance9(Module):
        def forward(self, *args):
            return args[0].var(unbiased=True)

    input_data = torch.rand(input_shape).float()
    verify_model(Variance1().float().eval(), input_data=input_data)
    verify_model(Variance2().float().eval(), input_data=input_data)
    verify_model(Variance3().float().eval(), input_data=input_data)
    verify_model(Variance4().float().eval(), input_data=input_data)
    verify_model(Variance5().float().eval(), input_data=input_data)
    verify_model(Variance6().float().eval(), input_data=input_data)
    verify_model(Variance7().float().eval(), input_data=input_data)
    verify_model(Variance8().float().eval(), input_data=input_data)
    verify_model(Variance9().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_rsub():
    """test_forward_rsub"""
    torch.set_grad_enabled(False)

    class Rsub1(Module):
        def forward(self, *args):
            return torch.rsub(args[0], args[1])

    class Rsub2(Module):
        def forward(self, *args):
            return torch.rsub(args[0], args[1], alpha=0.5)

    d1 = torch.rand([1, 3]).float()
    d2 = torch.rand([1, 3]).float()
    d3 = torch.rand([1, 3]).int()
    verify_model(Rsub1().float().eval(), input_data=[d1, d2])
    verify_model(Rsub1().float().eval(), input_data=[d1, d3])
    verify_model(Rsub2().float().eval(), input_data=[d1, d2])
    verify_model(Rsub2().float().eval(), input_data=[d1, d3])

    d1 = torch.rand([1, 3]).half()
    d2 = torch.rand([1, 3]).half()
    verify_model(Rsub1().half().eval(), input_data=[d1, d2])
    verify_model(Rsub1().half().eval(), input_data=[d1, d3])
    verify_model(Rsub2().half().eval(), input_data=[d1, d2])
    verify_model(Rsub2().half().eval(), input_data=[d1, d3])


@tvm.testing.uses_gpu
def test_forward_embedding():
    """test_forward_embedding"""
    torch.set_grad_enabled(False)

    input_data = torch.randint(0, 10, [2, 4]).long()
    verify_model(torch.nn.Embedding(10, 3).float().eval(), input_data=input_data)

    input_data = torch.randint(0, 4, [2, 3, 4]).long()
    verify_model(torch.nn.Embedding(4, 5, sparse=False).float().eval(), input_data=input_data)

    input_data = torch.randint(0, 4, [2, 3, 4]).long()
    verify_model(torch.nn.Embedding(4, 5, sparse=True).float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_onehot():
    """test_forward_onehot"""
    torch.set_grad_enabled(False)

    class OneHot1(Module):
        def forward(self, *args):
            return torch.nn.functional.one_hot(args[0], num_classes=3)

    class OneHot2(Module):
        def forward(self, *args):
            return torch.nn.functional.one_hot(args[0], num_classes=5)

    input_data = torch.arange(0, 5) % 3
    verify_model(OneHot1().float().eval(), input_data=input_data)

    input_data = torch.arange(0, 5) % 4
    verify_model(OneHot2().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_isfinite():
    """test_forward_isfinite"""
    torch.set_grad_enabled(False)

    class IsFinite1(Module):
        def forward(self, *args):
            return torch.isfinite(args[0])

    input_data = torch.tensor([1, float("inf"), 2, float("-inf"), float("nan")]).float()
    verify_model(IsFinite1().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_isnan():
    """test_forward_isnan"""
    torch.set_grad_enabled(False)

    class IsNan1(Module):
        def forward(self, *args):
            return torch.isnan(args[0])

    input_data = torch.tensor([1, float("inf"), 2, float("-inf"), float("nan")]).float()
    verify_model(IsNan1().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_isinf():
    """test_forward_isinf"""
    torch.set_grad_enabled(False)

    class IsInf1(Module):
        def forward(self, *args):
            return torch.isinf(args[0])

    input_data = torch.tensor([1, float("inf"), 2, float("-inf"), float("nan")]).float()
    verify_model(IsInf1().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_clamp():
    """test_forward_clamp"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Clamp1(Module):
        def forward(self, *args):
            return torch.clamp(args[0], min=-0.5, max=0.5)

    class Clamp2(Module):
        def forward(self, *args):
            return torch.clamp(args[0], min=-0.3)

    class Clamp3(Module):
        def forward(self, *args):
            return torch.clamp(args[0], max=1.0)

    class Clamp_MinExpr_MaxConstant(Module):
        def forward(self, *args):
            h, w = args[0].shape[2:]
            amin = h / 100.0
            return torch.clamp(args[0], min=amin, max=w)

    input_data = torch.rand(input_shape).float()
    verify_model(Clamp1().float().eval(), input_data=input_data)
    verify_model(Clamp2().float().eval(), input_data=input_data)
    verify_model(Clamp3().float().eval(), input_data=input_data)
    verify_model(Clamp_MinExpr_MaxConstant().float().eval(), input_data=input_data)

    verify_model(lambda inp: torch.clamp_min(inp, 0.5), input_data)
    inp_uint8 = torch.randint(low=0, high=256, size=(100, 100), dtype=torch.uint8)
    verify_model(lambda inp: torch.clamp_max(inp, 125), inp_uint8)


@tvm.testing.uses_gpu
def test_forward_clamp_():
    """test_forward_clamp_"""
    torch.set_grad_enabled(False)

    class ClampInPlace(Module):
        def __init__(self, i_min, i_max):
            super().__init__()
            self.min = i_min
            self.max = i_max

        def forward(self, *args):
            return torch.clamp_(args[0], self.min, self.max)

    for ishape, i_min, i_max in (([4, 8], 0.1, 0.9), ([7, 6], 0.2, 0.5)):
        input_data = torch.rand(ishape).float()
        verify_model(ClampInPlace(i_min, i_max).float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_ones():
    """test_forward_ones"""
    torch.set_grad_enabled(False)

    class Ones1(Module):
        def forward(self, *args):
            return torch.ones(2, 3)

    verify_model(Ones1().float().eval(), input_data=[])


@tvm.testing.uses_gpu
def test_forward_ones_like():
    """test_forward_ones_like"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class OnesLike1(Module):
        def forward(self, *args):
            return torch.ones_like(args[0])

    class OnesLike2(Module):
        def forward(self, *args):
            return torch.ones_like(args[0], dtype=torch.int8)

    class OnesLike3(Module):
        def forward(self, *args):
            return torch.ones_like(args[0], dtype=torch.float)

    input_data = torch.rand(input_shape).float()
    verify_model(OnesLike1().float().eval(), input_data=input_data)
    verify_model(OnesLike2().float().eval(), input_data=input_data)
    verify_model(OnesLike3().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_new_ones():
    """test_forward_new_ones"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    def test_func(input_tensor):
        return input_tensor.new_ones([3, 10, 10])

    verify_model_with_input(test_func, [torch.rand(input_shape).float()])


@tvm.testing.uses_gpu
def test_forward_zeros():
    """test_forward_zeros"""
    torch.set_grad_enabled(False)

    class Zeros1(Module):
        def forward(self, *args):
            return torch.zeros(2, 3)

    verify_model(Zeros1().float().eval(), input_data=[])


def test_forward_zero_():
    def test_func(x):
        return x.zero_()

    verify_model_with_input(test_func, [torch.rand([1, 3, 10, 10]).float()])


@tvm.testing.uses_gpu
def test_forward_zeros_like():
    """test_forward_zeros_like"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class ZerosLike1(Module):
        def forward(self, *args):
            return torch.zeros_like(args[0])

    class ZerosLike2(Module):
        def forward(self, *args):
            return torch.zeros_like(args[0], dtype=torch.int32)

    class ZerosLike3(Module):
        def forward(self, *args):
            return torch.zeros_like(args[0], dtype=torch.float)

    input_data = torch.rand(input_shape).float()
    verify_model(ZerosLike1().float().eval(), input_data=input_data)
    verify_model(ZerosLike2().float().eval(), input_data=input_data)
    verify_model(ZerosLike3().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_full():
    """test_forward_full"""
    torch.set_grad_enabled(False)

    class Full1(Module):
        def forward(self, *args):
            return torch.full((2, 3), 3.14)

    class Full2(Module):
        def forward(self, *args):
            return torch.full((1, 2, 3), 1.0, dtype=torch.int32)

    verify_model(Full1().float().eval(), input_data=[])
    verify_model(Full2().float().eval(), input_data=[])


@tvm.testing.uses_gpu
def test_forward_full_like():
    """test_forward_full_like"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class FullLike1(Module):
        def forward(self, *args):
            return torch.full_like(args[0], 3.14)

    class FullLike2(Module):
        def forward(self, *args):
            return torch.full_like(args[0], 22.22, dtype=torch.int32)

    class FullLike3(Module):
        def forward(self, *args):
            return torch.full_like(args[0], 1.4, dtype=torch.float)

    input_data = torch.rand(input_shape).float()
    verify_model(FullLike1().float().eval(), input_data=input_data)
    verify_model(FullLike2().float().eval(), input_data=input_data)
    verify_model(FullLike3().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_new_full():
    """test_forward_new_full"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    def test_func(input_tensor):
        return input_tensor.new_full([2, 3], 1)

    verify_model_with_input(test_func, [torch.rand(input_shape).float()])


def test_forward_fill_():
    def test_func(x):
        return x.fill_(3)

    verify_model_with_input(test_func, [torch.rand([1, 3, 10, 10]).float()])


def test_forward_fill_with_div():
    """test_forward_fill_with_div"""

    def test_func(x):
        y = torch.div(torch.tensor(6.0), torch.tensor(2.0))
        return x.fill_(y)

    verify_model_with_input(test_func, [torch.rand([1, 3, 10, 10]).float()])


@tvm.testing.uses_gpu
def test_forward_linspace():
    """test_forward_linspace"""
    torch.set_grad_enabled(False)

    class Linspace1(Module):
        def forward(self, *args):
            return torch.linspace(5, 10, steps=100)

    class Linspace2(Module):
        def forward(self, *args):
            return torch.linspace(-10, 10, steps=5)

    class Linspace3(Module):
        def forward(self, *args):
            return torch.linspace(start=-10, end=10, steps=5)

    class Linspace4(Module):
        def forward(self, *args):
            return torch.linspace(start=-10, end=10, steps=1)

    class Linspace5(Module):
        def forward(self, *args):
            return torch.linspace(1, 2, 1, dtype=torch.int32)

    class Linspace6(Module):
        def forward(self, *args):
            return torch.linspace(start=1, end=6, steps=2)

    class Linspace7(Module):
        def forward(self, *args):
            return torch.linspace(1, 4, steps=100, dtype=torch.float32)

    class Linspace8(Module):
        def forward(self, *args):
            return torch.linspace(1, 2, 1, dtype=torch.int16)

    verify_model(Linspace1().float().eval())
    verify_model(Linspace2().float().eval())
    verify_model(Linspace3().float().eval())
    verify_model(Linspace4().float().eval())
    verify_model(Linspace5().float().eval())
    verify_model(Linspace6().float().eval())
    verify_model(Linspace7().float().eval())
    verify_model(Linspace8().float().eval())


@tvm.testing.uses_gpu
def test_forward_take():
    """test_forward_take"""
    torch.set_grad_enabled(False)

    class Take1(Module):
        def forward(self, *args):
            indices = torch.tensor([[0, 0], [1, 0]])
            if torch.cuda.is_available():
                indices = indices.cuda()
            return torch.take(args[0], indices)

    class Take2(Module):
        def forward(self, *args):
            return torch.take(args[0], args[1])

    input_data = torch.tensor([[1, 2], [3, 4]])
    verify_model(Take1().float().eval(), input_data=input_data)
    indices = torch.tensor([[0, 0], [1, 0]])
    verify_model(Take2().float().eval(), input_data=[input_data, indices])
    indices = torch.tensor([0, -1])
    verify_model(Take2().float().eval(), input_data=[input_data, indices])


@tvm.testing.uses_gpu
def test_forward_topk():
    """test_forward_topk"""
    torch.set_grad_enabled(False)

    class Topk1(Module):
        def forward(self, *args):
            return torch.topk(args[0], k=3)

    class Topk2(Module):
        def forward(self, *args):
            return torch.topk(args[0], k=3, dim=-2)

    class Topk3(Module):
        def forward(self, *args):
            return torch.topk(args[0], k=3, dim=3)

    class Topk4(Module):
        def forward(self, *args):
            return torch.topk(args[0], k=3, largest=True)

    class Topk5(Module):
        def forward(self, *args):
            return torch.topk(args[0], k=3, largest=False)

    class Topk6(Module):
        def forward(self, *args):
            return torch.topk(args[0], k=3, sorted=True)

    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(Topk1().float().eval(), input_data=input_data)
    verify_model(Topk2().float().eval(), input_data=input_data)
    verify_model(Topk3().float().eval(), input_data=input_data)
    verify_model(Topk4().float().eval(), input_data=input_data)
    verify_model(Topk5().float().eval(), input_data=input_data)
    verify_model(Topk6().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_logical_not():
    """test_forward_logical_not"""
    torch.set_grad_enabled(False)

    class LogicalNot1(Module):
        def forward(self, *args):
            return torch.logical_not(args[0])

    input_data = torch.tensor([True, False])
    verify_model(LogicalNot1().float().eval(), input_data=input_data)

    input_data = torch.tensor([0, 1, -10], dtype=torch.int8)
    verify_model(LogicalNot1().float().eval(), input_data=input_data)

    input_data = torch.tensor([0.0, 1.5, -10.0], dtype=torch.double)
    verify_model(LogicalNot1().float().eval(), input_data=input_data)

    input_data = torch.tensor([0.0, 1.0, -10.0], dtype=torch.int32)
    verify_model(LogicalNot1().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_bitwise_not():
    """test_forward_bitwise_not"""
    torch.set_grad_enabled(False)

    class BitwiseNot1(Module):
        def forward(self, *args):
            return torch.bitwise_not(args[0])

    input_data = torch.tensor([0, 1, -10], dtype=torch.int8)
    verify_model(BitwiseNot1().float().eval(), input_data=input_data)

    input_data = torch.tensor([0.0, 1.0, -10.0], dtype=torch.int32)
    verify_model(BitwiseNot1().float().eval(), input_data=input_data)

    input_data = torch.tensor([True, False])
    verify_model(BitwiseNot1().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_bitwise_xor():
    """test_forward_bitwise_xor"""
    torch.set_grad_enabled(False)

    class BitwiseXor1(Module):
        def forward(self, *args):
            return torch.bitwise_xor(args[0], args[1])

    class BitwiseXor2(Module):
        def forward(self, *args):
            rhs = torch.tensor([1, 0, 3], dtype=torch.int8)
            if torch.cuda.is_available():
                rhs = rhs.cuda()
            return torch.bitwise_xor(args[0], rhs)

    lhs = torch.tensor([-1, -2, 3], dtype=torch.int8)
    rhs = torch.tensor([1, 0, 3], dtype=torch.int8)
    verify_model(BitwiseXor1().float().eval(), input_data=[lhs, rhs])

    lhs = torch.tensor([True, True, False])
    rhs = torch.tensor([False, True, False])
    verify_model(BitwiseXor1().float().eval(), input_data=[lhs, rhs])

    lhs = torch.tensor([-1, -2, 3], dtype=torch.int8)
    verify_model(BitwiseXor2().float().eval(), input_data=[lhs])


@tvm.testing.uses_gpu
def test_forward_logical_xor():
    """test_forward_logical_xor"""
    torch.set_grad_enabled(False)

    class LogicalXor1(Module):
        def forward(self, *args):
            return torch.logical_xor(args[0], args[1])

    class LogicalXor2(Module):
        def forward(self, *args):
            rhs = torch.tensor([1, 0, 3], dtype=torch.int8)
            if torch.cuda.is_available():
                rhs = rhs.cuda()
            return torch.logical_xor(args[0], rhs)

    lhs = torch.tensor([-1, -2, 3], dtype=torch.int8)
    rhs = torch.tensor([1, 0, 3], dtype=torch.int8)
    verify_model(LogicalXor1().float().eval(), input_data=[lhs, rhs])

    lhs = torch.tensor([True, True, False])
    rhs = torch.tensor([False, True, False])
    verify_model(LogicalXor1().float().eval(), input_data=[lhs, rhs])

    lhs = torch.tensor([-1, -2, 3], dtype=torch.int8)
    verify_model(LogicalXor2().float().eval(), input_data=[lhs])


@tvm.testing.uses_gpu
def test_forward_unary():
    """test_forward_unary"""
    torch.set_grad_enabled(False)

    class Sqrt1(Module):
        def forward(self, *args):
            return torch.sqrt(args[0])

    class RSqrt1(Module):
        def forward(self, *args):
            return torch.rsqrt(args[0])

    class Ceil1(Module):
        def forward(self, *args):
            return torch.ceil(args[0])

    class Floor1(Module):
        def forward(self, *args):
            return torch.floor(args[0])

    class Round1(Module):
        def forward(self, *args):
            return torch.round(args[0])

    class Cos1(Module):
        def forward(self, *args):
            return torch.cos(args[0])

    class Sin1(Module):
        def forward(self, *args):
            return torch.sin(args[0])

    class Tan1(Module):
        def forward(self, *args):
            return torch.tan(args[0])

    class Tanh1(Module):
        def forward(self, *args):
            return torch.tanh(args[0])

    class Acos1(Module):
        def forward(self, *args):
            return torch.acos(args[0])

    class Asin1(Module):
        def forward(self, *args):
            return torch.asin(args[0])

    class Atan1(Module):
        def forward(self, *args):
            return torch.atan(args[0])

    class Log1(Module):
        def forward(self, *args):
            return torch.log(args[0])

    class Exp1(Module):
        def forward(self, *args):
            return torch.exp(args[0])

    class Erf1(Module):
        def forward(self, *args):
            return torch.erf(args[0])

    class Trunc1(Module):
        def forward(self, *args):
            return torch.trunc(args[0])

    class Sign1(Module):
        def forward(self, *args):
            return torch.sign(args[0])

    class Neg1(Module):
        def forward(self, *args):
            return torch.neg(args[0])

    class Sinh1(Module):
        def forward(self, *args):
            return torch.sinh(args[0])

    class Cosh1(Module):
        def forward(self, *args):
            return torch.cosh(args[0])

    class Log2_1(Module):
        def forward(self, *args):
            return torch.log2(args[0])

    class Log10_1(Module):
        def forward(self, *args):
            return torch.log10(args[0])

    class Log1p_1(Module):
        def forward(self, *args):
            return torch.log1p(args[0])

    class Square(Module):
        def forward(self, *args):
            return torch.square(args[0])

    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(Square().float().eval(), input_data=input_data)
    verify_model(Sqrt1().float().eval(), input_data=input_data)
    verify_model(RSqrt1().float().eval(), input_data=input_data)
    verify_model(Ceil1().float().eval(), input_data=input_data)
    verify_model(Floor1().float().eval(), input_data=input_data)
    verify_model(Round1().float().eval(), input_data=input_data)
    verify_model(Cos1().float().eval(), input_data=input_data)
    verify_model(Cosh1().float().eval(), input_data=input_data)
    verify_model(Sin1().float().eval(), input_data=input_data)
    verify_model(Sinh1().float().eval(), input_data=input_data)
    verify_model(Tan1().float().eval(), input_data=input_data)
    verify_model(Tanh1().float().eval(), input_data=input_data)
    verify_model(Acos1().float().eval(), input_data=input_data)
    verify_model(Asin1().float().eval(), input_data=input_data)
    verify_model(Atan1().float().eval(), input_data=input_data)
    verify_model(Log1().float().eval(), input_data=input_data)
    verify_model(Log2_1().float().eval(), input_data=input_data)
    verify_model(Log10_1().float().eval(), input_data=input_data)
    verify_model(Log1p_1().float().eval(), input_data=input_data)
    verify_model(Exp1().float().eval(), input_data=input_data)
    verify_model(Erf1().float().eval(), input_data=input_data)
    verify_model(Trunc1().float().eval(), input_data=input_data)
    verify_model(Sign1().float().eval(), input_data=input_data)
    verify_model(Neg1().float().eval(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_tril():
    """test_forward_tril"""
    torch.set_grad_enabled(False)

    def test_func(input_data):
        return torch.tril(input_data)

    input_data = torch.rand([3, 3]).float()
    verify_model(test_func, input_data=input_data)
    input_data = torch.rand([1, 3, 10, 10]).float()
    verify_model(test_func, input_data=input_data)

    def test_func1(input_data):
        return torch.tril(input_data, 1)

    input_data = torch.rand([3, 3]).float()
    verify_model(test_func1, input_data=input_data)
    input_data = torch.rand([1, 3, 10, 10]).float()
    verify_model(test_func1, input_data=input_data)

    def test_func2(input_data):
        return torch.tril(input_data, -1)

    input_data = torch.rand([3, 3]).float()
    verify_model(test_func2, input_data=input_data)
    input_data = torch.rand([1, 3, 10, 10]).float()
    verify_model(test_func2, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_triu():
    """test_forward_triu"""
    torch.set_grad_enabled(False)

    def test_func(input_data):
        return torch.triu(input_data)

    input_data = torch.rand([3, 3]).float()
    verify_model(test_func, input_data=input_data)
    input_data = torch.rand([1, 3, 10, 10]).float()
    verify_model(test_func, input_data=input_data)

    def test_func1(input_data):
        return torch.triu(input_data, 1)

    input_data = torch.rand([3, 3]).float()
    verify_model(test_func1, input_data=input_data)
    input_data = torch.rand([1, 3, 10, 10]).float()
    verify_model(test_func1, input_data=input_data)

    def test_func2(input_data):
        return torch.triu(input_data, -1)

    input_data = torch.rand([3, 3]).float()
    verify_model(test_func2, input_data=input_data)
    input_data = torch.rand([1, 3, 10, 10]).float()
    verify_model(test_func2, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_where():
    """test_forward_where"""
    torch.set_grad_enabled(False)

    class Where1(Module):
        def forward(self, *args):
            y = torch.ones([3, 2])
            if torch.cuda.is_available():
                y = y.cuda()
            return torch.where(args[0] > 0, args[0], y)

    class Where2(Module):
        def forward(self, *args):
            return torch.where(args[0] > 0, args[0], args[1])

    class Where3(Module):
        def forward(self, *args):
            return torch.where(args[0])[0]

    x = torch.rand([3, 2]).float()
    verify_model(Where1(), input_data=[x])
    y = torch.rand([3, 2])
    verify_model(Where2(), input_data=[x, y])

    # a single argument variant, equivalent to torch.nonzero(..., as_tuple=True)
    inp = torch.rand([10])
    inp[3:8] = 0
    verify_trace_model(Where3(), [inp], ["llvm"])


@tvm.testing.uses_gpu
def test_forward_addcdiv():
    """test_forward_addcdiv"""
    torch.set_grad_enabled(False)

    class Addcdiv1(Module):
        def forward(self, *args):
            t1 = torch.ones([3, 1])
            t2 = torch.ones([1, 3])
            if torch.cuda.is_available():
                t1 = t1.cuda()
                t2 = t2.cuda()
            return torch.addcdiv(args[0], 0.1, t1, t2)

    class Addcdiv2(Module):
        def forward(self, *args):
            return torch.addcdiv(args[0], 0.5, args[1], args[2])

    input_data = torch.rand([1, 3]).float()
    verify_model(Addcdiv1().float().eval(), input_data=input_data)
    t1 = torch.rand([3, 1]).float()
    t2 = torch.rand([1, 3]).float()
    verify_model(Addcdiv2().float().eval(), input_data=[input_data, t1, t2])


@tvm.testing.uses_gpu
def test_forward_addcmul():
    """test_forward_addcmul"""
    torch.set_grad_enabled(False)

    class Addcmul1(Module):
        def forward(self, *args):
            t1 = torch.ones([3, 1])
            t2 = torch.ones([1, 3])
            if torch.cuda.is_available():
                t1 = t1.cuda()
                t2 = t2.cuda()
            return torch.addcmul(args[0], 0.1, t1, t2)

    class Addcmul2(Module):
        def forward(self, *args):
            return torch.addcmul(args[0], 0.5, args[1], args[2])

    input_data = torch.rand([1, 3]).float()
    verify_model(Addcmul1().float().eval(), input_data=input_data)
    t1 = torch.rand([3, 1]).float()
    t2 = torch.rand([1, 3]).float()
    verify_model(Addcmul2().float().eval(), input_data=[input_data, t1, t2])


@tvm.testing.uses_gpu
def test_forward_true_divide():
    """test_forward_true_divide"""
    if package_version.parse(torch.__version__) < package_version.parse("1.5.0"):
        return
    torch.set_grad_enabled(False)

    class TrueDivide(Module):
        def forward(self, *args):
            return torch.true_divide(args[0], args[1])

    dividend = torch.rand([5, 3]).float()
    # divisor could be either tensor or scalar
    divisor_tensor = torch.rand([5, 3]).float() + 0.5
    divisor_scalar = torch.tensor(1.0, dtype=torch.float32)
    verify_model(
        TrueDivide().float().eval(), input_data=[dividend, divisor_tensor], atol=1e-4, rtol=1e-4
    )
    verify_model(
        TrueDivide().float().eval(), input_data=[dividend, divisor_scalar], atol=1e-4, rtol=1e-4
    )


@tvm.testing.uses_gpu
def test_forward_is_floating_point():
    """test_forward_is_floating_point"""
    torch.set_grad_enabled(False)

    class IsFloatingPoint(Module):
        def forward(self, arg):
            # `torch.jit.trace` cannot accept something that outputs
            # a Bool, so `torch.jit.script` will be used instead
            return torch.is_floating_point(arg)

    targets = _get_default_vm_targets()
    verify_script_model(IsFloatingPoint(), [(1, 1)], targets, idtype=torch.float64)
    verify_script_model(IsFloatingPoint(), [(1, 1)], targets, idtype=torch.float32)
    verify_script_model(IsFloatingPoint(), [(1, 1)], targets, idtype=torch.float16)
    # todo(dvisnty): Run the test for bfloat16 when full bfloat16 support is implemented
    # verify_script_model(IsFloatingPoint(), [(1,1)], targets, idtype=torch.bfloat16)
    verify_script_model(IsFloatingPoint(), [(1, 1)], targets, idtype=torch.int64)
    verify_script_model(IsFloatingPoint(), [(1, 1)], targets, idtype=torch.int32)
    verify_script_model(IsFloatingPoint(), [(1, 1)], targets, idtype=torch.int16)
    verify_script_model(IsFloatingPoint(), [(1, 1)], targets, idtype=torch.int8)
    verify_script_model(IsFloatingPoint(), [(1, 1)], targets, idtype=torch.uint8)


@tvm.testing.uses_gpu
def test_forward_traced_function():
    """test_forward_traced_function"""

    def fn(t1, t2):
        return t1 + t2

    tensor1 = torch.randn(3, 4)
    tensor2 = torch.randn(3, 4)
    verify_model(fn, input_data=[tensor1, tensor2])


@tvm.testing.uses_gpu
def test_forward_dtypes():
    """test_forward_dtypes"""

    def fn(t1, t2):
        return 2.5 * t1 + t2

    for dt in [torch.int32, torch.int64, torch.double]:
        tensor1 = torch.randn(3, 4).to(dtype=dt)
        tensor2 = torch.randn(3, 4).to(dtype=dt)
        verify_model(fn, input_data=[tensor1, tensor2])

    class ModuleWithIntParameters(Module):
        def __init__(self, arr):
            super().__init__()
            self.param = torch.nn.Parameter(torch.LongTensor(arr), requires_grad=False)

        def forward(self, x):
            return x.long() + self.param

    shape = (10, 10)
    param = torch.ones(shape, dtype=torch.long)
    inp = torch.ones(shape, dtype=torch.int)
    verify_model(ModuleWithIntParameters(param), input_data=inp)


@tvm.testing.uses_gpu
def test_weight_names():
    tm = torch.jit.trace(torch.nn.Linear(3, 4), [torch.randn(2, 3)])
    _, params = relay.frontend.from_pytorch(tm, [("input", (2, 3))])
    assert set(params.keys()) == set(n for n, _ in tm.named_parameters())


@tvm.testing.uses_gpu
def test_duplicate_weight_use():
    """test_duplicate_weight_use"""
    # The test cases doesn't make any sense as a neural network,
    # the issue popped up in shared input/output embeddings of bert,
    # but this is quicker
    class Test(Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(5, 3)

        def forward(self, x):
            x = self.lin(x)
            x = x @ self.lin.weight
            return x

    verify_model(Test(), input_data=[torch.randn(5, 5)])


@tvm.testing.uses_gpu
def test_forward_matmul():
    """test_forward_matmul"""
    torch.set_grad_enabled(False)

    class MatMul1(Module):
        def forward(self, *args):
            return torch.matmul(args[0], args[1])

    # matrix x vector
    tensor1 = torch.randn(3, 4)
    tensor2 = torch.randn(4)
    verify_model(MatMul1().float().eval(), input_data=[tensor1, tensor2])

    # vector x matrix
    tensor1 = torch.randn(4)
    tensor2 = torch.randn(4, 3)
    verify_model(MatMul1().float().eval(), input_data=[tensor1, tensor2])

    # matrix x matrix
    tensor1 = torch.randn(10, 4)
    tensor2 = torch.randn(4, 10)
    verify_model(MatMul1().float().eval(), input_data=[tensor1, tensor2], expected_ops=["nn.dense"])

    # batched matrix x batched matrix
    tensor1 = torch.randn(10, 3, 4)
    tensor2 = torch.randn(10, 4, 5)
    verify_model(
        MatMul1().float().eval(), input_data=[tensor1, tensor2], expected_ops=["nn.batch_matmul"]
    )

    # batched matrix x broadcasted matrix
    tensor1 = torch.randn(10, 3, 4)
    tensor2 = torch.randn(4, 5)
    verify_model(MatMul1().float().eval(), input_data=[tensor1, tensor2], expected_ops=["nn.dense"])

    # broadcasted matrix x batched matrix
    tensor1 = torch.randn(10, 4)
    tensor2 = torch.randn(3, 4, 5)
    verify_model(MatMul1().float().eval(), input_data=[tensor1, tensor2], expected_ops=["nn.dense"])

    # batched matrix x batched matrix
    tensor1 = torch.randn(1, 12, 14, 64)
    tensor2 = torch.randn(1, 12, 64, 14)
    verify_model(MatMul1().float().eval(), input_data=[tensor1, tensor2])


def test_forward_index():
    """test_forward_index"""
    torch.set_grad_enabled(False)
    input_shape = [3, 4, 5, 6]

    class Index0(Module):
        def forward(self, x):
            return x[[0, 1], [0, 2], :2, 4]

    input_data = torch.rand(input_shape).float()
    verify_model(Index0().eval(), input_data=input_data)

    class Index1(Module):
        def forward(self, x):
            return x[[0], [1, 2, 3, 0], [3, 1, 2, 2], [4, 2, 1, 0]]

    input_data = torch.rand(input_shape).float()
    verify_model(Index1().eval(), input_data=input_data)

    def test_fn_bool_mask():
        return lambda data, mask: data[0, mask]

    data = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.tensor([True, True, False])

    verify_trace_model(test_fn_bool_mask(), [data, mask], ["llvm", "cuda"])


def test_logsumexp():
    """test_logsumexp"""

    class Logsumexp(Module):
        def __init__(self, dim, keepdim=False):
            super().__init__()
            self.dim = dim
            self.keepdim = keepdim

        def forward(self, x):
            return torch.logsumexp(x, self.dim, self.keepdim)

    input_shape = (100, 100)
    input_data = torch.rand(input_shape)

    verify_model(Logsumexp(0), input_data=input_data)
    verify_model(Logsumexp(0, keepdim=True), input_data=input_data)
    # Also test on double
    verify_model(Logsumexp(1, keepdim=True), input_data=input_data.double())


def test_stack():
    """test_stack"""

    class Stack(torch.nn.Module):
        def __init__(self, axis=0):
            super().__init__()
            self.axis = axis

        def forward(self, x):
            return torch.stack((x, x), dim=self.axis)

    inp = torch.randn(8, 8, 8)
    verify_model(Stack(), input_data=inp)
    verify_model(Stack(axis=-1), input_data=inp)
    verify_model(Stack(axis=3), input_data=inp)
    verify_model(Stack(axis=-4), input_data=inp)


def test_stack_dynamic():
    """test_stack_dynamic"""

    class Stack(torch.nn.Module):
        def forward(self, x):
            tensor_list = []
            for i in range(x.size(0)):
                # this is a workaround to avoid generating impure aten::append op
                tensor_list += [x[i]]
            # relay tensor array only supports stacking on the first axis
            return torch.stack(tensor_list, dim=0)

    verify_script_model(Stack(), [(8, 8, 8)], _get_default_vm_targets())


def test_forward_unbind():
    """test_forward_unbind"""

    class Unbind(torch.nn.Module):
        def __init__(self, axis=0):
            super().__init__()
            self.axis = axis

        def forward(self, x):
            return torch.unbind(x, self.axis)

    inp = torch.randn(8, 8, 8)
    verify_model(Unbind(0), input_data=inp)
    verify_model(Unbind(1), input_data=inp)
    verify_model(Unbind(2), input_data=inp)


def test_forward_nonzero():
    """test_forward_nonzero"""

    class Nonzero(Module):
        def __init__(self, as_tuple=False):
            super().__init__()
            self.as_tuple = as_tuple

        def forward(self, data):
            return torch.nonzero(data, as_tuple=self.as_tuple)

    inp = torch.Tensor(np.array([[0, 1, 0], [2, 0, 9], [-1, -1, 0]]).astype("float32"))
    verify_trace_model(Nonzero(), [inp], ["llvm"])


def test_forward_scatter():
    """test_forward_scatter"""
    # integer cannot be traced
    def test_fn_scatter(dim):
        return lambda data, index, src: torch.scatter(data, dim=dim, index=index, src=src)

    def test_fn_scatter_add(dim):
        return lambda data, index, src: torch.scatter_add(data, dim=dim, index=index, src=src)

    in_data = torch.zeros(3, 5)
    in_index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]])
    in_src = torch.rand(2, 5)

    targets = ["llvm", "cuda"]
    verify_trace_model(test_fn_scatter(0), [in_data, in_index, in_src], targets)
    verify_trace_model(test_fn_scatter_add(0), [in_data, in_index, in_src], targets)

    in_data = torch.zeros(2, 4)
    in_index = torch.tensor([[2], [3]])
    in_src = torch.rand(2, 1)

    verify_trace_model(test_fn_scatter(1), [in_data, in_index, in_src], targets)
    verify_trace_model(test_fn_scatter_add(1), [in_data, in_index, in_src], targets)


def test_forward_index_put():
    """test_forward_index_put"""
    # torch.index_put for 2D tensor and default accumulate (False)
    def test_fn_index_put2():
        return lambda data, xidx, yidx, values: torch.index_put(
            data, indices=[xidx, yidx], values=values
        )

    # torch.index_put for 3D tensor and accumulate=True
    def test_fn_index_put3a():
        return lambda data, xidx, yidx, zidx, values: torch.index_put(
            data, indices=[xidx, yidx, zidx], values=values, accumulate=True
        )

    shape = (3, 5)
    in_data = torch.zeros(shape)
    xidx = torch.tensor([0, 1, 2, 2])
    yidx = torch.tensor([0, 1, 3, 4])
    values = torch.tensor([2.0, 4.0, 7.0, 9.0])

    targets = ["llvm", "cuda"]
    verify_trace_model(test_fn_index_put2(), [in_data, xidx, yidx, values], targets)

    shape = (3, 5, 3)
    in_data = torch.zeros(shape)
    xidx = torch.tensor([0, 1, 2, 2, 0])
    yidx = torch.tensor([0, 1, 3, 4, 0])
    zidx = torch.tensor([0, 1, 1, 2, 0])
    values = torch.tensor([2.0, 4.0, 7.0, 9.0, 1.0])

    verify_trace_model(test_fn_index_put3a(), [in_data, xidx, yidx, zidx, values], targets)


def test_numel():
    """test_numel"""

    class Numel(Module):
        def forward(self, data):
            return torch.tensor(torch.numel(data))

    targets = _get_default_vm_targets()
    verify_script_model(Numel(), [(1,)], targets)
    verify_script_model(Numel(), [(3, 5)], targets)
    verify_script_model(Numel(), [(3, 5, 8)], targets)


def test_empty():
    """Test for aten::empty"""

    def test_func():
        return torch.empty([1, 3, 10, 10])

    verify_model_with_input(test_func, [], assert_shape_only=True)


def test_empty_like():
    """Test for aten::empty_like"""

    def test_func(data):
        return torch.empty_like(data)

    verify_model_with_input(test_func, [torch.rand([1, 3, 10, 10]).float()], assert_shape_only=True)


@tvm.testing.uses_gpu
def test_new_empty():
    """test_forward_new_ones"""
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    def test_func(input_tensor):
        return input_tensor.new_empty([3, 10, 10])

    verify_model_with_input(test_func, [torch.rand(input_shape).float()], assert_shape_only=True)

    def test_func1(input_tensor):
        return input_tensor.new_empty([3, 10, 10], dtype=torch.int32)

    verify_model_with_input(test_func1, [torch.rand(input_shape).float()], assert_shape_only=True)


def test_randn():
    """Test for aten::randn"""

    def test_func():
        return torch.randn([1, 3, 10, 10])

    verify_model_with_input(test_func, [], assert_shape_only=True)

    def test_func1():
        return torch.randn(1, 3, 10, 10)

    verify_model_with_input(test_func1, [], assert_shape_only=True)


def test_forward_pretrained_bert_base_uncased():
    ######################################################################
    # This is an example how to run BERT models using TVM
    # ---------------------------------------------------
    """
    Refer the bert example given in https://pypi.org/project/pytorch-pretrained-bert

    # To get started, pretrained bert package needs to be installed as prerequisite.

    .. code-block:: bash

        # install bert package
        pip install pytorch_pretrained_bert==0.6.2 --user
    """
    # pylint: disable=import-outside-toplevel
    try:
        from pytorch_pretrained_bert import BertForMaskedLM, BertTokenizer
    except ImportError:
        print("Torch pretrained bert package must be installed to run this script.")
        return

    ######################################################################
    # Load the tokenizer and tokenize the input
    # -----------------------------------------

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenized input
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = tokenizer.tokenize(text)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = 8
    tokenized_text[masked_index] = "[MASK]"
    assert tokenized_text == [
        "[CLS]",
        "who",
        "was",
        "jim",
        "henson",
        "?",
        "[SEP]",
        "jim",
        "[MASK]",
        "was",
        "a",
        "puppet",
        "##eer",
        "[SEP]",
    ]

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    ######################################################################
    # Load a pretrained PyTorch model bert-base-uncased
    # -------------------------------------------------

    # Bert Model with a language modeling
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.eval()

    ######################################################################
    # Predict all tokens with pytorch
    # -------------------------------

    with torch.no_grad():
        torch_preds = model(tokens_tensor, segments_tensors)

    ######################################################################
    # Make TorchScripted model via jit trace
    # --------------------------------------

    scripted_model = torch.jit.trace(model, (tokens_tensor, segments_tensors)).eval()

    ######################################################################
    # Import the graph to Relay
    # -------------------------
    # Convert PyTorch graph to Relay graph. The input name can be arbitrary.

    input_1 = "input_ids"
    input_2 = "input.2"
    shape_list = [(input_1, list(tokens_tensor.shape)), (input_2, list(segments_tensors.shape))]

    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    ######################################################################
    # Compile the model with relay
    # ----------------------------

    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        relay_graph, relay_lib, relay_params = relay.build(mod, target=target, params=params)

    ######################################################################
    # Execute on TVM
    # --------------

    dev = tvm.device(target, 0)
    relay_model = graph_executor.create(relay_graph, relay_lib, dev)
    relay_model.set_input(**relay_params)
    relay_model.set_input(input_1, tokens_tensor)
    relay_model.set_input(input_2, segments_tensors)
    relay_model.run()
    compiled_output = relay_model.get_output(0).numpy()

    ######################################################################
    # Validate the outputs
    # --------------------
    # Compare the torch and tvm outputs

    tvm.testing.assert_allclose(torch_preds, compiled_output, rtol=1e-3, atol=1e-3)

    ######################################################################
    # Process the output
    # ------------------
    # Process the model output to token.

    # Torch output to token
    torch_pred_idx = torch.argmax(torch_preds[0, masked_index]).item()
    torch_pred_token = tokenizer.convert_ids_to_tokens([torch_pred_idx])[0]

    # TVM output to token
    tvm_pred_idx = compiled_output[0, masked_index].argmax()
    tvm_pred_token = tokenizer.convert_ids_to_tokens([tvm_pred_idx])[0]

    assert torch_pred_idx == tvm_pred_idx
    assert torch_pred_token == tvm_pred_token

    # Print the outputs
    print(f"Torch top-1 id: {torch_pred_idx}, token: {torch_pred_idx}")
    print(f"TVM   top-1 id: {tvm_pred_idx}, token: {tvm_pred_token}")


@pytest.mark.skipif(
    platform.machine() == "aarch64",
    reason="Currently failing on AArch64",
)
def test_convert_torch_script_with_input_types():
    """test_convert_torch_script_with_input_types"""

    def model_fn(x, y):
        x = x.to(dtype=torch.int32)
        y = x + y
        return y

    ishape = (4, 5)
    input_x = torch.rand(ishape, dtype=torch.float32)
    input_y = torch.randint(low=0, high=100, size=ishape, dtype=torch.int32)
    inputs = [input_x, input_y]
    script_module = torch.jit.trace(model_fn, inputs)

    fname = "tmp.pt"
    torch.jit.save(script_module, fname)
    loaded = torch.jit.load(fname)
    os.remove(fname)

    verify_model(loaded.eval(), input_data=inputs)

    def expected(x_shape, y_shape):
        # use a fixed order of args so alpha equal check can pass
        x = relay.var("x", shape=x_shape, dtype="float32")
        y = relay.var("y", shape=y_shape, dtype="int32")
        args = [x, y]
        x1 = relay.cast(x, "int32")
        y1 = relay.add(x1, y)
        mod = tvm.IRModule.from_expr(relay.Function(args, y1))
        return mod["main"]

    input_infos = [("input0", (ishape, "float")), ("input1", (ishape, "int"))]
    mod, _ = relay.frontend.from_pytorch(loaded, input_infos)

    expected_mod = expected(ishape, ishape)

    assert tvm.ir.structural_equal(expected_mod, mod["main"], map_free_vars=True)


def test_bincount():
    """test_bincount"""

    def test_fn(x, weights=None):
        return torch.bincount(x, weights=weights)

    inp = torch.randint(0, 100, (10000,), dtype=torch.int64)
    weights = torch.linspace(0, 100, steps=10000)

    targets = ["llvm", "cuda"]
    verify_trace_model(test_fn, [inp], targets)
    verify_trace_model(test_fn, [inp, weights], targets)


def test_hard_swish():
    """test_hard_swish"""
    examples = [torch.rand(8).float(), torch.rand(8, 10).float(), torch.rand(1, 1, 10).float()]
    for input_data in examples:
        verify_model(torch.nn.Hardswish().eval(), input_data=input_data)
        verify_model(torch.nn.Hardswish(inplace=True).eval(), input_data=input_data)


def test_hard_sigmoid():
    """test_hard_sigmoid"""
    examples = [torch.rand(8).float(), torch.rand(8, 10).float(), torch.rand(1, 1, 10).float()]
    for input_data in examples:
        verify_model(torch.nn.Hardsigmoid().eval(), input_data=input_data)
        verify_model(torch.nn.Hardsigmoid(inplace=True).eval(), input_data=input_data)


def test_cumsum():
    """test_cumsum"""

    def test_fn(dim, dtype=None):
        return lambda x: torch.cumsum(x, dim=dim, dtype=dtype)

    inp = torch.randint(0, 100, (10000,), dtype=torch.int32)
    verify_model(test_fn(0), [inp])
    verify_model(test_fn(0), [inp.to(torch.int64)])
    verify_model(test_fn(0, dtype=torch.int64), [inp.to(torch.int64)])

    inp = torch.randn((100, 100), dtype=torch.float32)
    verify_model(test_fn(dim=0, dtype=torch.float64), [inp])
    verify_model(test_fn(dim=1), [inp])

    inp = torch.randn((100, 100), dtype=torch.float32) > 0.5
    verify_model(test_fn(dim=0, dtype=torch.int32), [inp])


def test_masked_fill():
    """test_transformer"""

    def test_fn(x, mask):
        return torch.masked_fill(x, mask, 0.0)

    inp = torch.randn(100, 100)
    verify_model(test_fn, [inp, inp > 0.5])
    verify_model(test_fn, [inp.to(torch.float64), inp > 0.5])


def test_transformer():
    """test_transformer"""
    model = torch.nn.Transformer(d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
    model = model.eval()
    src = torch.rand((10, 32, 256))
    tgt = torch.rand((20, 32, 256))
    verify_model(model.eval(), input_data=[src, tgt])


def test_argsort():
    """test_argsort"""

    def test_fn(dim, descending):
        return lambda x: torch.argsort(x, dim=dim, descending=descending)

    inp = torch.randn(100)
    verify_model(test_fn(0, True), [inp])
    verify_model(test_fn(0, False), [inp])

    inp = torch.randn(100, 100)
    verify_model(test_fn(0, True), [inp])
    verify_model(test_fn(0, False), [inp])
    verify_model(test_fn(1, True), [inp])
    verify_model(test_fn(1, False), [inp])


def test_sort():
    """test_sort"""

    def test_fn(dim, descending):
        return lambda x: torch.sort(x, dim=dim, descending=descending)

    inp = torch.randn(100)
    verify_model(test_fn(0, True), [inp])
    verify_model(test_fn(-1, False), [inp])

    inp = torch.randn(100, 100)
    verify_model(test_fn(0, True), [inp])
    verify_model(test_fn(-2, False), [inp])
    verify_model(test_fn(1, True), [inp])
    verify_model(test_fn(-1, False), [inp])


def test_logical_and():
    """test_logical_and"""

    def test_fn(x, y):
        return torch.logical_and(x, y)

    a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
    b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
    verify_model(test_fn, [a, b])

    a = torch.tensor([True, False, True])
    b = torch.tensor([True, False, False])
    verify_model(test_fn, [a, b])


def test_masked_select():
    """test_masked_select"""

    def test_fn(x, mask):
        return torch.masked_select(x, mask)

    for shape in [(10,), (3, 4), (16, 32, 64)]:
        x = torch.randn(*shape)
        mask = x.ge(0.5)
        verify_trace_model(test_fn, [x, mask], ["llvm", "cuda"])


def test_unique():
    """test_unique"""

    def test_fn(is_sorted, return_inverse, return_counts):
        return lambda x: torch.unique(x, is_sorted, return_inverse, return_counts)

    in_data = torch.randint(0, 20, (10,), dtype=torch.int32)
    targets = ["llvm", "cuda"]
    verify_trace_model(test_fn(True, True, True), [in_data], targets)
    verify_trace_model(test_fn(True, False, True), [in_data], targets)
    verify_trace_model(test_fn(True, True, False), [in_data], targets)
    verify_trace_model(test_fn(True, False, True), [in_data], targets)
    in_data = torch.randint(0, 20, (20,), dtype=torch.int64)
    verify_trace_model(test_fn(True, True, True), [in_data], targets)
    verify_trace_model(test_fn(True, False, True), [in_data], targets)
    verify_trace_model(test_fn(True, True, False), [in_data], targets)
    verify_trace_model(test_fn(True, False, True), [in_data], targets)


def test_forward_nll_loss():
    """test_forward_nll_loss"""
    torch.set_grad_enabled(False)
    N, C = 10, 3
    predictions = torch.rand((N, C)).float()
    targets = torch.randint(0, 3, (N,))
    weights = torch.tensor([1, 2, 3]).float()
    verify_model(torch.nn.NLLLoss().eval(), input_data=[predictions, targets])
    verify_model(torch.nn.NLLLoss(weight=weights).eval(), input_data=[predictions, targets])
    verify_model(torch.nn.NLLLoss(ignore_index=1).eval(), input_data=[predictions, targets])
    verify_model(torch.nn.NLLLoss(reduction="sum").eval(), input_data=[predictions, targets])
    verify_model(torch.nn.NLLLoss(reduction="none").eval(), input_data=[predictions, targets])

    # multidimension nll loss (aten::nll_loss2d)
    d1, d2 = 2, 3
    predictions = torch.rand((N, C, d1, d2)).float()
    targets = torch.randint(0, 3, (N, d1, d2))
    verify_model(torch.nn.NLLLoss().eval(), input_data=[predictions, targets])
    verify_model(torch.nn.NLLLoss(weight=weights).eval(), input_data=[predictions, targets])
    verify_model(torch.nn.NLLLoss(ignore_index=1).eval(), input_data=[predictions, targets])
    verify_model(torch.nn.NLLLoss(reduction="sum").eval(), input_data=[predictions, targets])
    verify_model(torch.nn.NLLLoss(reduction="none").eval(), input_data=[predictions, targets])


def test_cross_entropy_loss():
    """test_cross_entropy_loss"""
    torch.set_grad_enabled(False)
    N, C = 10, 3
    # class indices
    predictions = torch.rand((N, C)).float()
    targets = torch.randint(0, 3, (N,))
    weights = torch.tensor([1, 2, 3]).float()
    verify_model(torch.nn.CrossEntropyLoss().eval(), input_data=[predictions, targets])
    verify_model(
        torch.nn.CrossEntropyLoss(weight=weights).eval(), input_data=[predictions, targets]
    )

    # class probabilities
    predictions = torch.randn(N, C).float()
    targets = torch.randn(N, C)
    verify_model(torch.nn.CrossEntropyLoss().eval(), input_data=[predictions, targets])


def test_forward_l1_loss():
    """test_forward_l1_loss"""
    torch.set_grad_enabled(False)
    N, C = 10, 3
    predictions = torch.rand((N, C)).float()
    targets = torch.rand((N, C)).float()
    verify_model(torch.nn.L1Loss().eval(), input_data=[predictions, targets])
    verify_model(torch.nn.L1Loss(reduction="sum").eval(), input_data=[predictions, targets])
    verify_model(torch.nn.L1Loss(reduction="none").eval(), input_data=[predictions, targets])

    # multidimension l1 loss
    d1, d2 = 2, 3
    predictions = torch.rand((N, C, d1, d2)).float()
    targets = torch.rand((N, C, d1, d2)).float()
    verify_model(torch.nn.L1Loss().eval(), input_data=[predictions, targets])
    verify_model(torch.nn.L1Loss(reduction="sum").eval(), input_data=[predictions, targets])
    verify_model(torch.nn.L1Loss(reduction="none").eval(), input_data=[predictions, targets])


def test_forward_mse_loss():
    """test_forward_mse_loss"""
    torch.set_grad_enabled(False)
    N, C = 10, 3
    predictions = torch.rand((N, C)).float()
    targets = torch.rand((N, C)).float()
    verify_model(torch.nn.MSELoss().eval(), input_data=[predictions, targets])
    verify_model(torch.nn.MSELoss(reduction="sum").eval(), input_data=[predictions, targets])
    verify_model(torch.nn.MSELoss(reduction="none").eval(), input_data=[predictions, targets])

    # multidimension mse loss
    d1, d2 = 2, 3
    predictions = torch.rand((N, C, d1, d2)).float()
    targets = torch.rand((N, C, d1, d2)).float()
    verify_model(torch.nn.MSELoss().eval(), input_data=[predictions, targets])
    verify_model(torch.nn.MSELoss(reduction="sum").eval(), input_data=[predictions, targets])
    verify_model(torch.nn.MSELoss(reduction="none").eval(), input_data=[predictions, targets])


@tvm.testing.uses_gpu
def test_forward_flip():
    """Test for aten::flip"""
    torch.set_grad_enabled(False)

    class Flip(Module):
        def __init__(self, axis=0):
            super().__init__()
            self.axis = axis

        def forward(self, x):
            return x.flip([self.axis])

    input_t = torch.randn(2, 3, 4)
    verify_model(Flip(axis=0), input_data=input_t)
    verify_model(Flip(axis=1), input_data=input_t)
    verify_model(Flip(axis=2), input_data=input_t)
    verify_model(Flip(axis=-1), input_data=input_t)


def test_annotate_span():
    """test_annotate_span"""
    model = torchvision.models.resnet18().eval()
    inp = torch.randn([1, 3, 224, 224])
    trace = torch.jit.trace(model, inp).eval()
    mod, _ = relay.frontend.from_pytorch(
        trace, [("input", inp.shape)], use_parser_friendly_name=True
    )
    relay.transform.AnnotateSpans()(mod)


@tvm.testing.uses_gpu
def test_all_any():
    """test_all_any"""

    def test_fn(f, dim=None, keepdim=False):
        return lambda x: f(x, dim=dim, keepdim=keepdim)

    def test_fn_no_arg(f):
        return lambda x: f(x)  # pylint: disable=unnecessary-lambda

    for f in [torch.all, torch.any]:
        verify_model(test_fn(f, 0), [torch.rand(1, 2).bool()])
        verify_model(test_fn(f, 0), [torch.arange(0, 3).to(torch.uint8)])
        verify_model(test_fn(f, 1), [torch.rand(4, 2).bool()])
        verify_model(test_fn(f, 0, keepdim=True), [torch.rand(4, 2).bool()])
        verify_model(test_fn_no_arg(f), [torch.rand(1, 2).bool()])
        verify_model(test_fn_no_arg(f), [torch.arange(0, 3).to(torch.uint8)])


@tvm.testing.uses_gpu
def test_searchsorted():
    """test_searchsorted"""

    def test_fn(out_int32=False, right=False):
        return lambda x, y: torch.searchsorted(x, y, out_int32=out_int32, right=right)

    sorted_sequence = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
    values = torch.tensor([[3, 6, 9], [3, 6, 9]])
    verify_model(test_fn(), [sorted_sequence, values])
    verify_model(test_fn(out_int32=True), [sorted_sequence[0], values[0]])
    verify_model(test_fn(right=True), [sorted_sequence, values])

    sorted_sequence_1d = torch.tensor([1, 3, 5, 7, 9])
    values = torch.tensor([[3, 6, 9], [4, 2, 7]])
    verify_model(test_fn(), [sorted_sequence_1d, values])

    verify_model(test_fn(), [sorted_sequence_1d, torch.tensor(6)])


@tvm.testing.uses_gpu
def test_bucketize():
    """test_bucketize"""

    def test_fn(out_int32=False, right=False):
        return lambda x, y: torch.bucketize(x, y, out_int32=out_int32, right=right)

    boundaries = torch.tensor([1, 3, 5, 7, 9])
    values = torch.tensor([3, 6, 9])

    verify_model(test_fn(), [values, boundaries])
    verify_model(test_fn(out_int32=True, right=True), [values, boundaries])


@tvm.testing.uses_gpu
def test_roll():
    """Test for aten::roll"""

    def test_fn(shifts, dims):
        return lambda x: torch.roll(x, shifts, dims)

    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
    verify_model(test_fn(1, 0), [x])
    verify_model(test_fn(-1, 0), [x])
    verify_model(test_fn(shifts=(2, 1), dims=(0, 1)), [x])


@tvm.testing.uses_gpu
def test_einsum():
    """test_einsum"""

    def test_fn(equation):
        return lambda *x: torch.einsum(equation, *x)

    x = torch.ones([2, 3])
    y = torch.ones([3, 4])
    z = torch.ones([4, 5])
    verify_model(test_fn("ij,jk"), [x, y])
    verify_model(test_fn("ij,jk,km->im"), [x, y, z])


def test_stft():
    """test_stft"""

    def test_fn(n_fft, hop_length, win_length, center, pad_mode, normalized, onesided):
        return lambda input, window=None: torch.stft(
            input=input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
        )

    input_t = torch.rand([1, 12]).float()
    window = torch.tensor([2, 3, 4], dtype=torch.int32)
    targets = ["llvm", "cuda"]
    verify_trace_model(test_fn(3, 3, 3, False, "constant", False, True), [input_t, window], targets)
    verify_trace_model(test_fn(3, 3, 3, True, "constant", False, True), [input_t, window], targets)
    verify_trace_model(test_fn(3, 3, 3, False, "reflect", False, True), [input_t, window], targets)
    verify_trace_model(test_fn(3, 3, 3, True, "reflect", False, True), [input_t, window], targets)
    verify_trace_model(test_fn(3, 3, 3, True, "reflect", True, True), [input_t, window], targets)
    verify_trace_model(test_fn(3, 3, 3, True, "reflect", False, False), [input_t, window], targets)
    input_t = torch.rand([2, 12]).float()
    window = torch.tensor([2, 3, 4], dtype=torch.int32)
    verify_trace_model(test_fn(3, 3, 3, False, "reflect", False, True), [input_t, window], targets)
    window = torch.tensor([1, 3], dtype=torch.int32)
    verify_trace_model(test_fn(2, 1, 2, False, "reflect", False, True), [input_t, window], targets)
    verify_trace_model(test_fn(2, 1, 2, False, "reflect", False, True), [input_t], targets)


@tvm.testing.uses_gpu
def test_dot():
    """Test for aten::dot"""

    def test_fn(x):
        return x.dot(x)

    x = torch.randn([4])
    verify_model(test_fn, [x])


@tvm.testing.uses_gpu
def test_mv():
    """Test for aten::mv"""

    def test_fn(m, v):
        return m.mv(v)

    verify_model(test_fn, [torch.randn(4, 4), torch.randn(4)])
    verify_model(test_fn, [torch.randn(2, 2), torch.randn(2)])
    verify_model(test_fn, [torch.randn(3, 8), torch.randn(8)])


def test_grid_sample():
    """test_grid_sample"""

    class Grid_sample(Module):
        def __init__(self, method, padding_mode, align_corners):
            super().__init__()
            self._method = method
            self._padding_mode = padding_mode
            self._align_corners = align_corners

        def forward(self, x, y):
            return torch.nn.functional.grid_sample(
                input=x,
                grid=y,
                mode=self._method,
                padding_mode=self._padding_mode,
                align_corners=self._align_corners,
            )

    methods = ["nearest", "bilinear", "bicubic"]
    padding_modes = ["zeros", "border", "reflection"]
    align_corners = [True, False]

    data_2D = torch.rand([4, 4, 8, 8]).float()
    grid_2D = torch.rand([4, 16, 16, 2]).float()
    # choosing smaller sizes to be testable on weaker GPUs
    data_3D = torch.rand([4, 4, 4, 4, 4]).float()
    grid_3D = torch.rand([4, 8, 8, 8, 3]).float()

    for _method in methods:
        for _padding in padding_modes:
            for _align in align_corners:
                # ATTENTION:
                #   "nearest" + "reflection" result may be different with pytorch on cpu device,
                #   because pytorch's cpu result is different with gpu result,
                #   and gpu result used here as baseline in tvm topi.image.grid_sample.
                model = Grid_sample(_method, _padding, _align)
                verify_model(model, input_data=[data_2D, grid_2D])

                # 3D "bicubic"(tricubic) is not supported in pytorch
                if _method != "bicubic":
                    verify_model(model, input_data=[data_3D, grid_3D])


def test_list_tuple():
    """test compilation error for a Python list followed by a prim::TupleConstruct."""

    class List_tuple(Module):
        """List_tuple"""

        def forward(self, x):
            """forward"""
            merged = []
            mask_list = []
            for i in range(3):
                w0 = torch.sigmoid(x)
                merged.append((w0, w0))
                mask_list.append(x)

            for i in range(3):
                merged[i] = merged[i][0] + merged[i][1]
            return mask_list[2], merged

    x = torch.rand([4, 4, 16, 32]).float()
    script_module = torch.jit.trace(List_tuple(), x, strict=False).eval()
    relay.frontend.from_pytorch(script_module, [("x", x.shape)])


# pylint: disable=unnecessary-dunder-call
@tvm.testing.uses_gpu
def test_binary_bitwise():
    """Test for binary bitwise"""

    def test_ior(x, y):
        return x.__ior__(y)

    def test_iand(x, y):
        return x.__iand__(y)

    def test_ixor(x, y):
        return x.__ixor__(y)

    x = torch.tensor([7, 49, 16, 1, 2, 3], dtype=torch.uint8)
    y = torch.tensor([39, 128, 99, 228, 63, 17], dtype=torch.uint8)

    for test_fn in [test_ior, test_iand, test_ixor]:
        verify_model(test_fn, [x, y])


@tvm.testing.uses_gpu
def test_shift():
    """Test for aten::__lshift__, aten::__rshift__"""

    def test_lshift(x, y):
        return x << y

    def test_rshift(x, y):
        return x >> y

    x = torch.tensor([39, 128, 99, 228, 63, 17], dtype=torch.int32)
    y = torch.tensor([3, 2, 7, 4, 5, 9], dtype=torch.int32)

    for test_fn in [test_lshift, test_rshift]:
        verify_model(test_fn, [x, y])


@tvm.testing.uses_gpu
def test_mod():
    """Test for aten::fmod"""

    def test_fmod(x, y):
        return torch.fmod(x, y)

    def test_remainder(x, y):
        return torch.remainder(x, y)

    for test_fn in [test_fmod, test_remainder]:
        verify_model(test_fn, [torch.tensor([-3.0, -2, -1, 1, 2, 3]), torch.tensor(2)])
        verify_model(test_fn, [torch.tensor([1, 2, 3, 4, 5]), torch.tensor(-1.5)])


def test_softmax_fuse():
    """test_softmax_fuse"""
    # https://github.com/apache/tvm/issues/12001
    class Model(torch.nn.Module):
        """Pytorch model module"""

        def __init__(self, nchwc_post_op=False) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, (1, 1), 1)
            self.nchwc_post_op = nchwc_post_op

        @torch.no_grad()
        def forward(self, x):
            """forward"""
            t0a = self.conv(x)
            t0b = torch.floor(x)
            t2b = torch.softmax(t0a, dim=2)

            if self.nchwc_post_op:
                t3a = t0a - t0b
                t4a = t2b - t0b
                t6a = t3a + t4a
                return t6a

            return t2b + 1

    sh = [3, 3, 10, 1]
    inp = torch.ones(*sh, dtype=torch.float32)

    for model in [Model(nchwc_post_op=False).eval(), Model(nchwc_post_op=True).eval()]:
        output_torch = model(inp).numpy()

        mod, params = relay.frontend.from_pytorch(torch.jit.trace(model, inp), [("inp0", sh)])

        with tvm.transform.PassContext(opt_level=4):
            out = (
                relay.create_executor("graph", mod, params=params)
                .evaluate()(inp0=inp.numpy())
                .numpy()
            )

        tvm.testing.assert_allclose(out, output_torch, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_lerp():
    """test_lerp"""

    def test_fn(x, y, w):
        return torch.lerp(x, y, w)

    input_shape = [16]
    x = torch.rand(input_shape).float()
    y = torch.rand(input_shape).float()
    w = torch.rand(input_shape).float()

    # weight can be tensor or scalar
    verify_model(test_fn, [x, y, w])
    verify_model(test_fn, [x, y, w[0]])


def test_trilu():
    def _test_trilu(op, diagonal):
        return lambda inp: op(inp, diagonal)

    for op in [torch.triu, torch.tril]:
        verify_model(_test_trilu(op, 0), [torch.rand(size=[3, 3])])
        verify_model(_test_trilu(op, 1), [torch.rand(size=[6, 6])])
        verify_model(_test_trilu(op, -2), [torch.rand(size=[6, 6])])


def test_multinomial():
    def _test_multinomial(num_samples):
        return lambda inp: torch.multinomial(inp, num_samples=num_samples, replacement=True)

    # Dont check output since it's random. Instead we'll just make sure shapes are right.
    verify_model(
        _test_multinomial(2), [torch.rand(size=[3]).float()], cpu_only=True, check_correctness=False
    )
    verify_model(
        _test_multinomial(1),
        [torch.rand(size=[4, 5]).float()],
        cpu_only=True,
        check_correctness=False,
    )


if __name__ == "__main__":
    tvm.testing.main()
