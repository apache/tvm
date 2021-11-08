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
# pylint: disable=invalid-name
"""Module container of PyTorch custom class"""
from typing import List
import torch


class GraphModule(torch.nn.Module):
    r"""Module container of Pytorch class which wraps exported
    TVM op implementation library to be called on Pytorch side"""

    @classmethod
    def shape_repr(cls, input_shapes):
        return torch.ops.tvm_dsoop.tvm_shape_repr(input_shapes)

    def __init__(self, num_inputs, num_outputs, device=None):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        self.engine = None

        if device is not None:
            self.to(device)
        self.engine = torch.classes.tvm_dsoop.TvmGraphModule(num_inputs, num_outputs, self.device)

    def init(self, input_shapes, lib_path, graph_path, params_path):
        r"""Load tvm module"""
        self.engine.load_tvm_module(input_shapes, lib_path, graph_path, params_path)

    def forward(self, inputs: List[torch.Tensor]):
        r"""Call tvm module to forward"""
        return self.engine.forward(inputs)

    @property
    def device(self):
        r"""Get the device string"""
        return str(self.dummy_param.device)

    def _apply(self, fn):
        r"""Override to device function, manually move tvm module to desired device"""
        super()._apply(fn)
        if self.engine is not None:
            self.engine.to(self.device)
        return self


class VMModule(torch.nn.Module):
    r"""Module container of Pytorch class which wraps exported
    TVM op implementation library to be called on Pytorch side"""

    @classmethod
    def shape_repr(cls, input_shapes):
        return torch.ops.tvm_dsoop.tvm_shape_repr(input_shapes)

    def __init__(self, num_inputs, num_outputs, device=None):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        self.engine = None

        if device is not None:
            self.to(device)
        self.engine = torch.classes.tvm_dsoop.TvmVMModule(num_inputs, num_outputs, self.device)

    def init(self, input_shapes, lib_path, code_path):
        r"""Load tvm module"""
        self.engine.load_tvm_module(input_shapes, lib_path, code_path)

    def forward(self, inputs: List[torch.Tensor]):
        r"""Call tvm module to forward"""
        return self.engine.forward(inputs)

    @property
    def device(self):
        r"""Get the device string"""
        return str(self.dummy_param.device)

    def _apply(self, fn):
        r"""Override to device function, manually move tvm module to desired device"""
        super()._apply(fn)
        if self.engine is not None:
            self.engine.to(self.device)
        return self


class TraceTvmModule(torch.nn.Module):
    r"""Wrapper for trace GraphModule

    GraphModule and VMModule only supports List[Tensor] inputs and cannot be traced.
    This is a wrapper class for trace GraphModule or VMModule in order to support
    arbitrary number of inputs

    Example:
        import tvm.contrib.torch
        tvm_module = tvm.contrib.torch.GraphModule(1, 1, 'cuda:0')
        tvm_module.init(input_shapes, lib_path, graph_path, params_path)

        trace_wrapper = tvm.contrib.torch.TraceGraphModule(torch.jit.script(tvm_module))
        traced = torch.jit.trace(trace_wrapper, example_inputs)
    """

    def __init__(self, tvm_module):
        super().__init__()
        self.tvm_module = tvm_module

    def forward(self, *inputs):
        outputs = self.tvm_module(inputs)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
