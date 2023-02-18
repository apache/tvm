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

# pylint: disable=invalid-name, missing-function-docstring, not-callable
# pylint: disable=import-outside-toplevel, unused-argument
# mypy: ignore-errors
"""PyTorch Dynamo backend of Relax."""
import functools
from typing import Optional

import tvm
from tvm.relax.vm import build as relax_build
from tvm.relax.frontend.torch.fx_translator import from_fx


def device_from_inputs(example_inputs):
    for x in example_inputs:
        if hasattr(x, "device"):
            return x.device
    return None


def relax_dynamo(pipeline: Optional[tvm.transform.Pass] = None):
    """A helper function to create a relax backend.

    Parameters
    ----------
    pipeline : Optional[tvm.transform.Pass]
        The pipeline to be applied to the relax module before sent to build.

    Returns
    -------
    backend : Callable[[torch.fx.GraphModule, List[torch.Tensor]], Callable]
        The relax dynamo backend.
    """

    def _relax_backend(graph_module, example_inputs):
        import torch  # type: ignore[import]

        assert isinstance(graph_module, torch.fx.GraphModule)

        def to_torch_tensor(nd_tensor):
            """A helper function to transfer a NDArray to torch.tensor."""
            if isinstance(nd_tensor, tvm.nd.NDArray):
                return torch.from_numpy(nd_tensor.numpy())
            elif isinstance(nd_tensor, tvm.ir.Array):
                return tuple(to_torch_tensor(x) for x in nd_tensor)
            else:
                raise ValueError(f"Unsupported type {type(nd_tensor)}")

        def to_tvm_tensor(torch_tensor):
            """A helper function to transfer a torch.tensor to NDArray."""
            if not isinstance(torch_tensor, torch._subclasses.fake_tensor.FakeTensor):
                return tvm.nd.array(torch_tensor.numpy())
            # Fake Tensor
            real_tensor = torch.randn(torch_tensor.shape, dtype=torch_tensor.dtype)
            return tvm.nd.array(real_tensor.numpy())

        device = device_from_inputs(example_inputs)
        input_info = [(tuple(tensor.shape), str(tensor.dtype)) for tensor in example_inputs]
        mod = from_fx(graph_module, input_info)

        if device.type == "cuda":
            dev = tvm.cuda(device.index)
            target = tvm.target.cuda()
        else:
            dev = tvm.cpu(0)
            target = tvm.target.Target(llvm_target())

        # invoke optimization pipeline.
        if pipeline is None:
            # get default pipeline
            seq = tvm.relax.get_pipeline()
        elif isinstance(pipeline, str):
            # lookup by name
            seq = tvm.relax.get_pipeline(pipeline)
        else:
            seq = pipeline

        mod = mod.with_attr("target", target)
        mod = seq(mod)

        ex = relax_build(mod, target=target)

        vm = tvm.relax.vm.VirtualMachine(exec=ex.mod, device=dev)

        def exec_tvm(*i_args):
            args = [a.contiguous() for a in i_args]
            vm_args = list()
            for arg in args:
                if arg.dim() != 0:
                    if arg.requires_grad:
                        arg = arg.detach()
                    vm_args.append(to_tvm_tensor(arg))
            outputs = vm["main"](*vm_args)
            return to_torch_tensor(outputs)

        return exec_tvm

    return _relax_backend


def dynamo_capture_subgraphs(model, *params) -> tvm.ir.IRModule:
    """Capture subgraphs of the PyTorch model using torch.compile into an IRModule.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be captured.

    params : List[torch.Tensor]
        The parameters of the PyTorch model.

    Returns
    -------
    mod : tvm.ir.IRModule
        The IRModule that contains captured subgraphs.
    """
    import torch  # type: ignore[import]
    from torch import fx  # type: ignore[import]
    from torch import _dynamo as dynamo  # type: ignore[import]

    mod = tvm.IRModule()

    def _capture(graph_module: fx.GraphModule, example_inputs):
        assert isinstance(graph_module, torch.fx.GraphModule)
        input_info = [(tuple(tensor.shape), str(tensor.dtype)) for tensor in example_inputs]
        subgraph = from_fx(graph_module, input_info)
        mod["subgraph_" + str(len(mod.get_global_vars()))] = subgraph["main"]
        return graph_module.forward

    dynamo.reset()
    compiled_model = torch.compile(model, backend=_capture)
    compiled_model(*params)
    return mod


@functools.lru_cache(None)
def llvm_target():
    if "avx512" in open("/proc/cpuinfo").read():
        return "llvm -mcpu=skylake-avx512"
    return "llvm -mcpu=core-avx2"
