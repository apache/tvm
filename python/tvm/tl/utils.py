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
"""The profiler and convert to torch utils"""

from typing import Any, List
from enum import Enum
from functools import partial
import torch

from tvm.relay import TensorType
from tvm.contrib.dlpack import to_pytorch_func

from .engine import lower


class TensorSupplyType(Enum):
    Integer = 1
    Uniform = 2
    Normal = 3
    Randn = 4
    Zero = 5
    One = 6


def get_tensor_supply(supply_type: TensorSupplyType):
    def get_tensor(tensor: TensorType) -> torch.Tensor:
        dtype = torch.__getattribute__(str(tensor.dtype))
        device = torch.cuda.current_device()
        shape = list(map(int, tensor.shape))
        if not dtype.is_floating_point:
            return torch.ones(*shape, device=device, dtype=dtype)
        if supply_type == TensorSupplyType.Integer:
            return torch.randint(low=-2, high=3, size=shape, device=device, dtype=dtype)
        elif supply_type == TensorSupplyType.Uniform:
            return torch.empty(*shape, device=device, dtype=dtype).uniform_(-1.0, 1.0)
        elif supply_type == TensorSupplyType.Normal:
            return torch.empty(*shape, device=device, dtype=dtype).normal_(-1.0, 1.0)
        elif supply_type == TensorSupplyType.Randn:
            return torch.randn(*shape, device=device).to(dtype)
        elif supply_type == TensorSupplyType.Zero:
            return torch.zeros(*shape, device=device, dtype=dtype)
        elif supply_type == TensorSupplyType.One:
            return torch.ones(*shape, device=device, dtype=dtype)
        else:
            raise NotImplementedError(supply_type)

    return get_tensor


class ConvertTorch:
    def __init__(self, mod, params: List[TensorType], result_idx: List[int]) -> None:
        self.mod = mod
        self.params = params
        self.result_idx = result_idx
        self.func = self._convert_torch_func()

    def _convert_torch_func(self) -> callable:
        torch_func = to_pytorch_func(self.mod)

        def func(*ins: List[torch.Tensor]):
            assert len(ins) + len(self.result_idx) == len(self.params)
            ins_idx = 0
            args = []
            device = torch.cuda.current_device()
            for i in range(len(self.params)):
                if i in self.result_idx:
                    dtype = torch.__getattribute__(str(self.params[i].dtype))
                    shape = list(map(int, self.params[i].shape))
                    tensor = torch.empty(*shape, dtype=dtype, device=device)
                else:
                    tensor = ins[ins_idx]
                    ins_idx += 1
                args.append(tensor)
            torch_func(*args)
            if len(self.result_idx) == 1:
                return args[self.result_idx[0]]
            else:
                return [args[i] for i in self.result_idx]

        return func

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)

    def get_kernel_source(self) -> str:
        return self.mod.imported_modules[0].get_source()


class Profiler(ConvertTorch):
    def __init__(
        self,
        mod,
        params: List[TensorType],
        result_idx: List[int],
        supply_type: TensorSupplyType = TensorSupplyType.Normal,
    ):
        super().__init__(mod, params, result_idx)
        self.supply = get_tensor_supply(supply_type)

    def _get_inputs(self):
        ins = []
        for i in range(len(self.params)):
            if i not in self.result_idx:
                ins.append(self.supply(self.params[i]))
        return ins

    def assert_allclose(self, reference_program: callable, atol: float = 1e-8, rtol: float = 1e-5):
        ins = self._get_inputs()
        ref_outs = reference_program(*ins)
        torch.cuda.synchronize()
        lib_outs = self.func(*ins)
        torch.cuda.synchronize()

        if isinstance(lib_outs, torch.Tensor):
            lib_outs = [lib_outs]
        if isinstance(ref_outs, torch.Tensor):
            ref_outs = [ref_outs]
        assert len(lib_outs) == len(ref_outs)
        for lhs, rhs in zip(lib_outs, ref_outs):
            assert torch.allclose(lhs, rhs, rtol=rtol, atol=atol), (lhs, rhs)

    def run_once(self):
        ins = self._get_inputs()
        return self.__call__(*ins)

    def do_bench(self, func: callable, warmup=25, rep=100):
        ins = self._get_inputs()
        bench_func = partial(func, *ins)
        return do_bench(bench_func, warmup=warmup, rep=rep)


def do_bench(
    fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean"
):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float]
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    """
    assert return_mode in ["min", "max", "mean", "median"]
    fn()
    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float
    )
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()


_cached = {}


def cached(func, result_idx: List[int], *args):
    global _cached
    key = (func, tuple(result_idx), *args)
    if key not in _cached:
        program = func(*args)
        mod, params = lower(program)
        mod = ConvertTorch(mod, params, result_idx)
        _cached[key] = mod
    return _cached[key]
