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
"""Stream context."""
from ctypes import c_void_p
from typing import Optional, Any, Union
from . import core
from ._tensor import device

try:
    import torch

    class TorchStreamContext:
        def __init__(self, context: Optional[Any]):
            self.torch_context = context

        def __enter__(self):
            if self.torch_context:
                self.torch_context.__enter__()
            current_stream = torch.cuda.current_stream()
            self.ffi_context = core.StreamContext(
                device(str(current_stream.device)), current_stream.cuda_stream
            )
            self.ffi_context.__enter__()

        def __exit__(self, *args):
            if self.torch_context:
                self.torch_context.__exit__(*args)
            self.ffi_context.__exit__(*args)

    def use_torch_stream(context: Optional[Any] = None):
        """
        Create a ffi stream context with given torch stream,
        cuda graph or current stream if `None` provided.

        Parameters
        ----------
        context : Optional[Any]
            The wrapped torch stream or cuda graph.

        Returns
        -------
        context : tvm_ffi.TorchStreamContext
            The ffi stream context wrapping torch stream context.

        Examples
        --------
        .. code-block:: python
        s = torch.cuda.Stream()
        with tvm_ffi.use_torch_stream(torch.cuda.stream(s)):
          ...

        g = torch.cuda.CUDAGraph()
        with tvm_ffi.use_torch_stream(torch.cuda.graph(g)):
          ...
        """
        return TorchStreamContext(context)

except ImportError:

    def use_torch_stream(context: Optional[Any] = None):
        raise ImportError("Cannot import torch")


def use_raw_stream(device: core.Device, stream: Union[int, c_void_p]):
    """
    Create a ffi stream context with given device and stream handle.

    Parameters
    ----------
    device : tvm_ffi.Device
        The device to which the stream belongs.

    stream : Union[int, c_void_p]
        The stream handle.

    Returns
    -------
    context : tvm_ffi.StreamContext
        The ffi stream context.
    """
    if not isinstance(stream, (int, c_void_p)):
        raise ValueError(
            "use_raw_stream only accepts int or c_void_p as stram input, "
            "try use_torch_stream when using torch.cuda.Stream or torch.cuda.graph"
        )
    return core.StreamContext(device, stream)
