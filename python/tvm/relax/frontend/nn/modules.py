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
# pylint: disable=too-many-arguments,invalid-name,protected-access
"""Builtin Modules."""
from typing import List, Optional, Sequence, Union

from tvm import relax as rx
from tvm import tir
from tvm._ffi import register_func
from tvm.runtime import NDArray

from . import op
from .core import Effect, Module, Parameter, Tensor, get_default_dtype


class IOEffect(Effect):
    """
    Modeling IO side effect, for example, printing the content of NDArrays on screen, inserting
    debug breakpoints, etc.
    """

    effect: Optional[rx.Var]

    def __init__(self):
        self.effect = None

    def emit_init(self, name_hint, builder: rx.BlockBuilder) -> List[rx.DataflowVar]:
        return [builder.emit(rx.op.null_value(), f"{name_hint}.io")]

    def create(self, name_hint: str) -> List[rx.Var]:
        assert self.effect is None
        self.effect = rx.Var(f"{name_hint}.io", struct_info=rx.ObjectStructInfo())
        return [self.effect]

    def finalize(self) -> List[rx.Var]:
        result = self.effect
        self.effect = None
        return [result]

    def print_(self, tensor: Tensor) -> None:
        """Encloses the side effect of NDArray printing"""
        self.effect = rx.BlockBuilder.current().emit(
            rx.call_pure_packed(
                rx.extern("effect.print"),
                self.effect,
                tensor._expr,  # pylint: disable=protected-access
                sinfo_args=[rx.ObjectStructInfo()],
            ),
            name_hint=self.effect.name_hint,
        )


@register_func("effect.print")
def _print(_, array: NDArray) -> None:
    print(f"effect.print: shape = {array.shape}, dtype = {array.dtype}, data =\n{array}")


class SiLU(Module):
    """
    Module for SiLU activation layer.
    """

    def forward(self, x: Tensor):
        return op.silu(x)


class Linear(Module):
    """
    Module for linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[str] = None,
        out_dtype: Optional[str] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.weight = Parameter((out_features, in_features), dtype)
        if bias:
            self.bias = Parameter((out_features,), dtype)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward method for linear layer.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        ret : Tensor
            The output tensor for the linear layer.
        """
        # x: [*B, in_features]
        # w: [in_features, out_features]
        w = op.permute_dims(self.weight)
        # x: [*B, out_features]
        x = op.matmul(x, w, out_dtype=self.out_dtype)
        if self.bias is not None:
            x = x + self.bias
        return x


class Conv1D(Module):
    """
    Module for conv1d layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = Parameter(
            (
                self.out_channels,
                int(self.in_channels // self.groups),
                self.kernel_size,
            ),
            dtype,
        )
        if bias:
            self.bias = Parameter((self.out_channels,), dtype)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward method for conv1d layer.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        ret : Tensor
            The output tensor for the conv1d layer.
        """
        return op.conv1d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class Conv2D(Module):
    """
    Module for conv2d layer.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: Optional[str] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = Parameter(
            (
                self.out_channels,
                int(self.in_channels / self.groups),
                self.kernel_size,
                self.kernel_size,
            ),
            dtype,
        )
        if bias:
            self.bias = Parameter((self.out_channels,), dtype)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=invalid-name
        """
        Forward method for conv2d layer.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        ret : Tensor
            The output tensor for the conv2d layer.
        """
        return op.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class LayerNorm(Module):
    """
    Module for Layer Normalization
    """

    def __init__(
        self,
        normalized_shape: int,
        axes: Union[int, List[int]],
        eps: Optional[float] = 1e-5,
        dtype: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.axes = axes
        self.weight = Parameter((normalized_shape,), dtype=dtype)
        self.bias = Parameter((normalized_shape,), dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward method for layer normalization layer.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        ret : Tensor
            The output tensor for the layer normalization layer.
        """
        out = op.layer_norm(x, weight=self.weight, bias=self.bias, axes=self.axes, epsilon=self.eps)
        return out


class RMSNorm(Module):
    """
    Module for rms norm layer.
    """

    def __init__(
        self,
        hidden_size: int,
        axes: Union[int, List[int]],
        epsilon: float = 1e-5,
        bias: bool = True,
        dtype: Optional[str] = None,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.axes = axes
        self.weight = Parameter((hidden_size,), dtype=dtype)
        if bias:
            self.bias = Parameter((hidden_size,), dtype=dtype)
        else:
            self.bias = None

    def forward(self, x: Tensor):
        """
        Forward method for rms norm layer.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        ret : Tensor
            The output tensor for the rms norm layer.
        """
        out = op.rms_norm(x, weight=self.weight, axes=self.axes, epsilon=self.epsilon)
        if self.bias:
            out = op.add(out, self.bias)
        return out


class GroupNorm(Module):
    """
    Module for group norm layer.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        dtype: Optional[str] = None,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        if affine:
            self.weight = Parameter((num_channels,), dtype=dtype)
            self.bias = Parameter((num_channels,), dtype=dtype)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor):
        """
        Forward method for group norm layer.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        ret : Tensor
            The output tensor for the group norm layer.
        """
        return op.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class KVCache(Effect):
    """
    Effect to implement KVCache.
    """

    init_seq_len: int
    unit_shape: List[int]
    dtype: str
    cache: Optional[rx.Var]

    def __init__(
        self,
        init_seq_len: int,
        unit_shape: Sequence[int],
        dtype: Optional[str] = None,
    ):
        if dtype is None:
            dtype = get_default_dtype()
        # Usually the shape is: [init_seq_len, num_heads, head_dim]
        # and unit_shape = [num_heads, head_dim]
        self.init_seq_len = init_seq_len
        self.unit_shape = [int(i) for i in unit_shape]
        self.dtype = dtype

    def emit_init(self, name_hint: str, bb: rx.BlockBuilder):  # pylint: disable=arguments-renamed
        """
        Emit the initialization of the KVCache effect.

        Parameters
        ----------
        name_hint : str
            The name hint of the initialization binding Var.

        bb : relax.BlockBuilder
            The relax BlockBuilder to emit.
        """
        init_shape = rx.ShapeExpr([self.init_seq_len] + self.unit_shape)
        return [
            bb.emit(
                rx.Call(
                    rx.extern("vm.builtin.attention_kv_cache_create"),
                    args=[rx.op.zeros(init_shape, self.dtype), init_shape, rx.PrimValue(0)],
                    sinfo_args=[rx.ObjectStructInfo()],
                ),
                name_hint=name_hint,
            )
        ]

    def create(self, name_hint: str) -> rx.Var:
        """
        Create the implicit inputs to a relax.Function that represents the KVCache effect.

        Parameters
        ----------
        name_hint : str
            The name hint of the relax.Var.

        Returns
        -------
        ret : relax.Var
            The relax.Var for KVCache.
        """
        self.cache = rx.Var(name_hint, struct_info=rx.ObjectStructInfo())
        return [self.cache]

    def finalize(self) -> List[rx.Var]:
        """
        Finalize the KVCache effect as the implicit return value of a relax.Function.

        Returns
        -------
        ret : List[rx.Var]
            The output relax.Var as KVCache.
        """
        result = self.cache
        self.cache = None
        return [result]

    def to(self, dtype: Optional[str] = None) -> None:
        """
        Convert the KVCache effect to specific dtype.

        Parameters
        ----------
        dtype : Optional[str]
            The target data type to convert.
        """
        if dtype is not None:
            self.dtype = dtype

    def view(self, seq_len: tir.Var) -> Tensor:
        """
        View the last elements in KVCache.

        Parameters
        ----------
        seq_len : tir.Var
            The number of last elements to view.

        Returns
        -------
        ret : Tensor
            The last tensor to view.
        """
        shape = rx.ShapeExpr([seq_len] + self.unit_shape)
        return Tensor(
            _expr=rx.BlockBuilder.current().emit(
                rx.Call(
                    rx.extern("vm.builtin.attention_kv_cache_view"),
                    args=[self.cache, shape],
                    sinfo_args=[rx.TensorStructInfo(shape, self.dtype)],
                )
            )
        )

    def append(self, new_element: Tensor) -> None:
        """
        Append a new element in KVCache.

        Parameters
        ----------
        new_element : Tensor
            The new tensor to append.
        """
        if new_element.dtype != self.dtype:
            raise TypeError(
                f'KVCache has been set to use dtype "{self.dtype}", '
                f'but got "{new_element.dtype}"'
            )
        self.cache = rx.BlockBuilder.current().emit(
            rx.Call(
                rx.extern("vm.builtin.attention_kv_cache_append"),
                args=[self.cache, new_element._expr],
                sinfo_args=[rx.ObjectStructInfo()],
            )
        )


class Embedding(Module):
    """
    Module for embedding layer.
    """

    def __init__(self, num: int, dim: int, dtype: Optional[str] = None):
        self.num = num
        self.dim = dim
        self.weight = Parameter((num, dim), dtype=dtype)

    def forward(self, x: Tensor):
        """
        Forward method for embedding layer.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        ret : Tensor
            The output tensor for the embedding layer.
        """
        if x.ndim == 1:
            return op.take(self.weight, x, axis=0)
        return op.reshape(
            op.take(
                self.weight,
                op.reshape(x, shape=[-1]),
                axis=0,
            ),
            shape=[*x.shape, self.dim],  # TODO(@junrushao): revisit and remove self.dim
        )


class TimestepEmbedding(Module):
    """
    Module for HF TimestepEmbedding layer.
    """

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim: Optional[int] = None,
    ):
        self.linear_1 = Linear(in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        assert act_fn == "silu", "Only SiLU activations are supported."
        self.act = SiLU()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim

        self.linear_2 = Linear(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            self.post_act = None
        else:
            assert self.post_act == "silu", "Only SiLU post-activation supported."
            self.post_act = SiLU()

    def forward(self, sample: Tensor, condition: Optional[Tensor] = None):
        """
        Forward method for TimestepEmbedding layer.

        Parameters
        ----------
        sample : Tensor
            The input timestep that should be looked up.
        condition : Optional[Tensor]
            Optional additional projection matrix.

        Returns
        -------
        ret : Tensor
            The resulting embedding lookup for the input sample.
        """
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(Module):
    """
    Module for HF timesteps layer.
    """

    def __init__(
        self, num_channels: int, flip_sin_to_cos: bool = False, downscale_freq_shift: float = 1
    ):
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, x: Tensor):
        return op.get_timestep_embedding(
            x,
            embedding_dim=self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
