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
# pylint: disable=too-many-arguments,invalid-name,protected-access,unused-argument
"""Builtin Modules."""
from typing import List, Optional, Sequence, Union

from tvm import relax as rx
from tvm import tir

from . import op
from .core import Effect, Module, ModuleList, Parameter, Tensor, get_default_dtype


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
        effect = rx.Var(f"{name_hint}.io", struct_info=rx.ObjectStructInfo())
        return [effect]

    def set_state(self, state_vars: List[rx.Var]) -> None:
        (self.effect,) = state_vars

    def finalize(self) -> List[rx.Var]:
        result = self.effect
        self.effect = None
        return [result]


class ReLU(Module):
    """Module for ReLU activation layer."""

    def forward(self, x: Tensor):
        return op.relu(x)


class SiLU(Module):
    """Module for SiLU activation layer."""

    def forward(self, x: Tensor):
        return op.silu(x)


class GELU(Module):
    """Module for GELU activation layer."""

    def forward(self, x: Tensor):
        return op.gelu(x)


class Identity(Module):
    """Module that does nothing, sometimes useful for naming purposes."""

    def forward(self, x: Tensor):
        """Forward method for identity.

        Parameters
        ----------
        x : Tensor
            The input tensor.
        Returns
        -------
        Result : Tensor
            The unchanged input tensor.
        """
        return x


class Linear(Module):
    """
    Module for linear layer.
    """

    def __init__(
        self,
        in_features: Union[int, str, tir.PrimExpr],
        out_features: Union[int, str, tir.PrimExpr],
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
            self.bias = Parameter((out_features,), dtype=dtype if out_dtype is None else out_dtype)
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

    def to(self, dtype: Optional[str] = None) -> None:
        """
        Override to() such that we do not convert bias if there is `out_dtype`.
        Otherwise, we might run into dtype mismatch when computing `x + self.bias`
        since x is of type `out_dtype` and bias becomes `dtype`, potentially different.
        """
        self.weight.to(dtype=dtype)
        if self.bias is not None and self.out_dtype is None:
            self.bias.to(dtype=dtype)
        if dtype is not None and isinstance(getattr(self, "dtype", None), str):
            self.dtype = dtype  # pylint: disable=attribute-defined-outside-init


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
        kernel_size: Union[List[int], int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: Optional[str] = None,
        data_layout: str = "NCHW",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.data_layout = data_layout

        # Allow dynamic input channels.
        if isinstance(self.in_channels, int):
            in_channels = int(self.in_channels / self.groups)
        else:
            in_channels = tir.floordiv(self.in_channels, self.groups)

        # Expand kernel size if provided an integer.
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size] * 2
        else:
            self.kernel_size = kernel_size

        kernel_shape = [self.out_channels, in_channels] + list(self.kernel_size)

        self.weight = Parameter(kernel_shape, dtype)

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
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.data_layout,
        )


class Conv3D(Module):
    """
    Module for conv3d layer.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[List[int], int],
        stride: Union[List[int], int] = 1,
        padding: Union[List[int], int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: Optional[str] = None,
        data_layout: str = "NCDHW",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.data_layout = data_layout

        # Allow dynamic input channels.
        if isinstance(self.in_channels, int):
            in_channels = int(self.in_channels / self.groups)
        else:
            in_channels = tir.floordiv(self.in_channels, self.groups)

        # Expand kernel size if given an integer.
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size] * 3
        else:
            self.kernel_size = kernel_size

        kernel_shape = [self.out_channels, self.in_channels] + list(self.kernel_size)

        self.weight = Parameter(kernel_shape, dtype)

        if bias:
            self.bias = Parameter((self.out_channels,), dtype)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=invalid-name
        """
        Forward method for conv3d layer.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        ret : Tensor
            The output tensor for the conv3d layer.
        """
        return op.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.data_layout,
        )


class ConvTranspose1D(Module):
    """
    Module for ConvTranspose1D layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
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
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

        self.weight = Parameter(
            (
                self.in_channels,
                int(self.out_channels // self.groups),
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
        Forward method for conv transpose 1d layer.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        ret : Tensor
            The output tensor for the conv transpose 1d layer.
        """
        return op.conv1d_transpose(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )


class LayerNorm(Module):
    """
    Module for Layer Normalization
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: Optional[float] = 1e-5,
        elementwise_affine: bool = True,
        dtype: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter((normalized_shape,), dtype=dtype)
            self.bias = Parameter((normalized_shape,), dtype=dtype)
        else:
            self.weight = None
            self.bias = None

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
        return op.layer_norm(
            x,
            normalized_shape=self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )


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

    def forward(self, x: Tensor, channel_axis: int = 1, axes: Optional[List[int]] = None):
        """
        Forward method for group norm layer.

        Parameters
        ----------
        x : Tensor
            The input tensor.
        channel_axis : int
            Channel axis of the input data.
        axes : Optional[List[int]]
            Optional list of axes to compute norm over, if not specified,
            assumes that the first two axes should be left alone.

        Returns
        -------
        ret : Tensor
            The output tensor for the group norm layer.
        """
        return op.group_norm(
            x, self.num_groups, self.weight, self.bias, self.eps, channel_axis, axes
        )


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
                rx.op.call_pure_packed(
                    "vm.builtin.attention_kv_cache_create",
                    rx.op.zeros(init_shape, self.dtype),
                    init_shape,
                    rx.PrimValue(0),
                    sinfo_args=rx.ObjectStructInfo(),
                ),
                name_hint=name_hint,
            )
        ]

    def create(self, name_hint: str) -> List[rx.Var]:
        """
        Create the implicit inputs to a relax.Function that represents the KVCache effect.

        Parameters
        ----------
        name_hint : str
            The name hint of the relax.Var.

        Returns
        -------
        ret : List[relax.Var]
            The relax.Var for KVCache.
        """
        cache = rx.Var(name_hint, struct_info=rx.ObjectStructInfo())
        return [cache]

    def set_state(self, state_vars: List[rx.Var]) -> None:
        (self.cache,) = state_vars

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
                rx.op.call_pure_packed(
                    "vm.builtin.attention_kv_cache_view",
                    self.cache,
                    shape,
                    sinfo_args=rx.TensorStructInfo(shape, self.dtype),
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
            rx.op.call_inplace_packed(
                "vm.builtin.attention_kv_cache_append",
                self.cache,
                new_element._expr,
                inplace_indices=[0],
                sinfo_args=rx.ObjectStructInfo(),
            )
        )


class Embedding(Module):
    """
    Module for embedding layer.
    """

    def __init__(
        self,
        num: Union[int, str, tir.PrimExpr],
        dim: Union[int, str, tir.PrimExpr],
        dtype: Optional[str] = None,
    ):
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


class Attention(Module):
    """
    A cross attention layer.

    Parameters
    ----------
        query_dim : int
            The number of channels in the query.
        cross_attention_dim : Optional[int]
            The number of channels in the encoder_hidden_states.
            If not given, defaults to `query_dim`.
        heads : int
            The number of heads to use for multi-head attention.
        dim_head : int
            The number of channels in each head.
        bias : bool
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        norm_num_groups : Optional[int]
            When set, group norm is applied to the input using this number of groups.
        out_bias : bool
            Set to `True` to apply a bias to the output linear layer.
        scale_qk : bool
            Whether to apply scaling to query and key tensors.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = False,
        norm_num_groups: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
    ):
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim else query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.bias = bias
        self.norm_num_groups = norm_num_groups
        self.out_bias = out_bias
        self.scale_qk = scale_qk

        self.scale = dim_head**-0.5 if self.scale_qk else 1.0
        self.inner_dim = dim_head * heads

        self.to_q = Linear(self.query_dim, self.inner_dim, bias=self.bias)
        self.to_k = Linear(self.cross_attention_dim, self.inner_dim, bias=self.bias)
        self.to_v = Linear(self.cross_attention_dim, self.inner_dim, bias=self.bias)

        if self.norm_num_groups is not None:
            self.group_norm = GroupNorm(
                num_channels=self.query_dim, num_groups=self.norm_num_groups, affine=True
            )
        else:
            self.group_norm = None

        self.to_out = ModuleList([Linear(self.inner_dim, self.query_dim, bias=self.out_bias)])

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        **cross_attention_kwargs,
    ):
        """
        Forward method for Attention layer.

        Parameters
        ----------
        hidden_states : Tensor
            The input sample tensor.
        encoder_hidden_states : Optional[Tensor]
            Previous hidden step hidden states.
        attention_mask : Optional[Tensor]
            Mask tensor for attention, currently not supported.

        Returns
        -------
        ret : Tensor
            The output tensor for the embedding layer.
        """
        # This implementation assumes use of torch 2.0 scaled_dot_product attention.
        assert attention_mask is None, "Attention mask not yet supported."

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states, channel_axis=2, axes=[1])

        query = self.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        head_dim = int(self.inner_dim // self.heads)

        query = op.reshape(query, [0, -1, self.heads, head_dim])
        key = op.reshape(key, [0, -1, self.heads, head_dim])
        value = op.reshape(value, [0, -1, self.heads, head_dim])

        hidden_states = op.scaled_dot_product_attention(query, key, value, is_causal=False)

        # Return to proper shape.
        hidden_states = op.reshape(hidden_states, (0, -1, self.heads * head_dim))

        # Linear projection
        hidden_states = self.to_out[0](hidden_states)

        return hidden_states
