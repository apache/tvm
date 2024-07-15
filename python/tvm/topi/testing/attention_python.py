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

"""Attention operator in python"""
from typing import Optional
import numpy as np
from .softmax_python import softmax_python


def attention_python(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    bias: Optional[np.ndarray],
    qk_scale: float,
    causal: str,
    window_size: Optional[int] = None,
    layout: str = "BSNH",
):  # pylint: disable=too-many-arguments, too-many-locals, invalid-name
    """Attention operator in python

    Parameters
    ----------
    q : np.ndarray
        Query tensor with shape [batch, seq_length, num_heads, head_dim] in the layout specified by
        `layout`.
    k : np.ndarray
        Key tensor with shape [batch, seq_length_kv, num_kv_heads, head_dim] in the layout specified
        by `layout`.
    v : np.ndarray
        Value tensor with shape [batch, seq_length_kv, num_kv_heads, head_dim_v] in the layout
        specified by `layout`.
    bias : np.ndarray
        Bias tensor with shape [batch, num_heads, seq_length, seq_length]
    qk_scale : float
        Scale factor for the query-key product.
    causal : str
        The type of causal mask to apply. Can be "none", "TopLeft", or "BottomRight".
    window_size : Optional[int]
        The window size for the causal mask.
    layout : str
        The layout of the input tensors, e.g. "BSNH" or "BNSH".

    Returns
    -------
    np.ndarray
        The output tensor with shape [batch, seq_length, num_heads, head_dim_v] in the layout
        specified by `layout`.
    """
    assert layout in ["BSNH", "BNSH", "SBNH"]

    dim_b = layout.find("B")
    dim_s = layout.find("S")
    dim_n = layout.find("N")
    dim_h = layout.find("H")

    q = q.transpose(dim_b, dim_n, dim_s, dim_h)  # b, n, s, h
    k = k.transpose(dim_b, dim_n, dim_s, dim_h)  # b, n, s_kv, h
    kt = k.transpose(0, 1, 3, 2)  # b, n, h, s_kv
    v = v.transpose(dim_b, dim_n, dim_s, dim_h)

    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    s = q.shape[2]
    s_kv = k.shape[2]

    if num_heads != num_kv_heads:
        assert num_heads % num_kv_heads == 0
        factor = num_heads // num_kv_heads
        kt = np.repeat(kt, factor, axis=1)
        v = np.repeat(v, factor, axis=1)

    if not qk_scale == "none":
        score = q @ kt * qk_scale  # b, n, s, s_kv
    else:
        score = q @ kt / np.sqrt(q.shape[-1])  # b, n, s, s_kv
    if bias is not None:
        score = score + bias  # b, n, s, s_kv
    if causal == "none":
        attn = softmax_python(score, -1)
    else:
        if causal == "TopLeft":
            offset = 0
        elif causal == "BottomRight":
            offset = abs(s - s_kv)
        else:
            raise ValueError(f"Unsupported causal type: {causal}")
        score_masked = np.tril(score, k=offset)

        if window_size:
            score_masked = np.triu(
                score_masked, -window_size + 1  # pylint: disable=invalid-unary-operand-type
            )

        score_masked_exp = np.tril(
            np.exp(score_masked - np.max(score_masked, axis=-1, keepdims=True)), k=offset
        )

        if window_size:
            score_masked_exp = np.triu(
                score_masked_exp, -window_size + 1  # pylint: disable=invalid-unary-operand-type
            )

        score_masked_sum = np.sum(score_masked_exp, axis=-1, keepdims=True)
        attn = np.divide(score_masked_exp, score_masked_sum)

    out = attn @ v  # b, n, s, h_v
    return out.transpose(*np.argsort([dim_b, dim_n, dim_s, dim_h]).tolist())
