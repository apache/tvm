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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Dense alter op functions for ARM"""

import tvm
from tvm import relay
from .. import nn
from ..nn import dense_alter_layout


def check_vrmpy_applicable(x, y):
    return (
        "int8" in x.dtype and "int8" in y.dtype and y.shape[-2] % 32 == 0 and y.shape[-1] % 4 == 0
    )


@dense_alter_layout.register(["hexagon"])
def _alter_dense_layout(attrs, inputs, tinfos, out_type):
    data_tensor, weight_tensor = tinfos
    out_dtype = out_type.dtype

    if check_vrmpy_applicable(data_tensor, weight_tensor):
        weight_layout = "NC32n4c"
        return relay.nn.contrib_dense_pack(inputs[0], inputs[1], weight_layout, None, out_dtype)
    else:
        return None


def vrmpy_legalize(x, w, arg_types, op, attrs):
    """
    Legalizes int8 inputs to dense for vrmpy.
    X'_u8 = X_s8 + 128
    X_s8 * W_s8 = (X'_u8 - 128) * (W'_u8 - 128)
                = X'_u8 * W'_u8 - X'_u8 * 128 - 128 * W'_u8 + 128 * 128
    X_u8 * W_s8 = X_u8 * (W'_u8 - 128)
                = X'_u8 * W'_u8 - X_u8 * 128
    """
    if not check_vrmpy_applicable(arg_types[0], arg_types[1]):
        return None

    def cast_to_uint8(x):
        x = relay.cast(x, "int32")
        x = relay.add(x, relay.const(128, "int32"))
        return relay.cast(x, "uint8")

    if arg_types[0].dtype == "int8" and arg_types[1].dtype == "int8":
        x = cast_to_uint8(x)
        w = cast_to_uint8(w)

        W_u8x128 = relay.const(-128, "int32") * relay.sum(relay.cast(w, "int32"), axis=[-1])
        X_u8x128 = relay.const(-128, "int32") * relay.sum(relay.cast(x, "int32"), axis=[-1])
        X_u8x128 = relay.expand_dims(X_u8x128, axis=1)

        out = op(x, w, **attrs)

        out += W_u8x128
        out += X_u8x128

        k_dim = int(arg_types[0].shape[-1])
        return out + relay.const(128 * 128 * k_dim, "int32")

    if arg_types[0].dtype == "uint8" and arg_types[1].dtype == "int8":
        w = cast_to_uint8(w)

        X_u8x128 = relay.expand_dims(
            relay.const(-128, "int32") * relay.sum(relay.cast(x, "int32"), axis=[-1]), axis=1
        )

        out = op(x, w, **attrs)

        return out + X_u8x128

    return None


@nn.dense_legalize.register("hexagon")
def _dense_legalize(attrs, inputs, arg_types):
    """Legalize dense op for HVX vectorization and vrmpy tensorization.

    Given a workload with a matrix X of shape (M, K) and a matrix Y of (N, K),
    we first pad the N dimension to be a multiple of the output vector length.

    And if the inputs are signed or unsigned int8 and the Y matrix can be packed into the
    NK32n4k layout, we convert both inputs to uint8 to apply the most efficient variant of vrmpy.
    """
    new_attrs = {k: attrs[k] for k in attrs.keys()}
    # Collect the input tensors.
    x_tensor, y_tensor = arg_types[0], arg_types[1]
    dtype = x_tensor.dtype

    # Collect the output tensor.
    output_tensor = arg_types[2]

    # Collect the input exprs.
    x, y = inputs

    N, _ = y_tensor.shape

    if dtype == "float16":
        vec_len = 64
    elif "int8" in dtype:
        vec_len = 32
    else:
        return None

    if N % vec_len != 0:
        N_padded = ((N + vec_len) // vec_len) * vec_len
        dn = N_padded - N

        y_ = relay.nn.pad(y, pad_width=((0, dn), (0, 0)))

        # If units is explicitly specified, it is used to compute the output shape.
        # We need to update units after padding to prevent a type error.
        if attrs["units"] is not None:
            new_attrs["units"] = N + dn

        arg_types = [
            arg_types[0],
            tvm.ir.tensor_type.TensorType([N + dn, arg_types[1].shape[1]], arg_types[1].dtype),
        ]

        vrmpy_out = vrmpy_legalize(x, y_, arg_types, relay.nn.dense, new_attrs)

        if vrmpy_out is None:
            out_ = relay.nn.dense(x, y_, **new_attrs)
        else:
            out_ = vrmpy_out

        out = relay.strided_slice(out_, begin=[0, 0], end=[x.value for x in output_tensor.shape])
        return out

    return vrmpy_legalize(inputs[0], inputs[1], arg_types, relay.nn.dense, attrs)
