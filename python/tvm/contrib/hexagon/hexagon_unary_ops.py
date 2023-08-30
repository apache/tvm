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
"""Primitive Function for lut and Take Op"""
import numpy as np
from scipy import special
from typing import List
from tvm import te
from tvm.tir.function import PrimFunc

def saturate(x: te.Tensor, dtype: str):
    """Saturate value for the specified data type"""
    return te.max(te.min_value(dtype), te.min(x, te.max_value(dtype)))


def hardswish_func(x):
    """Hardswich Function"""
    x_2 = np.add(x, 3.0)
    x_2 = np.clip(x_2, 0.0, 6.0)
    return x * x_2 / 6.0


def lut_generation(inp_scale, inp_zp, out_scale, out_zp, op_name) -> List[np.uint8]:
    """Generating the Look Up Table for unary ops"""
    lut = []
    for i in range(256):
        i = np.int32(i)
        # converting the constants to the numpy value
        if inp_zp.data.shape == ():
            i_zp = inp_zp.data.numpy()[()]
        if inp_scale.data.shape == ():
            i_scale = inp_scale.data.numpy()[()]
        if out_zp.data.shape == ():
            o_zp = out_zp.data.numpy()[()]
        if out_scale.data.shape == ():
            o_scale = out_scale.data.numpy()[()]
        # Dequantization followed by computing the op value
        dequant = (i - i_zp) * i_scale
        if op_name == "tanh":
            op_val = np.tanh(dequant)
        elif op_name == "sqrt":
            op_val = np.sqrt(dequant)
        elif op_name == "rsqrt":
            op_val = 1 / np.sqrt(dequant)
        elif op_name == "exp":
            op_val = np.exp(dequant)
        elif op_name == "erf":
            op_val = special.erf(dequant)
        elif op_name == "sigmoid":
            op_val = 1 / (1 + np.exp(np.negative(dequant)))
        elif op_name == "hardswish":
            op_val = hardswish_func(dequant)
        elif op_name == "log":
            op_val = np.log(dequant)
        elif op_name == "abs":
            op_val = np.abs(dequant)
        # Quantizing the value generated and appending in the Look Up Table
        quant = np.round((op_val) / o_scale) + o_zp
        val = np.maximum(0, np.minimum(quant, 255)).astype(np.uint8)
        lut.append(val)
    return lut


def generate_take_primfunc(inp, struct_info) -> PrimFunc:
    """Generating the take op

    Parameters
    ----------
    inp : expr.Var 
        The input to be searched in the lut and whose take op needs to be returned
    
    struct_info : TensorStructInfo
        The struct info of the input data
    
    Returns
    ----------
    mod : PrimFunc
        The take op primitive function
    """
    n, h, w, c = inp.struct_info.shape
    data = te.placeholder((n, h, w, c), dtype=struct_info.dtype, name="data")
    lut_func = te.placeholder((256,), dtype="uint8", name="lut")
    take = te.compute(
        struct_info.shape,
        lambda *indices: saturate(
            (lut_func[data[indices].astype("uint8")]), struct_info.dtype
        ).astype(struct_info.dtype),
        name="take_op",
    )
    mod = te.create_prim_func([data, lut_func, take])
    return mod
