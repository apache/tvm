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

"""Schedule for injective operators"""

import math
from textwrap import dedent
import tvm
from tvm.script import parser
import numpy as np


def lutize(out, x):
    """Takes an arbitrary python function and turns
    the operation into a LUT for that function

    func: python funcion
     eg. lambda x: sqrt(x)

    inputs: tensor

    -----
    lut:
     computes func via LUT
    """
    assert out.dtype in ("uint8")
    assert out.dtype == x.dtype

    func = eval(out.name)
    lut_np = np.arange(256, dtype=out.dtype)
    lut_np = np.nan_to_num(np.vectorize(func)(lut_np), posinf=255, neginf=0).astype(out.dtype)

    # tvm script to represent LUT
    indices_str = str(["i" + str(index) for index in range(len(out.shape))])[1:-1].replace("'", "")
    ir_func = dedent(
        f"""
    @T.prim_func
    def main(x_in: T.Buffer[{tuple(x.shape)}, "{x.dtype}"],
             compute: T.Buffer[{tuple(out.shape)}, "{out.dtype}"],
             ) -> None:
        LUT = T.allocate_const({lut_np.tolist()}, "{out.dtype}", (256,))
        lut = T.buffer_decl(shape=(256,), dtype="{out.dtype}", data=LUT)
        for {indices_str} in T.grid({str(out.shape)[1:-1]}):
            with T.block("compute"):
                compute[{indices_str}] = lut[x_in[{indices_str}]]
        """
    )
    primfunc = parser.from_source(ir_func)

    # Vectorize
    sch = tvm.tir.Schedule(primfunc)
    compute = sch.get_block("compute")
    loops = sch.get_loops(compute)
    merged = sch.fuse(*loops)
    _, split_i = sch.split(merged, [None, 128])
    sch.vectorize(split_i)

    return sch.mod["main"]
