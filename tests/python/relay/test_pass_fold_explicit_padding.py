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
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass

import numpy as np


def test_simplify_conv_pad():
    convs = [relay.nn.conv1d, relay.nn.conv2d, relay.nn.conv3d]

    def validate(ndim, pad_width, pad_value, pad_mode, orig_padding, layout):
        if layout[1] == "C":
            shape = [1, 3] + [10] * ndim
            wshape = [8, 3] + [3] * ndim
        elif layout[-1] == "C":
            shape = [1] + [10] * ndim + [3]
            wshape = [8] + [3] * ndim + [3]
        else:
            raise ValueError("This test only supports NC* and N*C")

        x = relay.var("x", shape=shape, dtype="float32")
        w = relay.var("w", shape=wshape, dtype="float32")
        pad = relay.nn.pad(x, pad_width, pad_value, pad_mode)
        if layout[1] == "C":
            conv = convs[ndim - 1](pad, w, padding=orig_padding)
        else:
            conv = convs[ndim - 1](
                pad, w, padding=orig_padding, data_layout=layout, kernel_layout="DHWIO"[3 - ndim :]
            )

        if pad_mode == "constant" and pad_value == 0:
            new_padding = []
            for j in range(2):
                for i in range(len(pad_width)):
                    if layout[i] in ["D", "H", "W"]:
                        new_padding.append(pad_width[i][j])
            for i in range(len(new_padding)):
                new_padding[i] += orig_padding[i]
            if layout[1] == "C":
                after = convs[ndim - 1](x, w, padding=new_padding)
            else:
                after = convs[ndim - 1](
                    x, w, padding=new_padding, data_layout=layout, kernel_layout="DHWIO"[3 - ndim :]
                )
        else:
            after = conv

        zz = run_opt_pass(conv, transform.FoldExplicitPadding())
        expected = run_opt_pass(after, transform.InferType())
        assert tvm.ir.structural_equal(zz, expected)

        mod1 = tvm.IRModule.from_expr(conv)
        mod2 = tvm.IRModule.from_expr(zz)

        with tvm.transform.PassContext():
            ex1 = relay.create_executor("vm", mod=mod1, device=tvm.cpu(), target="llvm")
        ex2 = relay.create_executor("vm", mod=mod2, device=tvm.cpu(), target="llvm")
        x_np = np.random.rand(*shape).astype("float32")
        w_np = np.random.rand(*wshape).astype("float32")
        result1 = ex1.evaluate()(x_np, w_np)
        result2 = ex2.evaluate()(x_np, w_np)

        tvm.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-5, atol=1e-5)

    for orig_pad in [[0, 0], [2, 0], [0, 2]]:
        for i_pad in [[0, 0], [1, 1], [1, 0]]:
            for ndim in [1, 2, 3]:
                for channels_last in [0, 1]:
                    if channels_last:
                        layout = "NDHWC"
                        layout = layout[0:1] + layout[4 - ndim : 4] + layout[-1:]
                        padding = [[0, 0]] + [i_pad] * ndim + [[0, 0]]
                    else:
                        layout = "NCDHW"
                        layout = layout[0:2] + layout[5 - ndim :]
                        padding = [[0, 0]] * 2 + [i_pad] * ndim

                    validate(ndim, padding, 0, "constant", orig_pad * ndim, layout)
    ndim = 2
    validate(ndim, [[0, 0]] * 2 + [i_pad] * ndim, 1, "constant", orig_pad * ndim, "NCHW")
    validate(ndim, [[0, 0]] * 2 + [i_pad] * ndim, 0, "edge", orig_pad * ndim, "NCHW")


if __name__ == "__main__":
    test_simplify_conv_pad()
