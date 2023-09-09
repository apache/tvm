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
# pylint: disable=invalid-name, unused-argument

"""Module providing operator simplification
from Conv->bias_add->mul->add to Conv->bias_add sequence"""

import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    is_constant,
    is_op,
    # is_tuple,
    rewrite,
    wildcard,
)


class SimplifyMulAddInFunc(DFPatternCallback):
    """
    Simplify Conv->bias_add->mul->add to Conv->bias_add sequence if
    one of the inputs to Conv, bias_add, mul and add are constants.

    Replace
    def @main(%q1: Tensor[(1, 3, 224, 224), float32]) {
        %0 = nn.conv2d(%q1, meta[relay.Constant][0], padding=[3, 3, 3, 3],
                        channels=64, kernel_size=[7, 7]);
        %1 = nn.bias_add(%0, meta[relay.Constant][1]);
        %2 = multiply(%1, meta[relay.Constant][2]);
        add(%2, meta[relay.Constant][3])
    }


    with

    def @main(%q1: Tensor[(1, 3, 224, 224), float32]) {
        %0 = reshape(meta[relay.Constant][1], newshape=[64, 1, 1, 1]);
        %1 = multiply(meta[relay.Constant][0], %0);
        %2 = reshape(meta[relay.Constant][2], newshape=[64, 1, 1]);
        %3 = multiply(%2, meta[relay.Constant][1]);
        %4 = add(%3, meta[relay.Constant][3]);
        %5 = nn.conv2d(%q1, %1, padding=[3, 3, 3, 3], channels=64, kernel_size=[7, 7]);
        %6 = reshape(%4, newshape=[64]);
        nn.bias_add(%5, %6)
    }


    res[p,q,r,s] = ({SUM{i=[0,c-1], j=[0,kh-1], k=[0,kw-1]}
                    (a[p,i,r+j,s+k] * W[q,i,j,k])} + b[q]) * c1[q] + c2[q]

    res[p,q,r,s] = {SUM{i=[0,c-1], j=[0,kh-1], k=[0,kw-1]}
                    (a[p,i,r+j,s+k] * W[q,i,j,k])} * c1[q] + b[q]*c1[q] + c2[q]

    res[p,q,r,s] = Conv2d(a, W*c1) + bias_add(b*c1+c2)

    In the above, %1, %3, %4 are constants and can be folded, so we're
    left with 2 ops, as opposed to the original 4 ops
    """

    def __init__(self):
        super(SimplifyMulAddInFunc, self).__init__()
        self.inp = wildcard()
        self.weights = is_constant()
        self.conv2d_op = is_op("nn.conv2d")(self.inp, self.weights).has_attr(
            {"data_layout": "NCHW", "kernel_layout": "OIHW"}
        )
        self.bias = is_op("nn.bias_add")(self.conv2d_op, is_constant())
        self.mul = is_op("multiply")(self.bias, is_constant())
        self.pattern = is_op("add")(self.mul, is_constant())

    def reshape_tensor(self, tensor1, tensor2):
        """Function reshapes tensor1 to the shape of tensor2"""
        tensor1_shape = tvm.relay.transform.InferTypeLocal(tensor1).shape
        tensor2_shape = tvm.relay.transform.InferTypeLocal(tensor2).shape
        if len(tensor1_shape) < len(tensor2_shape):
            new_shape = []
            for i in range(len(tensor2_shape)):
                if i < len(tensor1_shape):
                    new_shape.append(tensor1_shape[i])
                else:
                    new_shape.append(1)
            new_mul = relay.reshape(tensor1, new_shape)
            return new_mul
        else:
            return tensor1

    def callback(self, pre, post, node_map):
        """Function performs transformation for Conv->bias_add->mul->add sequence,
        so that constant folding can reduce it to Conv->bias_add sequence"""
        new_mul = self.reshape_tensor(node_map[self.mul][0].args[1], node_map[self.weights][0])
        new_weights = relay.multiply((node_map[self.weights][0]), new_mul)
        new_conv2d = relay.nn.conv2d(
            (node_map[self.inp][0]), new_weights, **((node_map[self.conv2d_op][0]).attrs)
        )
        new_bias_const = self.reshape_tensor(
            node_map[self.bias][0].args[1], node_map[self.mul][0].args[1]
        )
        new_mul = relay.multiply(new_bias_const, (node_map[self.mul][0].args[1]))
        summed_const = relay.add(new_mul, (node_map[self.pattern][0].args[1]))
        final_bias_shape = tvm.relay.transform.InferTypeLocal(summed_const)
        final_bias = relay.reshape(summed_const, (final_bias_shape.shape[0].value))
        return relay.nn.bias_add(new_conv2d, final_bias)


# Right now context is ignored
@tvm.transform.module_pass(opt_level=1)
def simplify_mul_add(mod, _=None):
    """top level function for conv pattern simplification"""
    for global_var in mod.functions.keys():
        mod[global_var] = rewrite(SimplifyMulAddInFunc(), mod[global_var])
    return mod
