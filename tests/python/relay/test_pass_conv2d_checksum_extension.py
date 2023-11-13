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
# pylint: disable=invalid-name,too-many-locals,too-many-arguments,missing-module-docstring
# Changed by Kappes Johannes @2023

import tvm
from tvm import relay
from tvm.relay import transform


def run_opt_pass(expr, opt_pass):
    "runs the opt_pass on the expr of a function the function"
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = tvm.relay.transform.InferType()(mod)
    mod = opt_pass(mod)
    return mod["main"]




def test_single_standard_conv2d():
    """Simple testcase."""

    def before(x, w1):
        args = [x, w1]
        y = relay.nn.conv2d(x, w1, out_dtype="int32")
        return relay.Function(args, y)

    def expected(x, w1,ones_shape):
        # use a fixed order of args so alpha equal check can pass
        args = [x, w1]
        
        
        
        y0 = relay.ones(shape=ones_shape, dtype="int8") 
        y1 = relay.nn.conv2d(x, y0, padding=(0,0), groups=int(ones_shape[0]), channels=ones_shape[0], kernel_size=[ones_shape[2], ones_shape[3]], out_dtype="int32")
        y2 = relay.cast(w1, dtype="int32") 
        y3 = relay.sum(y1, axis=[0], keepdims=True) 
        y4 = relay.sum(y2, axis=[0], keepdims=True) 
        y5 = relay.nn.conv2d(y3, y4, padding=(0,0), channels=1, groups=1, kernel_size=[w1.type_annotation.shape[2], w1.type_annotation.shape[3]], out_layout="NCHW", out_dtype="int64") 
        y6 = relay.nn.conv2d(x, w1, padding=(0,0), out_dtype="int32") 
        y7 = relay.cast(y6, dtype="int64")
        y8 = relay.sum(y5, axis=[0, 1, 2, 3])
        y9 = relay.sum(y7, axis=[0, 1, 2, 3])
        y10 = relay.not_equal(y8, y9)
        y = relay.Tuple([y6, y10])
        return relay.Function(args, y)
        
        

    def check(x_shape, w_shape, ones_shape):
        x = relay.var("x", shape=x_shape, dtype="int8")
        w1 = relay.var("w1", shape=w_shape, dtype="int8")


        y_before = before(x, w1)
        y = run_opt_pass(y_before, transform.Extend2DConv())
        y = run_opt_pass(y, transform.InferType())
        y_expected = expected(x, w1, ones_shape)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        print(tvm.ir.base.get_first_structural_mismatch(y, y_expected))
        assert tvm.ir.structural_equal(y, y_expected, map_free_vars=True)

#Calculate dimension of ones tensor for Input checksum calc 
# Use ones tensor to reduce input tensor to reduced weight dimension
#P = x.type_annotation.shape[2] - w1.type_annotation.shape[2] + 1
#Q = x.type_annotation.shape[3] - w1.type_annotation.shape[3] + 1
#C = x.type_annotation.shape[1]  ones = (C,1,P,Q)

    check((1, 4, 20, 20),(1, 4, 16, 16), (4,1,5,5))
    check((1, 831, 16, 16),(1, 831, 16, 16), (831,1,1,1))
    check((5, 20, 17, 17),(2, 20, 16, 16), (20,1,2,2))
    check((5, 8, 17, 17),(2, 8, 12, 11), (8,1,6,7))


def test_single_depthwise_conv2d():
    """Simple testcase."""

    def before(x, w1):
        args = [x, w1]
        y = relay.nn.conv2d(x, w1, out_dtype="int32",groups=int(w1.type_annotation.shape[0]),channels=w1.type_annotation.shape[0])
        return relay.Function(args, y)

    def expected(x, w1,ones_shape):
        # use a fixed order of args so alpha equal check can pass
        args = [x, w1]
        
        
        
        y0 = relay.ones(shape=ones_shape, dtype="int8") 
        y1 = relay.nn.conv2d(x, y0, padding=(0,0), groups=int(ones_shape[0]), channels=ones_shape[0], kernel_size=[ones_shape[2], ones_shape[3]], out_dtype="int32")
        y2 = relay.reshape(w1, newshape=(w1.type_annotation.shape[1],w1.type_annotation.shape[0],w1.type_annotation.shape[2],w1.type_annotation.shape[3]))
        y3 = relay.sum(y1, axis=[0], keepdims=True) 
        y5 = relay.nn.conv2d(y3, y2, padding=(0,0), channels=1, groups=1, kernel_size=[w1.type_annotation.shape[2], w1.type_annotation.shape[3]], out_layout="NCHW", out_dtype="int64") 
        y6 = relay.nn.conv2d(x, w1, out_dtype="int32",groups=int(w1.type_annotation.shape[0]),channels=w1.type_annotation.shape[0])
        y7 = relay.cast(y6, dtype="int64")
        y8 = relay.sum(y5, axis=[0, 1, 2, 3])
        y9 = relay.sum(y7, axis=[0, 1, 2, 3])
        y10 = relay.not_equal(y8, y9)
        y = relay.Tuple([y6, y10])
        return relay.Function(args, y)
        
        

    

    def check(x_shape, w_shape, ones_shape):
        x = relay.var("x", shape=x_shape, dtype="int8")
        w1 = relay.var("w1", shape=w_shape, dtype="int8")


        y_before = before(x, w1)
        y = run_opt_pass(y_before, transform.Extend2DConv())
        y = run_opt_pass(y, transform.InferType())
        y_expected = expected(x, w1, ones_shape)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        print(y)
        print(y_expected)
        print(tvm.ir.base.get_first_structural_mismatch(y, y_expected))
        assert tvm.ir.structural_equal(y, y_expected, map_free_vars=True)

#Calculate dimension of ones tensor for Input checksum calc 
# Use ones tensor to reduce input tensor to reduced weight dimension
#P = x.type_annotation.shape[2] - w1.type_annotation.shape[2] + 1
#Q = x.type_annotation.shape[3] - w1.type_annotation.shape[3] + 1
#C = x.type_annotation.shape[1]  ones = (C,1,P,Q)

    check((1, 64, 56, 56),(64, 1, 3, 3), (64,1,54,54))
    check((1, 831, 16, 16),(831, 1, 16, 16), (831,1,1,1))
    check((5, 20, 17, 17),(20, 1, 16, 16), (20,1,2,2))
    check((5, 8, 17, 17),(8, 1, 12, 11), (8,1,6,7))
    check((1, 37, 56, 56),(37, 1, 11, 11), (37,1,46,46))


if __name__ == "__main__":
    test_single_standard_conv2d()
    test_single_depthwise_conv2d()
