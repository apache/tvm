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
import numpy as np
from tvm.contrib import graph_executor


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
    check((1, 831, 16, 16),(1, 831, 6, 6), (831,1,11,11))
    check((5, 20, 17, 17),(2, 20, 12, 12), (20,1,6,6))
    check((5, 8, 13, 17),(2, 8, 12, 11), (8,1,2,7))


def test_multiple_standard_conv2d():
    """More complex testcase."""

    def before(x, w1, w2):
        args = [x, w1, w2]
        x1 = relay.multiply(x, relay.const(2,dtype="int8"))
        conv = relay.nn.conv2d(x1, w1, out_dtype="int32")
        conv2 = relay.nn.conv2d(x, w2, out_dtype="int32")
        y = relay.sum(conv,axis=[0])
        mult_out = relay.Tuple([y,conv2])
        return relay.Function(args, mult_out)

    def expected(x, w1, w1_ones_shape, w2,w2_ones_shape):
        # use a fixed order of args so alpha equal check can pass
        args = [x, w1, w2]
        x1 = relay.multiply(x, relay.const(2,dtype="int8"))
        conv = relay.nn.conv2d(x1, w1, out_dtype="int32")
        y = relay.sum(conv,axis=[0])
        y0 = relay.ones(shape=w1_ones_shape, dtype="int8")
        y1 = relay.nn.conv2d(x1, y0, padding=(0,0), groups=int(w1_ones_shape[0]), channels=w1_ones_shape[0], kernel_size=[w1_ones_shape[2], w1_ones_shape[3]], out_dtype="int32")
        y2 = relay.cast(w1, dtype="int32")
        y3 = relay.sum(y1, axis=[0], keepdims=True)
        y4 = relay.sum(y2, axis=[0], keepdims=True)
        y5 = relay.nn.conv2d(y3, y4, padding=(0,0), channels=1, groups=1, kernel_size=[w1.type_annotation.shape[2], w1.type_annotation.shape[3]], out_layout="NCHW", out_dtype="int64") 
        y7 = relay.cast(conv, dtype="int64")
        y8 = relay.sum(y5, axis=[0, 1, 2, 3])
        y9 = relay.sum(y7, axis=[0, 1, 2, 3])
        y10 = relay.not_equal(y8, y9)

        conv2 = relay.nn.conv2d(x, w2, out_dtype="int32")
        convy0 = relay.ones(shape=w2_ones_shape, dtype="int8")
        convy1 = relay.nn.conv2d(x, convy0, padding=(0,0), groups=int(w2_ones_shape[0]), channels=w2_ones_shape[0], kernel_size=[w2_ones_shape[2], w2_ones_shape[3]], out_dtype="int32")
        convy2 = relay.cast(w2, dtype="int32")
        convy3 = relay.sum(convy1, axis=[0], keepdims=True)
        convy4 = relay.sum(convy2, axis=[0], keepdims=True)
        convy5 = relay.nn.conv2d(convy3, convy4, padding=(0,0), channels=1, groups=1, kernel_size=[w2.type_annotation.shape[2], w2.type_annotation.shape[3]], out_layout="NCHW", out_dtype="int64") 
        convy7 = relay.cast(conv2, dtype="int64")
        convy8 = relay.sum(convy5, axis=[0, 1, 2, 3])
        convy9 = relay.sum(convy7, axis=[0, 1, 2, 3])
        convy10 = relay.not_equal(convy8, convy9)

        mult_out = relay.Tuple([y,conv2])
        comp_output = relay.Tuple([mult_out, y10, convy10])
        return relay.Function(args, comp_output)


    def check(x_shape, w1_shape, w1_ones_shape, w2_shape, w2_ones_shape):
        x = relay.var("x", shape=x_shape, dtype="int8")
        w1 = relay.var("w1", shape=w1_shape, dtype="int8")
        w2 = relay.var("w2", shape=w2_shape, dtype="int8")

        y_before = before(x, w1, w2)
        y = run_opt_pass(y_before, transform.Extend2DConv())
        y = run_opt_pass(y, transform.InferType())
        y_expected = expected(x, w1, w1_ones_shape, w2, w2_ones_shape)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        print(tvm.ir.base.get_first_structural_mismatch(y, y_expected))
        assert tvm.ir.structural_equal(y, y_expected, map_free_vars=True)

#Calculate dimension of ones tensor for Input checksum calc
# Use ones tensor to reduce input tensor to reduced weight dimension
#P = x.type_annotation.shape[2] - w1.type_annotation.shape[2] + 1
#Q = x.type_annotation.shape[3] - w1.type_annotation.shape[3] + 1
#C = x.type_annotation.shape[1]  ones = (C,1,P,Q)

    check((1, 64, 56, 56), (1, 64, 3, 3),    (64,1,54,54),  (1, 64, 8, 8),     (64,1,49,49))
    check((1, 831, 16, 16),(1, 831, 6, 6),   (831,1,11,11), (1, 831, 10, 10),  (831,1,7,7))
    check((5, 20, 17, 17), (2, 20, 12, 12),  (20,1,6,6),    (45, 20, 5, 5),    (20,1,13,13))
    check((5, 8, 13, 17),  (2, 8, 12, 11),   (8,1,2,7),     (2, 8, 12, 11),    (8,1,2,7))



def test_mixed_conv2d():
    """Combine depthwise and standard Conv2D."""

    def before(x, w1, w2):
        args = [x, w1, w2]
        x1 = relay.multiply(x, relay.const(2,dtype="int8"))
        conv = relay.nn.conv2d(x1, w1, out_dtype="int32",groups=int(w1.type_annotation.shape[0]),channels=w1.type_annotation.shape[0])
        conv2 = relay.nn.conv2d(x, w2, out_dtype="int32")
        y = relay.sum(conv,axis=[0])
        mult_out = relay.Tuple([y,conv2])
        return relay.Function(args, mult_out)

    def expected(x, w1, w1_ones_shape, w2,w2_ones_shape):
        # use a fixed order of args so alpha equal check can pass
        args = [x, w1, w2]
        x1 = relay.multiply(x, relay.const(2,dtype="int8"))
        conv = relay.nn.conv2d(x1, w1, out_dtype="int32",groups=int(w1.type_annotation.shape[0]),channels=w1.type_annotation.shape[0])
        y = relay.sum(conv,axis=[0])

        #depthwise conv
        y0 = relay.ones(shape=w1_ones_shape, dtype="int8")
        y1 = relay.nn.conv2d(x1, y0, padding=(0,0), groups=int(w1_ones_shape[0]), channels=w1_ones_shape[0], kernel_size=[w1_ones_shape[2], w1_ones_shape[3]], out_dtype="int32")
        y2 = relay.reshape(w1, newshape=(w1.type_annotation.shape[1],w1.type_annotation.shape[0],w1.type_annotation.shape[2],w1.type_annotation.shape[3]))
        y2 = relay.cast(y2, dtype="int32")
        y3 = relay.sum(y1, axis=[0], keepdims=True)
        y5 = relay.nn.conv2d(y3, y2, padding=(0,0), channels=1, groups=1, kernel_size=[w1.type_annotation.shape[2], w1.type_annotation.shape[3]], out_layout="NCHW", out_dtype="int64")
        y7 = relay.cast(conv, dtype="int64")
        y8 = relay.sum(y5, axis=[0, 1, 2, 3])
        y9 = relay.sum(y7, axis=[0, 1, 2, 3])
        y10 = relay.not_equal(y8, y9)

        conv2 = relay.nn.conv2d(x, w2, out_dtype="int32")
        convy0 = relay.ones(shape=w2_ones_shape, dtype="int8")
        convy1 = relay.nn.conv2d(x, convy0, padding=(0,0), groups=int(w2_ones_shape[0]), channels=w2_ones_shape[0], kernel_size=[w2_ones_shape[2], w2_ones_shape[3]], out_dtype="int32")
        convy2 = relay.cast(w2, dtype="int32")
        convy3 = relay.sum(convy1, axis=[0], keepdims=True)
        convy4 = relay.sum(convy2, axis=[0], keepdims=True)
        convy5 = relay.nn.conv2d(convy3, convy4, padding=(0,0), channels=1, groups=1, kernel_size=[w2.type_annotation.shape[2], w2.type_annotation.shape[3]], out_layout="NCHW", out_dtype="int64")
        convy7 = relay.cast(conv2, dtype="int64")
        convy8 = relay.sum(convy5, axis=[0, 1, 2, 3])
        convy9 = relay.sum(convy7, axis=[0, 1, 2, 3])
        convy10 = relay.not_equal(convy8, convy9)

        mult_out = relay.Tuple([y,conv2])
        comp_output = relay.Tuple([mult_out, y10, convy10])
        return relay.Function(args, comp_output)


    def check(x_shape, w1_shape, w1_ones_shape, w2_shape, w2_ones_shape):
        x = relay.var("x", shape=x_shape, dtype="int8")
        w1 = relay.var("w1", shape=w1_shape, dtype="int8")
        w2 = relay.var("w2", shape=w2_shape, dtype="int8")

        y_before = before(x, w1, w2)
        y = run_opt_pass(y_before, transform.Extend2DConv())
        y = run_opt_pass(y, transform.InferType())
        y_expected = expected(x, w1, w1_ones_shape, w2, w2_ones_shape)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        print(tvm.ir.base.get_first_structural_mismatch(y, y_expected))
        assert tvm.ir.structural_equal(y, y_expected, map_free_vars=True)

#Calculate dimension of ones tensor for Input checksum calc
# Use ones tensor to reduce input tensor to reduced weight dimension
#P = x.type_annotation.shape[2] - w1.type_annotation.shape[2] + 1
#Q = x.type_annotation.shape[3] - w1.type_annotation.shape[3] + 1
#C = x.type_annotation.shape[1]  ones = (C,1,P,Q)
# w1=depthwise w2=standard conv2D
    #Check(    x_shape,       w1_shape,         w1_ones,       w2_shape,        w2_ones)
    check((1, 64, 56, 56), (64, 1, 3, 3),    (64,1,54,54),  (1, 64, 8, 8),     (64,1,49,49))
    check((1, 831, 16, 16),(831,1, 6, 6),    (831,1,11,11), (1, 831, 10, 10),  (831,1,7,7))
    check((1, 20, 17, 17), (20, 1, 12, 12),  (20,1,6,6),    (45, 20, 5, 5),    (20,1,13,13))
    check((1, 8, 13, 17),  (8, 1, 12, 11),   (8,1,2,7),     (2, 8, 12, 11),    (8,1,2,7))



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
        y2 = relay.cast(y2, dtype="int32")
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


def test_output():

    def before():
        x = relay.var("data", shape=(1, 37, 56, 56), dtype="int8")
        w1 = relay.var("kernel", shape=(37, 1, 11, 11), dtype="int8")
        y = relay.nn.conv2d(x, w1, out_dtype="int32",groups=int(w1.type_annotation.shape[0]),channels=w1.type_annotation.shape[0])
        return relay.Function([x,w1], y)
    
    func = before()
    func = run_opt_pass(func,transform.Extend2DConv())
    verify(func,(1, 37, 56, 56),"int8",(37, 1, 11, 11),"int8")






def verify(ref_func, data_shape, data_dtype, kernel_shape, kernel_dtype):
    def get_inputs(data_shape, data_dtype, kernel_shape, kernel_dtype):
        # Keeping inputs multiple of 4 because of a bug in Average Pool2d
        # https://discuss.tvm.apache.org/t/pool2d-gives-bad-output-for-integer-inputs/3377
        low = -128
        high = 127
        if data_dtype == "uint8":
            low = 0
            high = 255
        golden_data = np.random.randint(low=low, high=high, size=data_shape).astype(data_dtype)
        low = -128
        high = 127
        if kernel_dtype == "uint8":
            low = 0
            high = 255
        golden_weight = np.random.randint(low=low, high=high, size=kernel_shape).astype(
            kernel_dtype
        )
        return (golden_data, golden_weight)

    def get_output(func, golden_inputs):
        with tvm.transform.PassContext(opt_level=2):
            golden_data, golden_weight = golden_inputs
            params = {"kernel": golden_weight}
            graph, lib, params = relay.build(func, "llvm", params=params)
            mod = graph_executor.create(graph, lib, device=tvm.cpu(0))
            mod.set_input("data", golden_data)
            mod.set_input(**params)
            mod.run()
            res = mod.get_output(0).numpy()
            return res

    golden_inputs = get_inputs(data_shape, data_dtype, kernel_shape, kernel_dtype)
    golden_output = get_output(ref_func, golden_inputs)
    print(golden_output)
    #qnn_output = get_output(qnn_func, golden_inputs)
    #np.testing.assert_equal(qnn_output, golden_output)




if __name__ == "__main__":
    test_single_standard_conv2d()
    test_single_depthwise_conv2d()
    test_multiple_standard_conv2d()
    test_mixed_conv2d()
    test_output()