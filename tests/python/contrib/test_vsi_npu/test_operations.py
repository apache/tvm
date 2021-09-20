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
#from python.tvm.relay.qnn.op.qnn import dequantize
import tvm
from tvm import relay
from tvm.relay import testing
import numpy as np
from infrastructure import verify_vsi_result

def _single_operation_test(relay_nn_func, dtype, data_shape, out_shape,
        *args, **kargs):
    op_name = relay_nn_func.__name__.upper()
    print("Testing {0: <50}".format(op_name), end="")
    data = relay.var("data", shape=data_shape, dtype=dtype)

    out = relay_nn_func(data, *args, **kargs)

    args = relay.analysis.free_vars(out)
    net = relay.Function(args, out)

    mod, params = relay.testing.init.create_workload(net)

    verify_vsi_result(mod, params, data_shape, out_shape, dtype)

def test_global_avg_pool2d():
    func = relay.nn.global_avg_pool2d

    dtype = "float32"
    data_shape = (1, 20, 12, 9)
    out_shape = (1, 20, 1, 1)
    _single_operation_test(func, dtype, data_shape, out_shape)

def test_global_max_pool2d():
    func = relay.nn.global_max_pool2d

    dtype = "float32"
    data_shape = (1, 20, 12, 9)
    out_shape = (1, 20, 1, 1)
    _single_operation_test(func, dtype, data_shape, out_shape)

def test_avg_pool2d():
    func = relay.nn.avg_pool2d

    dtype = "float32"
    data_shape = (1, 1, 20, 20)
    out_shape = (1, 1, 10, 10)
    _single_operation_test(func, dtype, data_shape, out_shape, pool_size=(3, 3),
            strides=(2, 2), padding=(1,1,1,1))

def test_max_pool2d():
    func = relay.nn.max_pool2d

    dtype = "float32"
    data_shape = (1, 1, 20, 20)
    out_shape = (1, 1, 10, 10)
    _single_operation_test(func, dtype, data_shape, out_shape, pool_size=(2, 2),
            strides=(2, 2), padding=(0,0,0,0))

def test_softmax():
    func = relay.nn.softmax

    dtype = "float32"
    data_shape = (1, 20, 12, 9)
    out_shape = data_shape
    _single_operation_test(func, dtype, data_shape, out_shape)

def test_relu():
    func = relay.nn.relu

    dtype = "float32"
    data_shape = (1, 20, 12, 9)
    out_shape = data_shape
    _single_operation_test(func, dtype, data_shape, out_shape)

def test_add():
    dtype = "float32"
    data_shape = (1, 20, 12, 9)
    out_shape = data_shape

    def get_workload(data_shape, dtype="float32"):
        '''customized keywords(like data0,data1...) are not supported in \
        relay.testing.init.create_workload
        '''
        data0 = relay.var("data", shape=data_shape, dtype=dtype)
        data1 = relay.var("weight", shape=data_shape, dtype=dtype)

        out = relay.add(data0, data1)

        args = relay.analysis.free_vars(out)
        net = relay.Function(args, out)

        return relay.testing.init.create_workload(net)

    print("Testing {0: <50}".format("ADD"), end="")
    mod, params = get_workload(data_shape, dtype)
    verify_vsi_result(mod, params, data_shape, out_shape, dtype)

def test_batch_flatten():
    func = relay.nn.batch_flatten

    dtype = "float32"
    data_shape = (1, 5, 10, 10)
    out_shape = (1, 500)
    _single_operation_test(func, dtype, data_shape, out_shape)

def test_batch_norm():
    data_shape = (1, 4)
    c_shape = (4,)
    out_shape = (1, 4)
    dtype="float32"

    def get_workload(data_shape, weight_shape, dtype="float32"):
        data = relay.var("data", shape=data_shape, dtype=dtype)

        w = tvm.nd.array(np.ones(weight_shape, dtype))
        gamma = relay.const(w, dtype)
        beta = relay.const(w, dtype)
        moving_mean = relay.const(w, dtype)
        moving_var = relay.const(w, dtype)

        bn = relay.nn.batch_norm(data, gamma, beta, moving_mean, moving_var)
        out = bn[0]

        args = relay.analysis.free_vars(out)
        net = relay.Function(args, out)

        return relay.testing.init.create_workload(net)

    print("Testing {0: <50}".format("BATCH_NORM"), end="")
    mod, params = get_workload(data_shape, c_shape, dtype)
    verify_vsi_result(mod, params, data_shape, out_shape, dtype)

def test_conv2d():
    data_shape = (1, 256, 64, 64)
    weight_shape = (256, 256, 3, 3)
    out_shape = (1, 256, 64, 64)
    dtype="float32"
    Pad=(1,1,1,1)
    Strides=(1,1)
    Dilation=(1,1)
    Ksize=(3,3)
    Groups=1

    def get_workload(data_shape, weight_shape, dtype="float32"):
        """Function to construct a MobileNet"""
        data = relay.var("data", shape=data_shape, dtype=dtype)
        weight = relay.var("conv_weight")
        conv = relay.nn.conv2d(
            data,
            weight,
            channels=weight_shape[0],
            kernel_size=Ksize,
            strides=Strides,
            padding=Pad,
            groups=Groups,
            data_layout="NCHW",
            kernel_layout="OIHW"
        )
        args = relay.analysis.free_vars(conv)
        net = relay.Function(args, conv)
        return relay.testing.init.create_workload(net)

    print("Testing {0: <50}".format("CONV2D"), end="")
    mod, params = get_workload(data_shape, weight_shape, dtype)
    verify_vsi_result(mod, params, data_shape, out_shape, dtype)


def test_dense():
    data_shape = (1, 784)
    weight_shape = (128, 784)
    out_shape = (1, 128)
    dtype="float32"

    def get_workload(data_shape, weight_shape, dtype="float32"):
        data = relay.var("data", shape=data_shape, dtype=dtype)
        fc1 = relay.nn.dense(data, relay.var("fc1_weight"), units=weight_shape[0])
        fc = relay.nn.bias_add(fc1, relay.var("fc1_bias"), axis=1)
        args = relay.analysis.free_vars(fc)
        net = relay.Function(args, fc)

        return relay.testing.init.create_workload(net)

    print("Testing {0: <50}".format("DENSE_BIAS_ADD"), end="")
    mod, params = get_workload(data_shape, weight_shape, dtype)
    verify_vsi_result(mod, params, data_shape, out_shape, dtype)

def test_concatenate():
    dtype = "float32"
    data_shape = (1, 20, 12)
    out_shape = (1, 20, 24)

    def get_workload(data_shape, dtype="float32"):
        '''customized keywords(like data0,data1...) are not supported in \
        relay.testing.init.create_workload
        '''
        data0 = relay.var("data", shape=data_shape, dtype=dtype)
        data1 = relay.var("bias", shape=data_shape, dtype=dtype)

        out = relay.concatenate([data0, data1], axis=-1)
        args = relay.analysis.free_vars(out)
        net = relay.Function(args, out)

        return relay.testing.init.create_workload(net)

    print("Testing {0: <50}".format("CONCATENATE"), end="")
    mod, params = get_workload(data_shape, dtype)
    verify_vsi_result(mod, params, data_shape, out_shape, dtype)

def test_image_resize():
    data_dtype = 'float32'
    in_size = (33, 33)
    out_size = (257, 257)
    channel = (21,)

    def run_test(layout, method, mode):
        if (layout == "NHWC"):
            data_shape = (1,) + in_size + channel
            out_shape = (1,) + out_size + channel
        else:
            data_shape = (1,) + channel + in_size
            out_shape = (1,) + channel + out_size
        x = relay.var("data", shape=data_shape, dtype=data_dtype)
        out = relay.image.resize(x, out_size, layout, method, mode)
        args = relay.analysis.free_vars(out)
        net = relay.Function(args, out)

        mod, params = relay.testing.init.create_workload(net)
        verify_vsi_result(mod, params, data_shape, out_shape, data_dtype)
    print("Testing {0: <50}".format("RESIZE_1"), end="")
    run_test("NHWC", "nearest_neighbor", "asymmetric")
    print("Testing {0: <50}".format("RESIZE_2"), end="")
    run_test("NHWC", "bilinear", "asymmetric")
    print("Testing {0: <50}".format("RESIZE_3"), end="")
    run_test("NHWC", "bilinear", "half_pixel")
    print("Testing {0: <50}".format("RESIZE_4"), end="")
    run_test("NHWC", "bilinear", "align_corners")
    print("Testing {0: <50}".format("RESIZE_5"), end="")
    run_test("NCHW", "nearest_neighbor", "asymmetric")
    print("Testing {0: <50}".format("RESIZE_6"), end="")
    run_test("NCHW", "bilinear", "asymmetric")
    print("Testing {0: <50}".format("RESIZE_7"), end="")
    run_test("NCHW", "bilinear", "half_pixel")
    print("Testing {0: <50}".format("RESIZE_8"), end="")
    run_test("NCHW", "bilinear", "align_corners")

def test_dropout():
    func = relay.nn.dropout

    dtype = "float32"
    data_shape = (1, 20, 12, 9)
    out_shape = data_shape
    _single_operation_test(func, dtype, data_shape, out_shape, rate=0.5)

def test_qnn_add():
    data_dtype = "uint8"
    data_shape = (1, 96, 96, 64)
    data2_shape = (64,)
    out_shape = (1, 96, 96, 64)

    x = relay.var("x", shape=data_shape, dtype=data_dtype)
    y = relay.var("y", shape=data2_shape, dtype=data_dtype)
    out = relay.qnn.op.add(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(0.020283, "float32"),
        lhs_zero_point=relay.const(112, "int32"),
        rhs_scale=relay.const(0.000316, "float32"),
        rhs_zero_point=relay.const(119, "int32"),
        output_scale=relay.const(0.020144, "float32"),
        output_zero_point=relay.const(112, "int32"),
    )

    print("Testing {0: <50}".format("QNN.ADD"), end="")
    inputs = {
        "x": tvm.nd.array(np.random.randint(1, high=101, size=data_shape, dtype="uint8")),
    }
    params = {
        "y": tvm.nd.array(np.random.randint(1, high=101, size=data_shape, dtype="uint8")),
    }
    verify_vsi_result(inputs, out,params, data_shape, out_shape, data_dtype)

def test_float_add():
    dtype = "float32"
    data_0_shape = (1,7 ,7 , 768)
    data_1_shape = (1, 1, 1, 768)
    out_shape = data_0_shape
    data0 = relay.var("a", shape=data_0_shape, dtype=dtype)
    data1 = relay.var("b", shape=data_1_shape, dtype=dtype)

    out = relay.op.add(lhs=data0, rhs=data1)

    print("Testing {0: <50}".format("ADD"), end="")
    inputs = {
        "a": tvm.nd.array(np.random.uniform(size=data_0_shape).astype(dtype)),
        #"b": tvm.nd.array(np.random.uniform(size=data_1_shape).astype(dtype)),
    }
    params = {
        #"weight": tvm.nd.array(np.ones(weight_shape,dtype)),
        "b": tvm.nd.array(np.random.uniform(size=data_1_shape).astype(dtype)),
    }
    verify_vsi_result(inputs, out, params, data_0_shape, out_shape, dtype)

def test_float_relu():
    dtype ="float32"
    data_shape = (2, 2, 2, 2)
    out_shape = data_shape

    data = relay.var("data", shape=data_shape, dtype=dtype)
    out = relay.op.nn.relu(data)
    inputs = {
        "data": tvm.nd.array(np.random.uniform(-1.0, 1.0, size=data_shape).astype(dtype)),
    }
    print("Testing {0: <50}".format("RELU"), end="")
    verify_vsi_result(inputs, out, [], data_shape, out_shape, dtype)

def test_uint8_relu():
    input_dtype = "float32"
    output_dtype = "uint8"
    temp_dtype = "float32"
    data_shape = (1,100)
    data = relay.var("data", shape=data_shape, dtype=input_dtype)

    scale = 0.15294
    zero_point = 128
    quantize = lambda x: float(int(round(x / scale)) + zero_point)
    qmax = float(tvm.tir.op.max_value("uint8").value)

    quant = relay.qnn.op.quantize(data,
                            output_scale=relay.const(0.15294, "float32"),
                            output_zero_point=relay.const(128, "int32"),
                            axis = -1,
                            out_dtype=output_dtype
                            )
    op = relay.clip(quant, quantize(0), qmax)

    requantize_params = {
            "input_scale": relay.const(0.15294, "float32"),
            "input_zero_point": relay.const(128, "int32"),
            "output_scale": relay.const(0.15294, "float32"),
            "output_zero_point": relay.const(128, "int32"),
            "out_dtype":output_dtype,
        }

    requantize = relay.qnn.op.requantize(op,**requantize_params)

    inputs = {
        "data": tvm.nd.array(np.random.uniform(-4, 4, size=data_shape).astype(input_dtype)),
    }
    print("Testing {0: <50}".format("RELU"), end="")
    verify_vsi_result(inputs, requantize, [], data_shape, data_shape, output_dtype)

def test_float_leaky_relu():
    dtype ="float32"
    data_shape = (2, 2, 2, 2)
    out_shape = data_shape

    data = relay.var("data", shape=data_shape, dtype=dtype)
    alpha = 0.1

    out = relay.op.nn.leaky_relu(data, alpha)
    inputs = {
        "data": tvm.nd.array(np.random.uniform(-1.0, 1.0, size=data_shape).astype(dtype)),
    }
    print("Testing {0: <50}".format("LEAKY RELU"), end="")
    verify_vsi_result(inputs, out, [], data_shape, out_shape, dtype)

def test_uint8_leaky_relu():
    input_dtype = "uint8"
    output_dtype = input_dtype
    temp_dtype = "float32"
    data_shape = (1,100)
    data = relay.var("data", shape=data_shape, dtype=input_dtype)
    alpha = 0.1

    dequantize_op = relay.qnn.op.dequantize(data,
                            input_zero_point=relay.const(128, "int32"),
                            input_scale=relay.const(0.15294, "float32"),
                            axis = -1,
                            )
    op = relay.op.nn.leaky_relu(dequantize_op, alpha)

    quantize = relay.qnn.op.quantize(op,
                            output_scale=relay.const(0.15294, "float32"),
                            output_zero_point=relay.const(128, "int32"),
                            axis = -1,
                            out_dtype=output_dtype
                            )
    inputs = {
        "data": tvm.nd.array(np.random.uniform(0, 255, size=data_shape).astype(input_dtype)),
    }
    print("Testing {0: <50}".format("LEAKY RELU"), end="")
    verify_vsi_result(inputs, quantize, [], data_shape, data_shape, output_dtype)


def test_float_softmax():
    #func = relay.nn.softmax

    dtype = "float32"
    data_shape = (1,100)
    out_shape = data_shape
    axis = 1
    data = relay.var("data", shape=data_shape, dtype=dtype)
    out = relay.op.nn.softmax(data,axis)
    inputs = {
        "data": tvm.nd.array(np.random.uniform(size=data_shape).astype(dtype)),
        #"data": tvm.nd.array(np.arange(1000).reshape(data_shape).astype(dtype)),
    }
    print("Testing {0: <50}".format("SOFTMAX"), end="")
    verify_vsi_result(inputs, out, [], data_shape, out_shape, dtype)

def test_float32_conv2d():
    data_shape = (1, 2, 5, 5)
    weight_shape = (2, 2, 3, 3)
    out_shape = (1, 2, 3, 3)
    dtype="float32"
    Pad=(0,0,0,0)
    Strides=(1,1)
    Dilation=(1,1)
    Ksize=(3,3)
    Groups=1

    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var("weight")
    out = relay.nn.conv2d(
            data,
            weight,
            channels=weight_shape[0],
            kernel_size=Ksize,
            strides=Strides,
            padding=Pad,
            groups=Groups,
            data_layout="NCHW",
            kernel_layout="OIHW"
        )
    inputs = {
        "data": tvm.nd.array(np.arange(50).reshape(data_shape).astype(dtype)),
    }

    params = {
        "weight": tvm.nd.array(np.ones(weight_shape,dtype)),
    }
    print("Testing {0: <50}".format("CONV2D"), end="")
    verify_vsi_result(inputs, out, params, data_shape, out_shape, dtype)

def test_float32_conv2d_permute():
    data_shape = (1, 4, 4, 4)
    weight_shape = (3, 3, 4, 5)
    out_shape = (1, 2, 2, 5)
    dtype="float32"
    Pad=(0,0,1,1)
    Strides=(2,2)
    Dilation=(1,1)
    Ksize=(3,3)
    Groups=1

    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var("weight",shape=weight_shape,dtype=dtype)
    out = relay.nn.conv2d(
            data,
            weight,
            channels=weight_shape[3],
            padding=Pad,
            kernel_size=Ksize,
            strides=Strides,
            groups=Groups,
            data_layout="NHWC",
            kernel_layout="HWIO"
        )
    inputs = {
        #"data": tvm.nd.array(np.ones(data_shape).astype(dtype)),
        #"data": tvm.nd.array(np.arange(1*4*4*4).reshape(data_shape).astype(dtype)),
        "data": tvm.nd.array(np.random.uniform(size=data_shape).astype(dtype)),
    }

    params = {
        #"weight": tvm.nd.array(np.arange(3*4*3*5).reshape(weight_shape).astype(dtype)),
        "weight": tvm.nd.array(np.random.uniform(size=weight_shape).astype(dtype)),
    }
    print("Testing {0: <50}".format("CONV2D"), end="")
    verify_vsi_result(inputs, out, params, data_shape, out_shape, dtype)

def test_float32_depthwise_conv2d_permute():
    data_shape = (1, 28, 28, 192)
    weight_shape = (3, 3, 192, 1)
    out_shape = (1, 14, 14, 192)
    dtype="float32"
    Pad=(0,0,1,1)
    Strides=(2,2)
    Dilation=(1,1)
    Ksize=(3,3)
    Groups=192

    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var("conv_weight",shape=weight_shape,dtype=dtype)
    out = relay.nn.conv2d(
            data,
            weight,
            channels=Groups,
            padding=Pad,
            kernel_size=Ksize,
            strides=Strides,
            groups=Groups,
            data_layout="NHWC",
            kernel_layout="HWOI"
        )
    inputs = {
        #"data": tvm.nd.array(np.ones(data_shape,dtype)),
        "data": tvm.nd.array(np.arange(data_shape[1]*data_shape[2]*data_shape[3]).reshape(data_shape).astype(dtype)),
        #"weight": tvm.nd.array(np.ones(weight_shape,dtype)),
    }
    params = {
        "conv_weight": tvm.nd.array(np.random.uniform(size=weight_shape).astype(dtype)),
    }
    print("Testing {0: <50}".format("CONV2D"), end="")
    verify_vsi_result(inputs, out, params, data_shape, out_shape, dtype)

def test_float_reshape():
    data_dtype = "float32"
    data_shape = (1,1,1,1000)
    out_shape = (1,1000)
    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    out = relay.op.reshape(data,out_shape)
    inputs = {
        "data": tvm.nd.array(np.ones(data_shape,data_dtype)),
    }
    print("Testing {0: <50}".format("RESHAPE"), end="")
    verify_vsi_result(inputs, out, [], data_shape, out_shape, data_dtype)

def test_float_tranpose():
    data_dtype = "float32"
    data_shape = (1,1,192,256)
    out_shape = (256,192,1,1)
    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    out = relay.op.transpose(data,(3,2,1,0))
    inputs = {
        "data": tvm.nd.array(np.arange(data_shape[0]*data_shape[1]*data_shape[2]*data_shape[3]).reshape(data_shape).astype(data_dtype)),
    }
    print("Testing {0: <50}".format("TRANSPOSE"), end="")
    verify_vsi_result(inputs, out, [], data_shape, out_shape, data_dtype)

def test_uint8_tranpose():
    data_dtype = "uint8"
    data_shape = (1,1,192,256)
    out_shape = (256,192,1,1)
    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    out = relay.op.transpose(data,(3,2,1,0))
    inputs = {
        "data": tvm.nd.array(np.arange(data_shape[0]*data_shape[1]*data_shape[2]*data_shape[3]).reshape(data_shape).astype(data_dtype)),
    }
    print("Testing {0: <50}".format("TRANSPOSE"), end="")
    verify_vsi_result(inputs, out, [], data_shape, out_shape, data_dtype)

def test_float_relu6():
    dtype = "float32"
    data_shape = (1, 2, 1, 1)
    out_shape = data_shape

    data = relay.var("data", shape=data_shape, dtype=dtype)
    out = relay.clip(data, 0, 6)
    inputs = {
        "data": tvm.nd.array(np.random.uniform(-1, 1, size=data_shape).astype(dtype)),
    }
    print("Testing {0: <50}".format("RELU6"), end="")
    verify_vsi_result(inputs, out, [], data_shape, out_shape, dtype)

def test_uint8_relu6():
    input_dtype = "float32"
    output_dtype = "uint8"
    temp_dtype = "float32"
    data_shape = (1,100)
    data = relay.var("data", shape=data_shape, dtype=input_dtype)

    scale = 0.15294
    zero_point = 128
    quantize = lambda x: float(int(round(x / scale)) + zero_point)

    quant = relay.qnn.op.quantize(data,
                            output_scale=relay.const(0.15294, "float32"),
                            output_zero_point=relay.const(128, "int32"),
                            axis = -1,
                            out_dtype=output_dtype
                            )
    op = relay.clip(quant, quantize(0.0), quantize(6.0))

    requantize_params = {
            "input_scale": relay.const(0.15294, "float32"),
            "input_zero_point": relay.const(128, "int32"),
            "output_scale": relay.const(0.15294, "float32"),
            "output_zero_point": relay.const(128, "int32"),
            "out_dtype":output_dtype,
        }

    requantize = relay.qnn.op.requantize(op,**requantize_params)

    inputs = {
        "data": tvm.nd.array(np.random.uniform(-4, 4, size=data_shape).astype(input_dtype)),
    }
    print("Testing {0: <50}".format("RELU"), end="")
    verify_vsi_result(inputs, requantize, [], data_shape, data_shape, output_dtype)


def test_sample_model():
    conv_data_shape = (1, 224, 224, 3)
    weight_shape = (3, 3, 3, 1)
    reshape_data_shape = (1, 112, 112, 1)
    softmax_data_shape = (1,12544)

    #conv
    dtype="float32"
    Pad=(0,0,1,1)
    Strides=(2,2)
    Dilation=(1,1)
    Ksize=(3,3)
    Groups=1

    data = relay.var("data", shape=conv_data_shape, dtype=dtype)
    weight = relay.var("weight",shape=weight_shape,dtype=dtype)
    conv = relay.nn.conv2d(
            data,
            weight,
            channels=weight_shape[3],
            padding=Pad,
            kernel_size=Ksize,
            strides=Strides,
            groups=Groups,
            data_layout="NHWC",
            kernel_layout="HWIO"
        )
    inputs = {
        "data": tvm.nd.array(np.random.uniform(size=conv_data_shape).astype(dtype)),
    }

    reshape = relay.op.reshape(conv,softmax_data_shape)
    softmax = relay.op.nn.softmax(reshape,1)
    params = {
        "weight": tvm.nd.array(np.random.uniform(size=weight_shape).astype(dtype)),
    }
    verify_vsi_result(inputs, softmax, params, conv_data_shape, softmax_data_shape, dtype)

def test_quantize():
    input_dtype = "float32"
    output_dtype = "uint8"
    data_shape = (1, 2, 2, 3)
    out_shape = data_shape
    scale=relay.const(0.1, input_dtype),
    zero_point=relay.const(125, "int32"),

    data = relay.var("data", shape=data_shape, dtype=input_dtype)
    out = relay.qnn.op.quantize(data,
                            output_scale=relay.const(0.00784314, input_dtype),
                            output_zero_point=relay.const(127, "int32"),
                            axis = -1,
                            out_dtype=output_dtype
                            )
    inputs = {
        "data": tvm.nd.array(np.random.uniform(size=data_shape).astype(input_dtype)),
    }
    print("Testing {0: <50}".format("QUANTIZE"), end="")
    verify_vsi_result(inputs, out, [], data_shape, out_shape, output_dtype)

def test_dequantize():
    input_dtype = "uint8"
    output_dtype = "float32"
    data_shape = (1, 2, 2, 3)
    out_shape = data_shape
    scale=relay.const(0.1, input_dtype),
    zero_point=relay.const(125, "int32"),

    data = relay.var("data", shape=data_shape, dtype=input_dtype)
    out = relay.qnn.op.dequantize(data,
                            input_zero_point=relay.const(127, "int32"),
                            input_scale=relay.const(0.00784314, output_dtype),
                            axis = -1,
                            )
    inputs = {
        "data": tvm.nd.array(np.random.uniform(0,10,size=data_shape).astype(input_dtype)),
    }
    print("Testing {0: <50}".format("QUANTIZE"), end="")
    verify_vsi_result(inputs, out, [], data_shape, out_shape, output_dtype)

def test_float_avg_pool():
    dtype = "float32"
    data_shape = (1, 7, 7, 768)
    out_shape = (1, 1, 1, 768)

    data = relay.var("data", shape=data_shape, dtype=dtype)
    out = relay.op.nn.avg_pool2d(data,pool_size=(7, 7),strides=(2, 2),layout="NHWC")
    inputs = {
        "data": tvm.nd.array(np.arange(7*7*768).reshape(data_shape).astype(dtype)),
    }
    print("Testing {0: <50}".format("AVG_POOL_2D"), end="")
    verify_vsi_result(inputs, out, [], data_shape, out_shape, dtype)

def test_float32_pattern():
    data_shape = (1, 4, 4, 4)
    weight_shape = (3, 3, 4, 5)
    add_shape = (5,)
    out_shape = (1,2,2,5)
    dtype="float32"
    Pad=(0,0,1,1)
    Strides=(2,2)
    Dilation=(1,1)
    Ksize=(3,3)
    Groups=1

    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var("weight",shape=weight_shape,dtype=dtype)
    add_data = relay.var("add", shape=add_shape, dtype=dtype)
    conv = relay.nn.conv2d(
            data,
            weight,
            channels=weight_shape[3],
            padding=Pad,
            kernel_size=Ksize,
            strides=Strides,
            groups=Groups,
            data_layout="NHWC",
            kernel_layout="HWIO"
        )

    add_op = relay.op.nn.bias_add(conv, add_data,3)
    inputs = {
        "data": tvm.nd.array(np.random.uniform(size=data_shape).astype(dtype)),
    }

    params = {
        "weight": tvm.nd.array(np.random.uniform(size=weight_shape).astype(dtype)),
        "add": tvm.nd.array(np.random.uniform(size=add_shape).astype(dtype)),
    }
    print("Testing {0: <50}".format("CONV2D"), end="")
    verify_vsi_result(inputs, add_op, params, data_shape, out_shape, dtype)


def test_requantize():
    input_shape = (1, 2, 2, 5)
    output_shape = input_shape
    intput_dtype="uint8"
    output_dtype="int8"

    data = relay.var("data", shape=input_shape, dtype=intput_dtype)

    op_params = {
            "input_scale": relay.const(0.00784314, "float32"),
            "input_zero_point": relay.const(127, "int32"),
            "output_scale": relay.const(0.01784314, "float32"),
            "output_zero_point": relay.const(127, "int32"),
            "out_dtype":output_dtype,
        }
    out = relay.qnn.op.requantize(data,**op_params)
    inputs = {
        "data": tvm.nd.array(np.random.uniform(0,100,size=input_shape).astype(intput_dtype)),
    }
    print("Testing {0: <50}".format("REQUANTIZE"), end="")
    verify_vsi_result(inputs, out, [], input_shape, output_shape, output_dtype)


def test_uint8_conv2d_pattern():
    data_shape = (1, 56, 56, 32)
    weight_shape = (1, 1, 32, 64)
    out_shape = (1, 56, 56, 64)
    add_shape = (64,)
    intput_dtype="int8"
    output_dtype=intput_dtype
    add_dtype = "int32"
    Pad=(0,0,0,0)
    Strides=(1,1)
    Dilation=(1,1)
    Ksize=(1,1)
    Groups=1

    data = relay.var("data", shape=data_shape, dtype=intput_dtype)
    weight = relay.var("weight",shape=weight_shape,dtype=intput_dtype)
    add = relay.var("add",shape=add_shape,dtype=add_dtype)

    conv_params = {
            "kernel_size": Ksize,
            "strides": Strides,
            "dilation": Dilation,
            "padding": Pad,
            "data_layout": "NHWC",
            "channels":weight_shape[3],
            "kernel_layout":"HWIO"
        }
    qnn_conv2d_params = dict(conv_params)
    qnn_conv2d_params["input_zero_point"] = relay.const(0, "int32")
    qnn_conv2d_params["kernel_zero_point"] = relay.const(77, "int32")
    qnn_conv2d_params["out_dtype"] = "int32"
    qnn_conv2d_params["input_scale"] = relay.const(0.023528, "float32")
    qnn_conv2d_params["kernel_scale"] = relay.const(0.045283, "float32")
    conv_op = relay.qnn.op.conv2d(
            data,
            weight,
            **qnn_conv2d_params
        )

    add_op = relay.op.nn.bias_add(conv_op, add,3)

    requantize_params = {
            "input_scale": relay.const(0.001065418, "float32"),
            "input_zero_point": relay.const(0, "int32"),
            "output_scale": relay.const(0.0235285, "float32"),
            "output_zero_point": relay.const(0, "int32"),
            "out_dtype":output_dtype,
        }

    out = relay.qnn.op.requantize(add_op,**requantize_params)

    inputs = {
        "data": tvm.nd.array(np.ones(data_shape).astype(intput_dtype)),
    }
    params = {
        "weight": tvm.nd.array(np.arange(weight_shape[0]*weight_shape[1]*weight_shape[2]*weight_shape[3]).reshape(weight_shape).astype(intput_dtype)),
        "add": tvm.nd.array(np.arange(64).reshape(add_shape).astype(add_dtype)),
    }
    print("Testing {0: <50}".format("QNN pattern"), end="")
    verify_vsi_result(inputs, out, params, data_shape, out_shape, output_dtype)

def test_cast():
    input_dtype = "uint8"
    output_dtype = "float32"
    input_shape = (1, 3, 3,1 )
    output_shape = input_shape

    data = relay.var("data", shape=input_shape, dtype=input_dtype)

    out = relay.op.cast(data,output_dtype)
    inputs = {
      "data": tvm.nd.array(np.random.uniform(0,20,size=input_shape).astype(input_dtype)),
    }
    verify_vsi_result(inputs, out, [], input_shape, output_shape, output_dtype)

def test_uint8_avg_pool():
    input_dtype = "uint8"
    temp_dtype = "int32"
    input_shape = (1, 7, 7, 768)
    output_shape = (1, 1, 1, 768)

    data = relay.var("data", shape=input_shape, dtype=input_dtype)

    cast_0 = relay.op.cast(data,temp_dtype)
    out = relay.op.nn.avg_pool2d(cast_0,pool_size=(7, 7),strides=(2, 2),layout="NHWC")
    cast_1 = relay.op.cast(out,input_dtype)
    inputs = {
        "data": tvm.nd.array(np.arange(7*7*768).reshape(input_shape).astype(input_dtype)),
    }
    print("Testing {0: <50}".format("AVG_POOL_2D"), end="")
    verify_vsi_result(inputs, cast_1, [], input_shape, output_shape, input_dtype)

def test_uint8_softmax():
    input_dtype = "uint8"
    output_dtype = input_dtype
    temp_dtype = "float32"
    data_shape = (1,100)
    axis = 1
    data = relay.var("data", shape=data_shape, dtype=input_dtype)

    dequantize_op = relay.qnn.op.dequantize(data,
                            input_zero_point=relay.const(76, "int32"),
                            input_scale=relay.const(0.15294, "float32"),
                            axis = -1,
                            )
    softmax_op = relay.op.nn.softmax(dequantize_op,axis)

    quantize = relay.qnn.op.quantize(softmax_op,
                            output_scale=relay.const(0.003906, "float32"),
                            output_zero_point=relay.const(0, "int32"),
                            axis = -1,
                            out_dtype=output_dtype
                            )
    inputs = {
        "data": tvm.nd.array(np.random.uniform(1,20,size=data_shape).astype(input_dtype)),
    }
    print("Testing {0: <50}".format("SOFTMAX"), end="")
    verify_vsi_result(inputs, quantize, [], data_shape, data_shape, output_dtype)

def test_uint8_reshape():
    data_dtype = "uint8"
    data_shape = (1,1,1,1000)
    out_shape = (1,1000)
    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    out = relay.op.reshape(data,out_shape)
    inputs = {
        "data": tvm.nd.array(np.ones(data_shape,data_dtype)),
    }
    print("Testing {0: <50}".format("RESHAPE"), end="")
    verify_vsi_result(inputs, out, [], data_shape, out_shape, data_dtype)

def test_uint8_max_pool():
    input_dtype = "uint8"
    input_shape = (1, 112, 112, 2)
    output_shape = (1, 56, 56, 2)

    data = relay.var("data", shape=input_shape, dtype=input_dtype)

    out = relay.op.nn.max_pool2d(data,pool_size=(3, 3),strides=(2, 2),padding=(0,0,1, 1),layout="NHWC")
    inputs = {
        "data": tvm.nd.array(np.random.uniform(1,20,size=input_shape).astype(input_dtype)),
    }
    print("Testing {0: <50}".format("MAX_POOL_2D"), end="")
    verify_vsi_result(inputs, out, [], input_shape, output_shape, input_dtype)


def test_uint8_concatenation():
    dtype = "uint8"
    data_0_shape = (1, 14, 14, 320)
    data_1_shape = (1, 14, 14, 160)
    data_2_shape = (1, 14, 14, 96)

    out_shape = (1, 14, 14, 576)

    data_0 = relay.var("data0", shape=data_0_shape, dtype=dtype)
    data_1 = relay.var("data1", shape=data_1_shape, dtype=dtype)
    data_2 = relay.var("data2", shape=data_2_shape, dtype=dtype)
    data = [data_0,data_1,data_2]

    input_scale_0 = relay.const(0.0673782, "float32")
    input_zp_0 =  relay.const(0, "int32")

    input_scale_1 = relay.const(0.0485237, "float32")
    input_zp_1 =  relay.const(0, "int32")

    input_scale_2 = relay.const(0.03775704, "float32")
    input_zp_2 =  relay.const(0, "int32")

    input_scales = (input_scale_0,input_scale_1,input_scale_2)
    input_zps = (input_zp_0,input_zp_1,input_zp_2)

    output_scale = relay.const(0.0673782, "float32")
    output_zp =   relay.const(0, "int32")
    out = relay.qnn.op.concatenate(data,input_scales=input_scales,input_zero_points=input_zps,
                                    output_scale=output_scale,output_zero_point=output_zp,axis=3)

    inputs = {
         "data0": tvm.nd.array(np.random.uniform(1,50,size=data_0_shape).astype(dtype)),
         "data1": tvm.nd.array(np.random.uniform(1,50,size=data_1_shape).astype(dtype)),
         "data2": tvm.nd.array(np.random.uniform(1,50,size=data_2_shape).astype(dtype)),
    }
    print("Testing {0: <50}".format("AVG_POOL_2D"), end="")
    verify_vsi_result(inputs, out, [], data_0_shape, out_shape, dtype)

def test_float_mean():
    input_dtype = "float32"
    axis_dtype = "int32"
    input_shape = (1, 20, 20, 5)
    output_shape = (1, 1,1,5)

    data = relay.var("data", shape=input_shape, dtype=input_dtype)
    axis = tuple(np.array([1,2],dtype=axis_dtype))
    out = relay.op.reduce.mean(data,axis,True)
    inputs = {
        "data": tvm.nd.array(np.random.uniform(1,20,size=input_shape).astype(input_dtype)),
    }
    print("Testing {0: <50}".format("MEAN"), end="")
    verify_vsi_result(inputs, out, [], input_shape, output_shape, input_dtype)

def test_uint8_mean():
    input_dtype = "uint8"
    temp_dtype = "int32"
    axis_dtype = "int32"
    output_dtype = input_dtype
    input_shape = (1, 20, 20, 5)
    axis_shape = (2,)
    output_shape = (1, 1,1,5)

    data = relay.var("data", shape=input_shape, dtype=input_dtype)
    axis = tuple(np.array([1,2],dtype=axis_dtype))

    cast_0 = relay.op.cast(data,temp_dtype)
    mean = relay.op.reduce.mean(data,axis,True)
    requantize_params = {
            "input_scale": relay.const(0.001065418, "float32"),
            "input_zero_point": relay.const(0, "int32"),
            "output_scale": relay.const(0.0235285, "float32"),
            "output_zero_point": relay.const(0, "int32"),
            "out_dtype":output_dtype,
        }

    out = relay.qnn.op.requantize(mean,**requantize_params)
    inputs = {
        "data": tvm.nd.array(np.random.uniform(1,20,size=input_shape).astype(input_dtype)),
    }
    print("Testing {0: <50}".format("MEAN"), end="")
    verify_vsi_result(inputs, out, [], input_shape, output_shape, input_dtype)

def test_uint8_resizeBilinear():
    input_dtype = "uint8"
    size_dtype = "int32"
    output_dtype = input_dtype
    input_shape = (1, 1, 1, 256)
    output_shape = (1, 33, 33, 256)

    data = relay.var("data", shape=input_shape, dtype=input_dtype)
    target_size = tuple(np.array([33,33],dtype=size_dtype))
    method = "bilinear"
    coord_trans = "align_corners"
    out = relay.image.resize(
            data, target_size, "NHWC", method, coordinate_transformation_mode=coord_trans
        )
    inputs = {
        "data": tvm.nd.array(np.random.uniform(1,20,size=input_shape).astype(input_dtype)),
    }
    verify_vsi_result(inputs, out, [], input_shape, output_shape, input_dtype)

def test_uint8_argmax():
    input_dtype = "uint8"
    output_dtype = "int32"
    input_shape = (1, 513, 513, 21)
    output_shape = (1, 513, 513)

    data = relay.var("data", shape=input_shape, dtype=input_dtype)

    out = relay.op.argmax(data, 3)
    inputs = {
        "data": tvm.nd.array(np.random.uniform(1,100,size=input_shape).astype(input_dtype)),
    }
    verify_vsi_result(inputs, out, [], input_shape, output_shape, output_dtype)

def test_uint8_argmin():
    input_dtype = "uint8"
    output_dtype = "int32"
    input_shape = (1, 513, 513, 21)
    output_shape = (1, 513, 513)

    data = relay.var("data", shape=input_shape, dtype=input_dtype)

    out = relay.op.argmin(data, 3)
    inputs = {
        "data": tvm.nd.array(np.random.uniform(1,100,size=input_shape).astype(input_dtype)),
    }
    verify_vsi_result(inputs, out, [], input_shape, output_shape, output_dtype)

def test_float_sigmoid():
    dtype = "float32"
    data_shape = (1,100)
    out_shape = data_shape
    data = relay.var("data", shape=data_shape, dtype=dtype)
    out = relay.op.sigmoid(data)
    inputs = {
        "data": tvm.nd.array(np.random.uniform(size=data_shape).astype(dtype)),
    }
    print("Testing {0: <50}".format("SIGMOID"), end="")
    verify_vsi_result(inputs, out, [], data_shape, out_shape, dtype)

def test_uint8_sigmoid():
    input_dtype = "uint8"
    output_dtype = input_dtype
    temp_dtype = "float32"
    data_shape = (1,100)
    data = relay.var("data", shape=data_shape, dtype=input_dtype)

    dequantize_op = relay.qnn.op.dequantize(data,
                            input_zero_point=relay.const(0, "int32"),
                            input_scale=relay.const(0.15294, "float32"),
                            axis = -1,
                            )
    sigmoid_op = relay.op.sigmoid(dequantize_op)

    quantize = relay.qnn.op.quantize(sigmoid_op,
                            output_scale=relay.const(0.15294, "float32"),
                            output_zero_point=relay.const(0, "int32"),
                            axis = -1,
                            out_dtype=output_dtype
                            )
    inputs = {
        "data": tvm.nd.array(np.random.uniform(1,20,size=data_shape).astype(input_dtype)),
    }
    print("Testing {0: <50}".format("SIGMOID"), end="")
    verify_vsi_result(inputs, quantize, [], data_shape, data_shape, output_dtype)

def test_float_batch_norm():
    data_shape = (1, 4)
    c_shape = (4,)
    out_shape = (1, 4)

    dtype = "float32"
    w = tvm.nd.array(np.ones(c_shape, dtype))
    gamma = relay.const(w, dtype)
    beta = relay.const(w, dtype)
    moving_mean = relay.const(w, dtype)
    moving_var = relay.const(w, dtype)

    epsilon = 1e-4

    data = relay.var("data", shape=data_shape, dtype=dtype)

    batch_norm = relay.nn.batch_norm(data, gamma, beta, moving_mean, moving_var,
        epsilon=epsilon)
    out = batch_norm[0]
    inputs = {
        "data": tvm.nd.array(np.random.uniform(size=data_shape).astype(dtype)),
    }

    verify_vsi_result(inputs, out, [], data_shape, out_shape, dtype)

def test_uint8_avg_pool2():
    input_dtype = "uint8"
    temp_dtype = "int32"
    input_shape = (1, 4, 4, 1)
    output_shape = (1, 4, 4, 1)

    data = relay.var("data", shape=input_shape, dtype=input_dtype)

    cast_0 = relay.op.cast(data,temp_dtype)
    out = relay.op.nn.avg_pool2d(cast_0,pool_size=(3, 3),strides=(1, 1),padding=(1,1,1,1),layout="NHWC")
    cast_1 = relay.op.cast(out,input_dtype)
    inputs = {
        #"data": tvm.nd.array(np.ones(input_shape,input_dtype)),
        "data": tvm.nd.array(np.arange(4*4*1).reshape(input_shape).astype(input_dtype)),
    }
    print("Testing {0: <50}".format("AVG_POOL_2D"), end="")
    verify_vsi_result(inputs, cast_1, [], input_shape, output_shape, input_dtype)

def test_uint8_depthwiseconv2d_pattern():
    data_shape = (1, 12, 12, 3)
    weight_shape = (7, 7, 3, 8)
    out_shape = (1, 6, 6, 24)
    add_shape = (24,)
    intput_dtype="uint8"
    output_dtype=intput_dtype
    add_dtype = "int32"
    Pad=(2,2,3,3)
    Strides=(2,2)
    Dilation=(1,1)
    Ksize=(7,7)
    Groups=3

    data = relay.var("data", shape=data_shape, dtype=intput_dtype)
    weight = relay.var("weight",shape=weight_shape,dtype=intput_dtype)
    add = relay.var("add",shape=add_shape,dtype=add_dtype)

    conv_params = {
            "kernel_size": Ksize,
            "strides": Strides,
            "dilation": Dilation,
            "padding": Pad,
            "data_layout": "NHWC",
            "channels":24,
            "kernel_layout":"HWOI",
            "groups":Groups
        }
    qnn_conv2d_params = dict(conv_params)
    qnn_conv2d_params["input_zero_point"] = relay.const(128, "int32")
    qnn_conv2d_params["kernel_zero_point"] = relay.const(148, "int32")
    qnn_conv2d_params["out_dtype"] = "int32"
    qnn_conv2d_params["input_scale"] = relay.const(0.0078125, "float32")
    qnn_conv2d_params["kernel_scale"] = relay.const(0.08764044, "float32")
    conv_op = relay.qnn.op.conv2d(
            data,
            weight,
            **qnn_conv2d_params
        )

    add_op = relay.op.nn.bias_add(conv_op, add,3)

    requantize_params = {
            "input_scale": relay.const(0.000684690952, "float32"),
            "input_zero_point": relay.const(0, "int32"),
            "output_scale": relay.const(0.906536, "float32"),
            "output_zero_point": relay.const(128, "int32"),
            "out_dtype":output_dtype,
        }

    out = relay.qnn.op.requantize(add_op,**requantize_params)

    inputs = {
        "data": tvm.nd.array(np.ones(data_shape).astype(intput_dtype)),
    }
    params = {
        "weight": tvm.nd.array(np.arange(weight_shape[0]*weight_shape[1]*weight_shape[2]*weight_shape[3]).reshape(weight_shape).astype(intput_dtype)),
        "add": tvm.nd.array(np.arange(24).reshape(add_shape).astype(add_dtype)),
    }
    print("Testing {0: <50}".format("QNN pattern"), end="")
    verify_vsi_result(inputs, out, params, data_shape, out_shape, output_dtype)


def test_uint8_fullconnected():
    input_dtype = "uint8"
    temp_dtype = "int32"
    output_dtype = input_dtype
    input_shape = (1, 1, 1, 1536)
    weight_shape = (1001, 1536)
    add_shape = (1001,)
    reshape_output_shape = (-1, 1536)
    output_shape = (1, 1001)

    data = relay.var("data", shape=input_shape, dtype=input_dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=input_dtype)
    add = relay.var("add", shape=add_shape, dtype=temp_dtype)
    reshape_op = relay.op.reshape(data, reshape_output_shape)
    dense_op = relay.qnn.op.dense(reshape_op, weight,
                                  input_zero_point=relay.const(0, "int32"),
                                  kernel_zero_point=relay.const(0, "int32"),
                                  input_scale=relay.const(
                                      1.0, "float32"),
                                  kernel_scale=relay.const(
                                      1.0, "float32"),
                                  units=weight_shape[0],
                                  out_dtype=temp_dtype)

    add_op = relay.op.nn.bias_add(dense_op, add)

    requantize_params = {
        "input_scale": relay.const(1.0, "float32"),
        "input_zero_point": relay.const(0, "int32"),
        "output_scale": relay.const(0.005, "float32"),
        "output_zero_point": relay.const(0, "int32"),
        "out_dtype": output_dtype,
    }

    out = relay.qnn.op.requantize(add_op, **requantize_params)
    inputs = {
        "data": tvm.nd.array(np.random.randint(1, high=10, size=input_shape, dtype=input_dtype)),
    }
    params = {
        "weight": tvm.nd.array(np.random.randint(1, high=10, size=weight_shape, dtype=input_dtype)),
        "add": tvm.nd.array(np.random.randint(1, high=10, size=add_shape, dtype=temp_dtype)),
    }
    print("Testing {0: <50}".format("AVG_POOL_2D"), end="")
    verify_vsi_result(inputs, out, params, input_shape,
                      output_shape, output_dtype)

def test_uint8_squeeze():
    data_dtype = "uint8"
    axis_dtype = "int32"
    data_shape = (1,1,1,1000)
    out_shape = (1,1000)

    axis = tuple(np.array([1,2],dtype=axis_dtype))
    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    out = relay.op.squeeze(data,axis)
    inputs = {
        "data": tvm.nd.array(np.ones(data_shape,data_dtype)),
    }
    print("Testing {0: <50}".format("RESHAPE"), end="")
    verify_vsi_result(inputs, out, [], data_shape, out_shape, data_dtype)


def test_uint8_depthtospace():
    input_dtype = "uint8"
    input_shape = (1, 256, 256, 256)
    out_shape = (1, 512, 512, 64)

    data = relay.var("data", shape=input_shape, dtype=input_dtype)
    out = relay.op.nn.depth_to_space(data, 2, layout="NHWC")
    inputs = {
        "data": tvm.nd.array(np.random.randint(1, high=10, size=input_shape, dtype=input_dtype)),
    }
    print("Testing {0: <50}".format("RESHAPE"), end="")
    verify_vsi_result(inputs, out, [], input_shape, out_shape, input_dtype)

def test_qnn_sub():
    data_dtype = "uint8"
    data_shape = (1, 8, 8, 1)
    out_shape = (1, 8, 8, 1)

    x = relay.var("x", shape=data_shape, dtype=data_dtype)
    y = relay.var("y", shape=data_shape, dtype=data_dtype)
    out = relay.qnn.op.subtract(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(0.00784314, "float32"),
        lhs_zero_point=relay.const(127, "int32"),
        rhs_scale=relay.const(0.00784314, "float32"),
        rhs_zero_point=relay.const(127, "int32"),
        output_scale=relay.const(0.00784314, "float32"),
        output_zero_point=relay.const(127, "int32"),
    )

    print("Testing {0: <50}".format("QNN.SUB"), end="")
    inputs = {
        "x": tvm.nd.array(np.random.randint(1, high=101, size=data_shape, dtype="uint8")),
        "y": tvm.nd.array(np.random.randint(1, high=101, size=data_shape, dtype="uint8")),
    }
    verify_vsi_result(inputs, out, [], data_shape, out_shape, data_dtype)

def test_qnn_multiply():
    data_dtype = "uint8"
    data_shape = (1, 8, 8, 1)
    out_shape = (1, 8, 8, 1)

    x = relay.var("x", shape=data_shape, dtype=data_dtype)
    y = relay.var("y", shape=data_shape, dtype=data_dtype)
    out = relay.qnn.op.mul(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(0.00784314, "float32"),
        lhs_zero_point=relay.const(127, "int32"),
        rhs_scale=relay.const(0.00784314, "float32"),
        rhs_zero_point=relay.const(127, "int32"),
        output_scale=relay.const(0.00784314, "float32"),
        output_zero_point=relay.const(127, "int32"),
    )

    print("Testing {0: <50}".format("QNN.SUB"), end="")
    inputs = {
        "x": tvm.nd.array(np.random.randint(1, high=101, size=data_shape, dtype="uint8")),
        "y": tvm.nd.array(np.random.randint(1, high=101, size=data_shape, dtype="uint8")),
    }
    verify_vsi_result(inputs, out, [], data_shape, out_shape, data_dtype)

def test_qnn_maximum():
    data_dtype = "uint8"
    data_shape = (1, 8, 8, 1)
    out_shape = (1, 8, 8, 1)

    x = relay.var("x", shape=data_shape, dtype=data_dtype)
    y = relay.var("y", shape=data_shape, dtype=data_dtype)
    out = relay.op.maximum(
        lhs=x,
        rhs=y,
    )

    print("Testing {0: <50}".format("MAXINUM"), end="")
    inputs = {
        "x": tvm.nd.array(np.random.randint(1, high=101, size=data_shape, dtype="uint8")),
        "y": tvm.nd.array(np.random.randint(1, high=101, size=data_shape, dtype="uint8")),
    }
    verify_vsi_result(inputs, out, [], data_shape, out_shape, data_dtype)

def test_qnn_minimum():
    data_dtype = "uint8"
    data_shape = (1, 8, 8, 1)
    out_shape = (1, 8, 8, 1)

    x = relay.var("x", shape=data_shape, dtype=data_dtype)
    y = relay.var("y", shape=data_shape, dtype=data_dtype)
    out = relay.op.minimum(
        lhs=x,
        rhs=y,
    )

    print("Testing {0: <50}".format("MININUM"), end="")
    inputs = {
        "x": tvm.nd.array(np.random.randint(1, high=101, size=data_shape, dtype="uint8")),
        "y": tvm.nd.array(np.random.randint(1, high=101, size=data_shape, dtype="uint8")),
    }
    verify_vsi_result(inputs, out, [], data_shape, out_shape, data_dtype)

def test_qnn_logical_and():
    data_dtype = "bool"
    data_shape = (1, 8, 8, 1)
    out_shape = (1, 8, 8, 1)

    x = relay.var("x", shape=data_shape, dtype=data_dtype)
    y = relay.var("y", shape=data_shape, dtype=data_dtype)
    out = relay.op.logical_and(lhs=x,rhs=y)

    print("Testing {0: <50}".format("QNN.LOGICAL_AND"), end="")
    inputs = {
        "x": tvm.nd.array(np.random.randint(0, high=2, size=data_shape, dtype=data_dtype)),
        "y": tvm.nd.array(np.random.randint(0, high=2, size=data_shape, dtype=data_dtype)),
    }
    verify_vsi_result(inputs, out, [], data_shape, out_shape, data_dtype)

def test_qnn_logical_or():
    data_dtype = "bool"
    data_shape = (1, 8, 8, 1)
    out_shape = (1, 8, 8, 1)

    x = relay.var("x", shape=data_shape, dtype=data_dtype)
    y = relay.var("y", shape=data_shape, dtype=data_dtype)
    out = relay.op.logical_or(lhs=x,rhs=y)

    print("Testing {0: <50}".format("QNN.LOGICAL_OR"), end="")
    inputs = {
        "x": tvm.nd.array(np.random.randint(0, high=2, size=data_shape, dtype=data_dtype)),
        "y": tvm.nd.array(np.random.randint(0, high=2, size=data_shape, dtype=data_dtype)),
    }
    verify_vsi_result(inputs, out, [], data_shape, out_shape, data_dtype)

def test_qnn_pad():
    data_dtype = "uint8"
    data_shape = (1, 8, 8, 2)
    out_shape = (1, 10, 10, 2)

    paddings_num = [[0,0],[1,1],[1,1],[0,0]]

    x = relay.var("x", shape=data_shape, dtype=data_dtype)
    paddings = tuple(tuple(l) for l in paddings_num)
    pad_value = float(0)
    out = relay.op.nn.pad(x,paddings,pad_value)

    print("Testing {0: <50}".format("QNN.LOGICAL_OR"), end="")
    inputs = {
        "x": tvm.nd.array(np.random.randint(0, high=100, size=data_shape, dtype=data_dtype)),
    }
    verify_vsi_result(inputs, out, [], data_shape, out_shape, data_dtype)

def test_uint8_resizeNear():
    input_dtype = "uint8"
    size_dtype = "int32"
    output_dtype = input_dtype
    input_shape = (1, 38, 38, 128)
    output_shape = (1, 76, 76, 128)

    data = relay.var("data", shape=input_shape, dtype=input_dtype)
    target_size = tuple(np.array([76,76],dtype=size_dtype))
    method = "nearest_neighbor"
    coord_trans = "asymmetric"
    out = relay.image.resize(
            data, target_size, "NHWC", method, coordinate_transformation_mode=coord_trans
        )
    inputs = {
        "data": tvm.nd.array(np.random.uniform(1,200,size=input_shape).astype(input_dtype)),
    }
    verify_vsi_result(inputs, out, [], input_shape, output_shape, input_dtype)

def test_uint8_mean():
    input_dtype = "uint8"
    temp_dtype = "int32"
    output_dtype = input_dtype
    input_shape = (1, 7, 7, 20)
    output_shape = (1, 1,1,20)

    data = relay.var("data", shape=input_shape, dtype=input_dtype)
    cast = relay.op.cast(data,temp_dtype)
    axis = tuple(np.array([1,2],dtype=temp_dtype))
    mean = relay.op.reduce.mean(cast,axis,True)

    requantize_params = {
        "input_scale": relay.const(0.1568378, "float32"),
        "input_zero_point": relay.const(0, "int32"),
        "output_scale": relay.const(0.1568378, "float32"),
        "output_zero_point": relay.const(0, "int32"),
        "out_dtype": output_dtype,
    }

    out = relay.qnn.op.requantize(mean, **requantize_params)
    inputs = {
        "data": tvm.nd.array(np.random.randint(1, high=10, size=input_shape, dtype=input_dtype)),
    }
    print("Testing {0: <50}".format("UINT MEAN"), end="")
    verify_vsi_result(inputs, out, [], input_shape,
                      output_shape, output_dtype)

def test_transpose_conv2d_pattern():
    data_shape = (1, 24, 24, 256)
    weight_shape = (256, 128, 2,2)
    out_shape = (1, 48, 48, 128)

    input_dtype = "uint8"
    out_dtype= input_dtype

    data = relay.var("data", shape=data_shape, dtype=input_dtype)
    weight = relay.var("weight",shape=weight_shape,dtype=input_dtype)

    out = relay.nn.conv2d_transpose(
        data,
        weight,
        strides=(2, 2),
        padding=(0, 0, 0, 0),
        channels=int(128),
        kernel_size=(int(2), int(2)),
        data_layout="NHWC",
        kernel_layout="OIHW",
        out_dtype=out_dtype,
    )

    inputs = {
        "data": tvm.nd.array(np.random.randint(1, high=10, size=data_shape, dtype=input_dtype)),

    }
    params = {
        "weight": tvm.nd.array(np.random.randint(1, high=200, size=weight_shape, dtype=input_dtype)),
    }
    print("Testing {0: <50}".format("QNN pattern"), end="")
    verify_vsi_result(inputs, out, params, data_shape, out_shape, out_dtype)

def test_uint8_transpose_conv2d_pattern():
    data_shape = (1, 24, 24, 256)
    weight_shape = (256, 128, 2,2)
    out_shape = (1, 48, 48, 128)

    input_dtype = "uint8"
    temp_dtype = "int32"
    output_dtype= input_dtype
    kernel_size=(2, 2)
    strides=(2, 2)
    padding=(0, 0, 0, 0)
    data_layout="NHWC"

    data = relay.var("data", shape=data_shape, dtype=input_dtype)
    weight = relay.var("weight",shape=weight_shape,dtype=input_dtype)


    conv_params = {
            "kernel_size": kernel_size,
            "padding": padding,
            "data_layout": data_layout,
            "channels":weight_shape[1],
            "out_dtype":temp_dtype,
            "strides":strides
        }
    qnn_conv2d_params = dict(conv_params)
    qnn_conv2d_params["input_zero_point"] = relay.const(0, "int32")
    qnn_conv2d_params["kernel_zero_point"] = relay.const(129, "int32")
    qnn_conv2d_params["out_dtype"] = "int32"
    qnn_conv2d_params["input_scale"] = relay.const(0.0109899, "float32")
    qnn_conv2d_params["kernel_scale"] = relay.const(0.00171253, "float32")
    conv_op = relay.qnn.op.conv2d_transpose(
            data,
            weight,
            **qnn_conv2d_params
        )

    requantize_params = {
            "input_scale": relay.const(0.0109899*0.00171253, "float32"),
            "input_zero_point": relay.const(0, "int32"),
            "output_scale": relay.const(0.00000125877, "float32"),
            "output_zero_point": relay.const(124, "int32"),
            "axis": 3,
            "out_dtype":output_dtype,
        }
    out = relay.qnn.op.requantize(conv_op,**requantize_params)


    inputs = {
        "data": tvm.nd.array(np.random.randint(1, high=20, size=data_shape, dtype=input_dtype)),

    }
    params = {
        "weight": tvm.nd.array(np.random.randint(1, high=20, size=weight_shape, dtype=input_dtype)),
    }
    print("Testing {0: <50}".format("QNN pattern"), end="")
    verify_vsi_result(inputs, out, params, data_shape, out_shape, output_dtype)

def test_uint8_transpose_conv2d_pattern2():
    data_shape = (1, 24, 24, 128)
    weight_shape = (128, 64, 3,3)
    out_shape = (1, 48, 48, 64)

    input_dtype = "uint8"
    temp_dtype = "int32"
    output_dtype= input_dtype
    kernel_size=(3, 3)
    strides=(2, 2)
    padding=(0, 0, 1, 1)
    data_layout="NHWC"

    data = relay.var("data", shape=data_shape, dtype=input_dtype)
    weight = relay.var("weight",shape=weight_shape,dtype=input_dtype)


    conv_params = {
            "kernel_size": kernel_size,
            "padding": padding,
            "data_layout": data_layout,
            "channels":weight_shape[1],
            "out_dtype":temp_dtype,
            "strides":strides
        }
    qnn_conv2d_params = dict(conv_params)
    qnn_conv2d_params["input_zero_point"] = relay.const(0, "int32")
    qnn_conv2d_params["kernel_zero_point"] = relay.const(129, "int32")
    qnn_conv2d_params["out_dtype"] = "int32"
    qnn_conv2d_params["input_scale"] = relay.const(0.0109899, "float32")
    qnn_conv2d_params["kernel_scale"] = relay.const(0.00171253, "float32")
    conv_op = relay.qnn.op.conv2d_transpose(
            data,
            weight,
            **qnn_conv2d_params
        )

    requantize_params = {
            "input_scale": relay.const(0.0109899*0.00171253, "float32"),
            "input_zero_point": relay.const(0, "int32"),
            "output_scale": relay.const(0.00000125877, "float32"),
            "output_zero_point": relay.const(124, "int32"),
            "axis": 3,
            "out_dtype":output_dtype,
        }
    out = relay.qnn.op.requantize(conv_op,**requantize_params)


    inputs = {
        "data": tvm.nd.array(np.random.randint(1, high=20, size=data_shape, dtype=input_dtype)),

    }
    params = {
        "weight": tvm.nd.array(np.random.randint(1, high=20, size=weight_shape, dtype=input_dtype)),
    }
    print("Testing {0: <50}".format("QNN pattern"), end="")
    verify_vsi_result(inputs, out, params, data_shape, out_shape, output_dtype)

def test_uint8_tanh():
    input_dtype = "uint8"
    output_dtype = input_dtype
    temp_dtype = "float32"
    data_shape = (1,100)
    data = relay.var("data", shape=data_shape, dtype=input_dtype)

    dequantize_op = relay.qnn.op.dequantize(data,
                            input_zero_point=relay.const(0, "int32"),
                            input_scale=relay.const(0.15294, "float32"),
                            axis = -1,
                            )
    sigmoid_op = relay.op.tanh(dequantize_op)

    quantize = relay.qnn.op.quantize(sigmoid_op,
                            output_scale=relay.const(0.15294, "float32"),
                            output_zero_point=relay.const(0, "int32"),
                            axis = -1,
                            out_dtype=output_dtype
                            )
    inputs = {
        "data": tvm.nd.array(np.random.uniform(1,20,size=data_shape).astype(input_dtype)),
    }
    print("Testing {0: <50}".format("SIGMOID"), end="")
    verify_vsi_result(inputs, quantize, [], data_shape, data_shape, output_dtype)

if __name__ == "__main__":
    #test_qnn_add()
    # test_float_add()
    #test_float_relu()
    #test_uint8_relu()
    #test_float_leaky_relu()
    #test_uint8_leaky_relu()
    #test_float_softmax()
    # test_float32_conv2d_permute()
    #test_float32_depthwise_conv2d_permute()
    #test_float_reshape()
    #test_float_tranpose()
    #test_float_relu6()
    #test_uint8_relu6()
    #test_sample_model()
    #test_dequantize()
    #test_quantize()
    #test_float_avg_pool()
    #test_float32_pattern()
    #test_uint8_depthwiseconv2d_pattern()
    test_uint8_conv2d_pattern()
    #test_uint8_avg_pool()
    #test_uint8_softmax()
    #test_uint8_reshape()
    #test_uint8_concatenation()
    #test_uint8_max_pool()
    #test_float_mean()
    #test_uint8_resizeBilinear()
    #test_uint8_argmax()
    #test_float_sigmoid()
    # test_uint8_sigmoid()
    #test_float_batch_norm()
    #test_uint8_fullconnected()
    #test_uint8_argmin()
    #test_uint8_squeeze()
    #test_uint8_depthtospace()
    #test_qnn_sub()
    #test_qnn_multiply()
    #test_qnn_maximum()
    #test_qnn_minimum()
    #test_qnn_logical_and()
    #test_qnn_logical_or()
    #test_uint8_resizeNear()
    #test_qnn_pad()
    #test_uint8_mean()
    #test_requantize()
    #test_uint8_transpose_conv2d_pattern()
    # test_uint8_transpose_conv2d_pattern2()
    #test_uint8_tanh()
