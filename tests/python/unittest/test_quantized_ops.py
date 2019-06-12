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
import numpy as np
from tvm import relay
from tvm.relay.testing import create_workload
from tvm.contrib import graph_runtime

# TODOs for janimesh before submitting this patch.
# TODO - Add tests for int8 input/weight dtype
# TODO - opt_level=0 fails mostly due to fusion.
# TODO - opt_level=3 fails, likely culprit kernel layout for int8
# compute. Work with Rankyung to see if this is the culprit. Handle
# it in a separate patch.
rounding_modes = ["FE_UPWARD", "FE_AWAY_FROM_ZERO"]
def run_infer_type(expr):
    mod = relay.Module.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_requantize():
    def verify(func, goldens):
        with relay.build_config(opt_level=0):
            graph, lib, params = relay.build(func, "llvm", params=None)
            golden_data, golden_output = goldens
            mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
            mod.set_input("quantized_data",golden_data)
            mod.set_input(**params)
            mod.run()
            res = mod.get_output(0).asnumpy()
            np.testing.assert_equal(res, golden_output)

    def get_func(data_shape, data_dtype, out_dtype, use_int_compute,
            rounding_mode, input_scale, output_scale, input_zero_point=0,
            output_zero_point=0):
        quantized_data = relay.var("quantized_data", shape=data_shape,
                dtype=data_dtype)
        func = relay.op.qnn.requantize(
                quantized_data,
                input_zero_point=input_zero_point,
                output_zero_point=output_zero_point,
                input_scale=input_scale,
                output_scale=output_scale,
                rounding_mode=rounding_mode,
                out_dtype=out_dtype,
                use_int_compute=use_int_compute)

        func = relay.Function(relay.analysis.free_vars(func),
                func)
        func = run_infer_type(func)
        func = relay.quantize.rewrite(func)
        print(func)
        return func


    def run_tests():
        def same_scale_test():
            # Have same scales, everything within range
            golden_data = np.arange(-100, 100, 1).astype('int32')
            golden_output = golden_data

            for rounding_mode in rounding_modes:
                for use_int_compute in [True, False]:
                    func = get_func(data_shape=(200, ),
                                    data_dtype='int32',
                                    out_dtype="int8",
                                    use_int_compute=use_int_compute,
                                    rounding_mode=rounding_mode,
                                    input_scale=0.5,
                                    output_scale=0.5)
                    verify(func, (golden_data, golden_output))

        def downscale_test():
            for rounding_mode in rounding_modes:
                for use_int_compute in [True, False]:
                    func = get_func(data_shape=(32, ),
                                    data_dtype='int32',
                                    out_dtype="int32",
                                    use_int_compute=use_int_compute,
                                    rounding_mode=rounding_mode,
                                    input_scale=1,
                                    output_scale=16)

                    # Try positive values
                    # 8 corresponds to 0.5, resulting in 1
                    golden_data = np.arange(0, 32, 1).astype('int32')
                    golden_output = np.repeat([0, 1, 2], [8, 16, 8])
                    verify(func, (golden_data, golden_output))

                    # Try negative values
                    # -8 corresponds to -0.5. For FE_UPWARD, this is 0
                    golden_data = np.arange(0, -32, -1).astype('int32')
                    if use_int_compute == True and rounding_mode == "FE_UPWARD":
                        golden_output = np.repeat([0, -1, -2], [9, 16, 7])
                    else:
                        golden_output = np.repeat([0, -1, -2], [8, 16, 8])
                    verify(func, (golden_data, golden_output))

                # Try a different scale
                for use_int_compute in [True, False]:
                    func = get_func(data_shape=(32, ),
                                    data_dtype='int32',
                                    out_dtype="int8",
                                    use_int_compute=use_int_compute,
                                    rounding_mode=rounding_mode,
                                    input_scale=1,
                                    output_scale=4)

                    # Try positive values
                    # 2I corresponds to 0.5, resulting in 1
                    golden_data = np.arange(0, 32, 1).astype('int32')
                    golden_output = np.repeat([0, 1, 2, 3, 4, 5, 6, 7, 8],
                                              [2, 4, 4, 4, 4, 4, 4, 4, 2])
                    verify(func, (golden_data, golden_output))

                    # Try negative values
                    # -8 corresponds to -0.5. For FE_UPWARD, this is 0
                    golden_data = np.arange(0, -32, -1).astype('int32')
                    if use_int_compute == True and rounding_mode == "FE_UPWARD":
                        golden_output = np.repeat([0, -1, -2, -3, -4, -5, -6, -7, -8],
                                                  [3, 4, 4, 4, 4, 4, 4, 4, 1])
                    else:
                        golden_output = np.repeat([0, -1, -2, -3, -4, -5, -6, -7, -8],
                                                  [2, 4, 4, 4, 4, 4, 4, 4, 2])
                    verify(func, (golden_data, golden_output))

        def upscale_test():
            for rounding_mode in rounding_modes:
                for use_int_compute in [True, False]:
                    func = get_func(data_shape=(32, ),
                                    data_dtype='int32',
                                    out_dtype="int8",
                                    use_int_compute=use_int_compute,
                                    rounding_mode=rounding_mode,
                                    input_scale=2,
                                    output_scale=1)

                    # Try positive values
                    # 8 corresponds to 0.5, resulting in 1
                    golden_data = np.arange(0, 32, 1).astype('int32')
                    golden_output = np.multiply(2, golden_data)
                    verify(func, (golden_data, golden_output))

                    # Try negative values
                    # -8 corresponds to -0.5. For FE_UPWARD, this is 0
                    golden_data = np.arange(0, -32, -1).astype('int32')
                    golden_output = np.multiply(2, golden_data)
                    verify(func, (golden_data, golden_output))

        def saturation_test():
            for rounding_mode in rounding_modes:
                for use_int_compute in [True, False]:
                    func = get_func(data_shape=(16, ),
                                    data_dtype='int32',
                                    out_dtype="int8",
                                    use_int_compute=use_int_compute,
                                    rounding_mode=rounding_mode,
                                    input_scale=0.5,
                                    output_scale=0.5)
                    golden_data = np.arange(0, 16, 1).astype('int32')
                    golden_data = np.add(120, golden_data)
                    output = np.array([120, 121, 122, 123, 124, 125, 126, 127,
                                       127, 127, 127, 127, 127, 127, 127, 127])
                    golden_output = output
                    verify(func, (golden_data, golden_output))

                    # Try negative numbers
                    golden_data = np.arange(0, -16, -1).astype('int32')
                    golden_data = np.add(-120, golden_data)
                    output = np.array([-120, -121, -122, -123, -124, -125, -126, -127,
                                       -128, -128, -128, -128, -128, -128, -128, -128])
                    golden_output = output
                    verify(func, (golden_data, golden_output))

        def zero_point_test():
            # Output zero point
            for rounding_mode in rounding_modes:
                for use_int_compute in [True, False]:
                    func = get_func(data_shape=(32, ),
                                    data_dtype='int32',
                                    out_dtype="int32",
                                    use_int_compute=use_int_compute,
                                    rounding_mode=rounding_mode,
                                    input_scale=1,
                                    output_scale=16,
                                    output_zero_point=1)

                    # Try positive values
                    # 8 corresponds to 0.5, resulting in 1
                    golden_data = np.arange(0, 32, 1).astype('int32')
                    golden_output = np.repeat([0, 1, 2], [8, 16, 8])
                    golden_output = np.add(1, golden_output)
                    verify(func, (golden_data, golden_output))

                    # Try negative values
                    # -8 corresponds to -0.5. For FE_UPWARD, this is 0
                    golden_data = np.arange(-32, -64, -1).astype('int32')
                    if use_int_compute == True and rounding_mode == "FE_UPWARD":
                        golden_output = np.repeat([-2, -3, -4], [9, 16, 7])
                    else:
                        golden_output = np.repeat([-2, -3, -4], [8, 16, 8])
                    golden_output = np.add(1, golden_output)
                    verify(func, (golden_data, golden_output))

            # Input zero point
            for rounding_mode in rounding_modes:
                for use_int_compute in [True, False]:
                    func = get_func(data_shape=(32, ),
                                    data_dtype='int32',
                                    out_dtype="int32",
                                    use_int_compute=use_int_compute,
                                    rounding_mode=rounding_mode,
                                    input_scale=1,
                                    output_scale=16,
                                    input_zero_point=16)

                    # Try positive values
                    golden_data = np.arange(32, 64, 1).astype('int32')
                    golden_output = np.repeat([2, 3, 4], [8, 16, 8])
                    golden_output = np.subtract(golden_output, 1)
                    verify(func, (golden_data, golden_output))

                    # Try negative values
                    golden_data = np.arange(-32, -64, -1).astype('int32')
                    if use_int_compute == True and rounding_mode == "FE_UPWARD":
                        golden_output = np.repeat([-2, -3, -4], [9, 16, 7])
                    else:
                        golden_output = np.repeat([-2, -3, -4], [8, 16, 8])
                    golden_output = np.subtract(golden_output, 1)
                    verify(func, (golden_data, golden_output))




        if __name__ == "__main__":
            same_scale_test()
            downscale_test()
            upscale_test()
            saturation_test()
            zero_point_test()

    run_tests()


def test_qconv2d_requantize():
    def verify(func, goldens):
        with relay.build_config(opt_level=0):
            graph, lib, params = relay.build(func, "llvm", params=None)
            golden_data, golden_weight, golden_output = goldens
            mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
            mod.set_input("quantized_data",golden_data)
            mod.set_input("weight",golden_weight)
            mod.set_input(**params)
            mod.run()
            res = mod.get_output(0).asnumpy()
            np.testing.assert_equal(res, golden_output)

    def get_func(data_shape, data_dtype, weight_shape, weight_dtype,
            strides, out_dtype, rounding_mode, use_int_compute):
        input_scale = 0.25098
        kernel_scale = 0.501961
        output_scale = 0.501961
        kernel_size = (2, 2)
        quantized_data = relay.var("quantized_data", shape=data_shape,
                dtype=data_dtype)
        quantized_weight = relay.var("weight", shape=weight_shape,
                dtype=weight_dtype)
        func = relay.op.qnn.conv2d(
                quantized_data, quantized_weight,
                input_zero_point=0,
                kernel_zero_point=0,
                kernel_size=kernel_size,
                strides=strides,
                out_dtype="int32",
                data_layout="NCHW",
                kernel_layout="OIHW")
        func = relay.op.qnn.requantize(
                func,
                input_zero_point=0,
                output_zero_point=0,
                input_scale=input_scale*kernel_scale,
                output_scale=output_scale,
                out_dtype=out_dtype,
                rounding_mode=rounding_mode,
                use_int_compute=use_int_compute)

        func = relay.Function(relay.analysis.free_vars(func),
                func)
        func = run_infer_type(func)
        func = relay.quantize.rewrite(func)
        print(func)
        return func

    def run_tests():
        def st1_basic_tests():
            # NCHW input
            golden_data = np.array([2, 1, 4, 3,
                                    5, 4, 2, 3,
                                    3, 8, 4, 9,
                                    6, 10, 1, 2]).astype('uint8')\
                                    .reshape((2, 1, 2, 4))
            # OIHW weight
            golden_weight = np.array([2, 4, 6, 8,
                                      4, 2, 6, 4,
                                      0, 4, 2, 8]).astype('uint8')\
                                     .reshape((3, 1, 2, 2))

            golden_output = np.array([18, 15, 14,
                                      14, 11, 12,
                                      12, 10, 10,
                                      39, 25, 17,
                                      26, 26, 12,
                                      31, 11, 14]).astype('uint8')\
                                      .reshape((2, 3, 1, 3))

            for use_int_compute in [True, False]:
                for rounding_mode in rounding_modes:
                    func = get_func(data_shape=(2, 1, 2, 4),
                                    data_dtype='uint8',
                                    weight_shape=(3, 1, 2, 2),
                                    weight_dtype='uint8',
                                    strides=(1, 1),
                                    out_dtype="uint8",
                                    rounding_mode=rounding_mode,
                                    use_int_compute=use_int_compute)
                    verify(func, (golden_data, golden_weight, golden_output))

            # Check the int8 input type as well
            for use_int_compute in [True, False]:
                for rounding_mode in rounding_modes:
                    func = get_func(data_shape=(2, 1, 2, 4),
                                    data_dtype='int8',
                                    weight_shape=(3, 1, 2, 2),
                                    weight_dtype='int8',
                                    strides=(1, 1),
                                    out_dtype="uint8",
                                    rounding_mode=rounding_mode,
                                    use_int_compute=use_int_compute)
                    verify(func, (golden_data, golden_weight, golden_output))


        def st2_basic_tests():
            # NCHW input
            golden_data = np.array([2, 1, 4, 3,
                                    5, 4, 2, 3,
                                    3, 8, 4, 9,
                                    6, 10, 1, 2]).astype('uint8')\
                                    .reshape((2, 1, 2, 4))
            # OIHW weight
            golden_weight = np.array([2, 4, 6, 8,
                                      4, 2, 6, 4,
                                      0, 4, 2, 8]).astype('uint8')\
                                     .reshape((3, 1, 2, 2))


            golden_output = np.array([18, 14, 14,
                                      12, 12, 10,
                                      39, 17, 26,
                                      12, 31, 14]).astype('uint8')\
                                      .reshape((2, 3, 1, 2))

            for use_int_compute in [True, False]:
                for rounding_mode in rounding_modes:
                    func = get_func(data_shape=(2, 1, 2, 4),
                                    data_dtype='uint8',
                                    weight_shape=(3, 1, 2, 2),
                                    weight_dtype='uint8',
                                    strides=(2, 2),
                                    out_dtype="uint8",
                                    rounding_mode=rounding_mode,
                                    use_int_compute=use_int_compute)
                    verify(func, (golden_data, golden_weight, golden_output))

        def st2_saturating_tests():
            # Check the output of saturating test
            golden_data = np.array([239, 239, 4, 3,
                                    6, 4, 2, 3,
                                    3, 8, 4, 9,
                                    6, 10, 1, 2]).astype('uint8')\
                                    .reshape((2, 1, 2, 4))
            # OIHW weight
            golden_weight = np.array([2, 4, 6, 8,
                                      4, 2, 6, 4,
                                      0, 4, 2, 8]).astype('uint8')\
                                     .reshape((3, 1, 2, 2))


            # Check uint8 output clamping
            golden_output = np.array([255, 14, 255,
                                      12, 251, 10,
                                      39, 17, 26,
                                      12, 31, 14]).astype('uint8')\
                                      .reshape((2, 3, 1, 2))
            for use_int_compute in [True, False]:
                for rounding_mode in rounding_modes:
                    func = get_func(data_shape=(2, 1, 2, 4),
                                    data_dtype='uint8',
                                    weight_shape=(3, 1, 2, 2),
                                    weight_dtype='uint8',
                                    strides=(2, 2),
                                    out_dtype="uint8",
                                    rounding_mode=rounding_mode,
                                    use_int_compute=use_int_compute)
                    verify(func, (golden_data, golden_weight, golden_output))

            # Check int8 output clamping
            golden_output = np.array([127, 14, 127,
                                      12, 127, 10,
                                      39, 17, 26,
                                      12, 31, 14]).astype('uint8')\
                                      .reshape((2, 3, 1, 2))
            for use_int_compute in [True, False]:
                for rounding_mode in rounding_modes:
                    func = get_func(data_shape=(2, 1, 2, 4),
                                    data_dtype='uint8',
                                    weight_shape=(3, 1, 2, 2),
                                    weight_dtype='uint8',
                                    strides=(2, 2),
                                    out_dtype="int8",
                                    rounding_mode=rounding_mode,
                                    use_int_compute=use_int_compute)
                    verify(func, (golden_data, golden_weight, golden_output))


            # Check that int16 does not clamp
            golden_output = np.array([377, 14, 373,
                                      12, 251, 10,
                                      39, 17, 26,
                                      12, 31, 14]).astype('uint16')\
                                      .reshape((2, 3, 1, 2))
            for use_int_compute in [True, False]:
                for rounding_mode in rounding_modes:
                    func = get_func(data_shape=(2, 1, 2, 4),
                                    data_dtype='uint8',
                                    weight_shape=(3, 1, 2, 2),
                                    weight_dtype='uint8',
                                    strides=(2, 2),
                                    out_dtype="int16",
                                    rounding_mode=rounding_mode,
                                    use_int_compute=use_int_compute)
                    verify(func, (golden_data, golden_weight, golden_output))

        #def cast_test():
        #    data = relay.var("data", shape=(1, 5), dtype="float32")    # --> [63, -4, 76, 0, -5]
        #    data1 = relay.var("data1", shape=(1, 5), dtype="float32")
        #    data2 = relay.var("data2", shape=(1, 5), dtype="float32")
        #    relu1 = relay.op.nn.relu(data)
        #    relu2 = relay.op.nn.relu(data1)
        #    relu3 = relay.op.nn.relu(data2)
        #    func = relay.op.where(relu1, relu2, relu3)
        #    func = relay.op.nn.relu(func)
        #    print(func)
        #    func = relay.Function(relay.analysis.free_vars(func),
        #            func)


        #    with relay.build_config(opt_level=1):
        #        graph, lib, params = relay.build(func, "llvm", params=None)
        #        golden_data = np.array([[63, -4, 76, 0, -5]])
        #        golden_data1 = np.array([[1, 2, 3, 4, 5]])
        #        golden_data2 = np.array([[6, 7, 8, 9, 10]])
        #        mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
        #        mod.set_input("data",golden_data)
        #        mod.set_input("data1",golden_data1)
        #        mod.set_input("data2",golden_data2)
        #        mod.set_input(**params)
        #        mod.run()
        #        res = mod.get_output(0).asnumpy()
        #        print(res)
        #        #np.testing.assert_equal(res, golden_output)


        if __name__ == "__main__":
            st1_basic_tests()
            st2_basic_tests()
            st2_saturating_tests()
            # cast_test()

    run_tests()

if __name__ == "__main__":
    test_requantize()
    test_qconv2d_requantize()
