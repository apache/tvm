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
from tvm.contrib import graph_runtime


def run_infer_type(expr):
    mod = relay.Module.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_qnn_max_pool2d():
    def verify(func, goldens):
        with relay.build_config(opt_level=0):
            graph, lib, params = relay.build(func, "llvm", params=None)
            golden_data, golden_output = goldens
            mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
            mod.set_input("quantized_data", golden_data)
            mod.set_input(**params)
            mod.run()
            res = mod.get_output(0).asnumpy()
            np.testing.assert_equal(res, golden_output)

    def get_func(data_shape, data_dtype, zero_point, pool_size, strides, padding, ceil_mode):
        quantized_data = relay.var("quantized_data", shape=data_shape,
                                   dtype=data_dtype)
        func = relay.qnn.op.max_pool2d(quantized_data,
                                       input_zero_point=zero_point,
                                       pool_size=pool_size,
                                       strides=strides,
                                       padding=padding,
                                       layout="NCHW",
                                       ceil_mode=ceil_mode)

        func = relay.Function(relay.analysis.free_vars(func), func)
        func = run_infer_type(func)
        print(func)
        return func

    def run_tests():
        def basic_tests():
            # NCHW input
            golden_data = np.array([2, 1, 4, 3,
                                    5, 4, 2, 3,
                                    3, 8, 4, 9,
                                    6, 10, 1, 2]).astype('uint8') \
                .reshape((2, 1, 2, 4))

            golden_output = np.array([4, 3,
                                      9, 8]).astype('uint8') \
                .reshape((2, 1, 1, 2))

            func = get_func(data_shape=(2, 1, 2, 4),
                            data_dtype='uint8',
                            zero_point=1,
                            pool_size=(2, 2),
                            strides=(2, 2),
                            padding=(0, 0, 0, 0),
                            ceil_mode=False)
            verify(func, (golden_data, golden_output))

            # Check the int8 input type as well
            func = get_func(data_shape=(2, 1, 2, 4),
                            data_dtype='int8',
                            zero_point=1,
                            pool_size=(2, 2),
                            strides=(2, 2),
                            padding=(0, 0, 0, 0),
                            ceil_mode=False)
            verify(func, (golden_data, golden_output))


        # Get the same input data as TF testcases in a NCHW layout
        def get_input_golden_data(data_dtype, shape_NCHW):
            data_shape_NHWC = [shape_NCHW[0], shape_NCHW[2], shape_NCHW[3], shape_NCHW[1]]
            total_size = np.prod(data_shape_NHWC)
            golden_data = np.array([f * 1.0 for f in range(1, total_size + 1)], dtype=data_dtype)
            golden_data = golden_data.reshape(data_shape_NHWC)
            return np.transpose(golden_data, [0, 3, 1, 2])

        # Get the same output data as TF testcases in a NCHW layout
        def get_output_golden_data(output_data, output_dtype, shape_NCHW):
            output_shape_NHWC = [shape_NCHW[0], shape_NCHW[2], shape_NCHW[3], shape_NCHW[1]]
            golden_output = np.array(output_data, dtype=output_dtype).reshape(output_shape_NHWC)
            return np.transpose(golden_output, [0, 3, 1, 2])

        def test_valid_padding():
            data_dtype = 'uint8'
            data_shape_NCHW = [1, 3, 3, 3]
            output_shape_NCHW = [1, 3, 1, 1]
            output_data = [13.0, 14.0, 15.0]
            golden_data = get_input_golden_data(data_dtype, data_shape_NCHW)
            golden_output = get_output_golden_data(output_data, data_dtype, output_shape_NCHW)
            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            zero_point=0,
                            pool_size=(2, 2),
                            strides=(2, 2),
                            padding=(0, 0, 0, 0),
                            ceil_mode=False)
            verify(func, (golden_data, golden_output))

        def test_same_padding():
            data_dtype = 'uint8'
            data_shape_NCHW = [1, 3, 2, 3]
            output_shape_NCHW = [1, 3, 1, 2]
            output_data = [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
            golden_data = get_input_golden_data(data_dtype, data_shape_NCHW)
            golden_output = get_output_golden_data(output_data, data_dtype, output_shape_NCHW)
            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            zero_point=0,
                            pool_size=(2, 2),
                            strides=(2, 2),
                            padding=(0, 0, 0, 1),
                            ceil_mode=False)
            verify(func, (golden_data, golden_output))

        def test_same_padding_non_square_window():
            data_dtype = 'uint8'
            data_shape_NCHW = [1, 1, 2, 2]
            output_shape_NCHW = [1, 1, 2, 2]
            output_data = [2.0, 2.0, 4.0, 4.0]
            golden_data = get_input_golden_data(data_dtype, data_shape_NCHW)
            golden_output = get_output_golden_data(output_data, data_dtype, output_shape_NCHW)
            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            zero_point=0,
                            pool_size=(1, 2),
                            strides=(1, 1),
                            padding=(0, 0, 0, 1),
                            ceil_mode=False)
            verify(func, (golden_data, golden_output))

        def test_valid_padding_uneven_stride():
            data_dtype = 'uint8'
            data_shape_NCHW = [1, 1, 4, 4]
            output_shape_NCHW = [1, 1, 3, 2]
            output_data = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
            golden_data = get_input_golden_data(data_dtype, data_shape_NCHW)
            golden_output = get_output_golden_data(output_data, data_dtype, output_shape_NCHW)
            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            zero_point=0,
                            pool_size=(2, 2),
                            strides=(1, 2),
                            padding=(0, 0, 0, 0),
                            ceil_mode=False)
            verify(func, (golden_data, golden_output))

            output_shape_NCHW = [1, 1, 2, 3]
            output_data = [6.0, 7.0, 8.0, 14.0, 15.0, 16.0]
            golden_output = get_output_golden_data(output_data, data_dtype, output_shape_NCHW)
            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            zero_point=0,
                            pool_size=(2, 2),
                            strides=(2, 1),
                            padding=(0, 0, 0, 0),
                            ceil_mode=False)
            verify(func, (golden_data, golden_output))

        def test_same_padding_filter4():
            data_dtype = 'uint8'
            data_shape_NCHW = [1, 4, 4, 4]
            output_shape_NCHW = [1, 4, 2, 2]
            output_data = [
                21.0, 22.0, 23.0, 24.0, 29.0, 30.0, 31.0, 32.0, 53.0, 54.0, 55.0, 56.0,
                61.0, 62.0, 63.0, 64.0
            ]
            golden_data = get_input_golden_data(data_dtype, data_shape_NCHW)
            golden_output = get_output_golden_data(output_data, data_dtype, output_shape_NCHW)
            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            zero_point=0,
                            pool_size=(2, 2),
                            strides=(2, 2),
                            padding=(0, 0, 0, 0),
                            ceil_mode=False)
            verify(func, (golden_data, golden_output))

        def test_same_padding_filter8():
            data_dtype = 'int32'
            data_shape_NCHW = [1, 8, 8, 8]
            output_shape_NCHW = [1, 8, 4, 4]
            output_data = [
                145.0, 146.0, 147.0, 148.0, 149.0, 150.0, 151.0, 152.0, 161.0, 162.0,
                163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 177.0, 178.0, 179.0, 180.0,
                181.0, 182.0, 183.0, 184.0, 185.0, 186.0, 187.0, 188.0, 189.0, 190.0,
                191.0, 192.0, 273.0, 274.0, 275.0, 276.0, 277.0, 278.0, 279.0, 280.0,
                289.0, 290.0, 291.0, 292.0, 293.0, 294.0, 295.0, 296.0, 305.0, 306.0,
                307.0, 308.0, 309.0, 310.0, 311.0, 312.0, 313.0, 314.0, 315.0, 316.0,
                317.0, 318.0, 319.0, 320.0, 401.0, 402.0, 403.0, 404.0, 405.0, 406.0,
                407.0, 408.0, 417.0, 418.0, 419.0, 420.0, 421.0, 422.0, 423.0, 424.0,
                433.0, 434.0, 435.0, 436.0, 437.0, 438.0, 439.0, 440.0, 441.0, 442.0,
                443.0, 444.0, 445.0, 446.0, 447.0, 448.0, 465.0, 466.0, 467.0, 468.0,
                469.0, 470.0, 471.0, 472.0, 481.0, 482.0, 483.0, 484.0, 485.0, 486.0,
                487.0, 488.0, 497.0, 498.0, 499.0, 500.0, 501.0, 502.0, 503.0, 504.0,
                505.0, 506.0, 507.0, 508.0, 509.0, 510.0, 511.0, 512.0
            ]
            golden_data = get_input_golden_data(data_dtype, data_shape_NCHW)
            golden_output = get_output_golden_data(output_data, data_dtype, output_shape_NCHW)
            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            zero_point=0,
                            pool_size=(3, 3),
                            strides=(2, 2),
                            padding=(0, 0, 1, 1),
                            ceil_mode=False)
            verify(func, (golden_data, golden_output))

        def test_kernel_smaller_than_stride_valid():
            data_dtype = 'uint8'
            data_shape_NCHW = [1, 1, 7, 7]
            output_shape_NCHW = [1, 1, 2, 2]
            output_data = [9, 12, 30, 33]
            golden_data = get_input_golden_data(data_dtype, data_shape_NCHW)
            golden_output = get_output_golden_data(output_data, data_dtype, output_shape_NCHW)
            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            zero_point=0,
                            pool_size=(2, 2),
                            strides=(3, 3),
                            padding=(0, 0, 0, 0),
                            ceil_mode=False)
            verify(func, (golden_data, golden_output))

        def test_kernel_smaller_than_stride_same():
            data_dtype = 'uint8'
            data_shape_NCHW = [1, 1, 3, 3]
            output_shape_NCHW = [1, 1, 2, 2]
            output_data = [1, 3, 7, 9]
            golden_data = get_input_golden_data(data_dtype, data_shape_NCHW)
            golden_output = get_output_golden_data(output_data, data_dtype, output_shape_NCHW)
            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            zero_point=0,
                            pool_size=(1, 1),
                            strides=(2, 2),
                            padding=(0, 0, 1, 1),
                            ceil_mode=False)
            verify(func, (golden_data, golden_output))

            data_shape_NCHW = [1, 1, 4, 4]
            output_shape_NCHW = [1, 1, 2, 2]
            output_data = [1, 3, 9, 11]
            golden_data = get_input_golden_data(data_dtype, data_shape_NCHW)
            golden_output = get_output_golden_data(output_data, data_dtype, output_shape_NCHW)
            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            zero_point=0,
                            pool_size=(1, 1),
                            strides=(2, 2),
                            padding=(0, 0, 0, 0),
                            ceil_mode=False)
            verify(func, (golden_data, golden_output))

        basic_tests()
        test_valid_padding()
        test_same_padding()
        test_same_padding_non_square_window()
        test_valid_padding_uneven_stride()
        test_same_padding_filter4()
        test_same_padding_filter8()
        test_kernel_smaller_than_stride_valid()
        test_kernel_smaller_than_stride_same()

    run_tests()


if __name__ == "__main__":
    test_qnn_max_pool2d()
