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

roundings = ["UPWARD", "TONEAREST"]

def test_requantize():
    def verify(mod, goldens):
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod, "llvm", params=None)
            golden_data, golden_output = goldens
            rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
            rt_mod.set_input("quantized_data",golden_data)
            rt_mod.set_input(**params)
            rt_mod.run()
            res = rt_mod.get_output(0).asnumpy()
            np.testing.assert_equal(res, golden_output)

    def get_mod(data_shape, data_dtype, out_dtype, input_scale, output_scale,
            input_zero_point=0, output_zero_point=0, rounding="TONEAREST"):
        quantized_data = relay.var("quantized_data", shape=data_shape,
                dtype=data_dtype)
        mod = relay.qnn.op.requantize(
                quantized_data,
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                output_scale=output_scale,
                output_zero_point=output_zero_point,
                rounding=rounding,
                out_dtype=out_dtype)

        mod = relay.Function(relay.analysis.free_vars(mod), mod)
        mod = relay.Module.from_expr(mod)
        return mod

    def same_scale_test():
        # Have same scales, everything within range
        golden_data = np.arange(-100, 100, 1).astype('int32')
        golden_output = golden_data

        for rounding in roundings:
            mod = get_mod(data_shape=(200, ),
                          data_dtype='int32',
                          out_dtype="int8",
                          input_scale=0.5,
                          output_scale=0.5,
                          rounding=rounding)
            assert 'right_shift' not in mod.astext()
            verify(mod, (golden_data, golden_output))

    def downscale_test():
        for rounding in roundings:
            mod = get_mod(data_shape=(32, ),
                          data_dtype='int32',
                          out_dtype='int8',
                          input_scale=1,
                          output_scale=16,
                          rounding=rounding)

            # Try positive values
            # 8 corresponds to 0.5, resulting in 1
            golden_data = np.arange(0, 32, 1).astype('int32')
            golden_output = np.repeat([0, 1, 2], [8, 16, 8])
            verify(mod, (golden_data, golden_output))

            # Try negative values
            # -8 corresponds to -0.5. For UPWARD, this is 0
            golden_data = np.arange(0, -32, -1).astype('int32')
            if rounding == "UPWARD":
                golden_output = np.repeat([0, -1, -2], [9, 16, 7])
            else:
                golden_output = np.repeat([0, -1, -2], [8, 16, 8])
            verify(mod, (golden_data, golden_output))

            # Try a different scale
            mod = get_mod(data_shape=(32, ),
                          data_dtype='int32',
                          out_dtype="int8",
                          input_scale=1,
                          output_scale=4,
                          rounding=rounding)

            # Try positive values
            # 2I corresponds to 0.5, resulting in 1
            golden_data = np.arange(0, 32, 1).astype('int32')
            golden_output = np.repeat([0, 1, 2, 3, 4, 5, 6, 7, 8],
                                      [2, 4, 4, 4, 4, 4, 4, 4, 2])
            verify(mod, (golden_data, golden_output))

            # Try negative values
            # -8 corresponds to -0.5. For UPWARD, this is 0
            golden_data = np.arange(0, -32, -1).astype('int32')
            if rounding == "UPWARD":
                golden_output = np.repeat([0, -1, -2, -3, -4, -5, -6, -7, -8],
                                          [3, 4, 4, 4, 4, 4, 4, 4, 1])
            else:
                golden_output = np.repeat([0, -1, -2, -3, -4, -5, -6, -7, -8],
                                          [2, 4, 4, 4, 4, 4, 4, 4, 2])
            verify(mod, (golden_data, golden_output))

            # Try uint8 out_dtype
            mod = get_mod(data_shape=(32, ),
                          data_dtype='int32',
                          out_dtype='uint8',
                          input_scale=1,
                          output_scale=16,
                          rounding=rounding)

            # Try positive values
            # 8 corresponds to 0.5, resulting in 1
            golden_data = np.arange(0, 32, 1).astype('int32')
            golden_output = np.repeat([0, 1, 2], [8, 16, 8])
            verify(mod, (golden_data, golden_output))

            # Try uint8 in_dtyope and uint8 out_dtype
            mod = get_mod(data_shape=(32, ),
                          data_dtype='uint8',
                          out_dtype='uint8',
                          input_scale=1,
                          output_scale=16,
                          rounding=rounding)

            # Try positive values
            # 8 corresponds to 0.5, resulting in 1
            golden_data = np.arange(0, 32, 1).astype('int32')
            golden_output = np.repeat([0, 1, 2], [8, 16, 8])
            verify(mod, (golden_data, golden_output))

    def upscale_test():
        for rounding in roundings:
            mod = get_mod(data_shape=(32, ),
                          data_dtype='int32',
                          out_dtype="int8",
                          input_scale=2,
                          output_scale=1,
                          rounding=rounding)

            # Try positive values
            # 8 corresponds to 0.5, resulting in 1
            golden_data = np.arange(0, 32, 1).astype('int32')
            golden_output = np.multiply(2, golden_data)
            verify(mod, (golden_data, golden_output))

            # Try negative values
            # -8 corresponds to -0.5. For UPWARD, this is 0
            golden_data = np.arange(0, -32, -1).astype('int32')
            golden_output = np.multiply(2, golden_data)
            verify(mod, (golden_data, golden_output))

    def saturation_test():
        for rounding in roundings:
            mod = get_mod(data_shape=(16, ),
                          data_dtype='int32',
                          out_dtype="int8",
                          input_scale=0.5,
                          output_scale=0.5,
                          rounding=rounding)
            golden_data = np.arange(0, 16, 1).astype('int32')
            golden_data = np.add(120, golden_data)
            output = np.array([120, 121, 122, 123, 124, 125, 126, 127,
                               127, 127, 127, 127, 127, 127, 127, 127])
            golden_output = output
            verify(mod, (golden_data, golden_output))

            # Try negative numbers
            golden_data = np.arange(0, -16, -1).astype('int32')
            golden_data = np.add(-120, golden_data)
            output = np.array([-120, -121, -122, -123, -124, -125, -126, -127,
                               -128, -128, -128, -128, -128, -128, -128, -128])
            golden_output = output
            verify(mod, (golden_data, golden_output))

    def zero_point_test():
        # Output zero point
        for rounding in roundings:
            mod = get_mod(data_shape=(32, ),
                          data_dtype='int32',
                          out_dtype='int8',
                          input_scale=1,
                          output_scale=16,
                          output_zero_point=1,
                          rounding=rounding)

            # Try positive values
            # 8 corresponds to 0.5, resulting in 1
            golden_data = np.arange(0, 32, 1).astype('int32')
            golden_output = np.repeat([0, 1, 2], [8, 16, 8])
            golden_output = np.add(1, golden_output)
            verify(mod, (golden_data, golden_output))

            # Try negative values
            # -8 corresponds to -0.5. For UPWARD, this is 0
            golden_data = np.arange(-32, -64, -1).astype('int32')
            if rounding == "UPWARD":
                golden_output = np.repeat([-2, -3, -4], [9, 16, 7])
            else:
                golden_output = np.repeat([-2, -3, -4], [8, 16, 8])
            golden_output = np.add(1, golden_output)
            verify(mod, (golden_data, golden_output))

        # Input zero point
        for rounding in roundings:
            mod = get_mod(data_shape=(32, ),
                          data_dtype='int32',
                          out_dtype='int8',
                          input_scale=1,
                          output_scale=16,
                          input_zero_point=16,
                          rounding=rounding)

            # Try positive values
            golden_data = np.arange(32, 64, 1).astype('int32')
            golden_output = np.repeat([2, 3, 4], [8, 16, 8])
            golden_output = np.subtract(golden_output, 1)
            verify(mod, (golden_data, golden_output))

            # Try negative values
            golden_data = np.arange(-32, -64, -1).astype('int32')
            if rounding == "UPWARD":
                golden_output = np.repeat([-2, -3, -4], [9, 16, 7])
            else:
                golden_output = np.repeat([-2, -3, -4], [8, 16, 8])
            golden_output = np.subtract(golden_output, 1)
            verify(mod, (golden_data, golden_output))

    same_scale_test()
    downscale_test()
    upscale_test()
    saturation_test()
    zero_point_test()

if __name__ == "__main__":
    test_requantize()
