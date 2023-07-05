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
# under the License
"""Unit tests for the CapturePostDfsIndexInSpans debugging pass."""

import tvm
import tvm.testing
import numpy as np


def make_const(dtype, shape):
    return tvm.relay.const(np.random.rand(*shape).astype(dtype))


def make_consts(dtype, shapes):
    return [make_const(dtype, shape) for shape in shapes]


metatable = {
    "relay.Constant": make_consts(
        "float16",
        [
            (2304, 768),  # 0
            (2304,),  # 1
            (600, 32, 64),  # 2
        ],
    )
}


def input_mod():
    return tvm.relay.parse(
        """
        #[version = "0.0.5"]
        def @main(%x0 : Tensor[(1600, 768), float16], %x3 : Tensor[(600, 32, 64), float16]) -> (Tensor[(1600, 2304), float16], Tensor[(600, 32, 32), float16]) {
          %0 = nn.dense(%x0, meta[relay.Constant][0], units=2304);
          %1 = add(%0, meta[relay.Constant][1]);
          %2 = fn(%y_3_i0: Tensor[(600, 32, 64), float16], %y_3_i1: Tensor[(600, 32, 64), float16],
                  Inline=1, Compiler="cublas", global_symbol="tvmgen_default_cublas_main_3", Primitive=1) -> Tensor[(600, 32, 32), float16] {
            %6 = fn (%FunctionVar_0_01: Tensor[(600, 32, 64), float16], %FunctionVar_0_11: Tensor[(600, 32, 64), float16],
                     PartitionedFromPattern="nn.batch_matmul_", Composite="cublas.batch_matmul") -> Tensor[(600, 32, 32), float16] {
              nn.batch_matmul(%FunctionVar_0_01, %FunctionVar_0_11, out_dtype="float16", transpose_b=True)
            };
            %6(%y_3_i0, %y_3_i1)
          };
          %3 = %2(%x3, meta[relay.Constant][2]);
          (%1, %3)
        }
        """,
        "from_string",
        None,
        metatable,
    )


expected_pretty_printed_output_mod = r"""def @main(%x0: Tensor[(1600, 768), float16] /* ty=Tensor[(1600, 768), float16] span=index:0:5 */, %x3: Tensor[(600, 32, 64), float16] /* ty=Tensor[(600, 32, 64), float16] span=index:1:18 */) -> (Tensor[(1600, 2304), float16], Tensor[(600, 32, 32), float16]) {
  %0 = nn.dense(%x0, meta[relay.Constant][0] /* ty=Tensor[(2304, 768), float16] span=index:4:5 */, units=2304) /* ty=Tensor[(1600, 2304), float16] span=index:5:7 */;
  %2 = fn (%y_3_i0: Tensor[(600, 32, 64), float16] /* ty=Tensor[(600, 32, 64), float16] span=index:8:15 */, %y_3_i1: Tensor[(600, 32, 64), float16] /* ty=Tensor[(600, 32, 64), float16] span=index:9:15 */, Inline=1, Compiler="cublas", global_symbol="tvmgen_default_cublas_main_3", Primitive=1) -> Tensor[(600, 32, 32), float16] {
    %1 = fn (%FunctionVar_0_01: Tensor[(600, 32, 64), float16] /* ty=Tensor[(600, 32, 64), float16] span=index:10:13 */, %FunctionVar_0_11: Tensor[(600, 32, 64), float16] /* ty=Tensor[(600, 32, 64), float16] span=index:11:13 */, PartitionedFromPattern="nn.batch_matmul_", Composite="cublas.batch_matmul") -> Tensor[(600, 32, 32), float16] {
      nn.batch_matmul(%FunctionVar_0_01, %FunctionVar_0_11, out_dtype="float16", transpose_b=True) /* ty=Tensor[(600, 32, 32), float16] span=index:13:14 */
    } /* ty=fn (Tensor[(600, 32, 64), float16], Tensor[(600, 32, 64), float16]) -> Tensor[(600, 32, 32), float16] span=index:14:15 */;
    %1(%y_3_i0, %y_3_i1) /* ty=Tensor[(600, 32, 32), float16] span=index:15:16 */
  } /* ty=fn (Tensor[(600, 32, 64), float16], Tensor[(600, 32, 64), float16]) -> Tensor[(600, 32, 32), float16] span=index:16:18 */;
  %3 = add(%0, meta[relay.Constant][1] /* ty=Tensor[(2304), float16] span=index:6:7 */) /* ty=Tensor[(1600, 2304), float16] span=index:7:19 */;
  %4 = %2(%x3, meta[relay.Constant][2] /* ty=Tensor[(600, 32, 64), float16] span=index:17:18 */) /* ty=Tensor[(600, 32, 32), float16] span=index:18:19 */;
  (%3, %4) /* ty=(Tensor[(1600, 2304), float16], Tensor[(600, 32, 32), float16]) span=index:19:20 */
}

"""


def test_capture_index_in_spans():
    output_mod = str(tvm.relay.transform.CapturePostDfsIndexInSpans()(input_mod()))
    assert output_mod == expected_pretty_printed_output_mod


if __name__ == "__main__":
    tvm.testing.main()
