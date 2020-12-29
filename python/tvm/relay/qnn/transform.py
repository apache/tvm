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
# pylint: disable=invalid-name,arguments-differ,no-else-return,unused-argument,missing-docstring
"""
QNN pass transformation infrastructure.
"""
from tvm import relay


def CanonicalizeOps():
    """Converts/Lowers an expression containing QNN ops to an expression containing only core
    (non-Dialect) Relay ops. Each QNN op is lowered to a sequence of existing Relay ops. This is a
    target-independent pass. One can register the lowering/transformation function for this op using
    FTVMQnnCanonicalize attr_name for FTVMLegalize op attribute.  An example of this transformation
    is below

    Examples
    ________

    .. code-block:: python

        # Original expression
        qnn_expr = relay.qnn.op.requantize(y,
                                           input_scale=1,
                                           input_zero_point=0,
                                           output_scale=1,
                                           output_zero_point=0,
                                           out_dtype='int8')

        # We want to utilize all the existing Relay infrastructure. So, instead of supporting this
        # QNN requantize op, we convert it into a sequence of existing Relay operators.
        mod = tvm.IRModule.from_expr(qnn_expr)
        mod = relay.qnn.transform.CanonicalizeOps()(mod)
        relay_expr = mod['main']
        print(relay_expr)

        def @main(%quantized_data: Tensor[(200), int32]) -> Tensor[(200), int8] {
          %0 = cast(%quantized_data, dtype="int64") /* ty=Tensor[(200), int64] */;
          %1 = multiply(%0, 2 /* ty=int64 */) /* ty=Tensor[(200), int64] */;
          %2 = multiply(%1, 1073741824 /* ty=int64 */) /* ty=Tensor[(200), int64] */;
          %3 = add(%2, 1073741824 /* ty=int64 */) /* ty=Tensor[(200), int64] */;
          %4 = right_shift(%3, 31 /* ty=int64 */) /* ty=Tensor[(200), int64] */;
          %5 = add(0 /* ty=int64 */, %4) /* ty=Tensor[(200), int64] */;
          %6 = clip(%5, a_min=-128f, a_max=127f) /* ty=Tensor[(200), int64] */;
          cast(%6, dtype="int8") /* ty=Tensor[(200), int8] */
        }

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that canonicalizes QNN ops to Relay ops.
    """

    return relay.transform.Legalize("FTVMQnnCanonicalize")


def Legalize():
    """Legalizes QNN ops. As opposed to Relay Legalize, this one legalizes only QNN ops. One can
    register a transformation/legalization function for an op by using the FTVMQnnLegalize attr_name
    for FTVMLegalize op attribute. The isolation of QNN and Relay Legalize gives us separation of
    concerns, leading to a better software practice. The legalization can be configured to happen
    per target. An example of this type of legalization is shown below.

    Examples
    ________

    Suppose the original graph is as follows

            data(u8)  weight(u8)
                |       |
                |       |
               qnn.conv2d (int32)
                   |
                   |
                nn.relu (int32)

    Now, we know that Intel Cascade Lake has VNNI instructions to speedup convolution. However, it
    only works on u8 x i8 inputs. So, here, we can use QNN Legalize to transform the above graph as
    follows

            data(u8)  weight(u8)
               |          |
               |          |
               |     requantize(i8)
               |        |
               |        |
               qnn.conv2d (int32)
                   |
                   |
                 nn.relu (int32)

    In this legalization, since we have isolated legalization for QNN ops, it will only trigger the
    transformation for qnn.conv2d (and not nn.relu). This pass can be followed by CanonicalizeOps to
    further lower the qnn.requantize and qnn.conv2d into an expr containing only Relay ops.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that legalizes QNN ops.
    """

    return relay.transform.Legalize("FTVMQnnLegalize")
