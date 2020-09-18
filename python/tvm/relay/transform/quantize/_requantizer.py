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

"""Removes extraneous qnn.quantize and qnn.dequantize from calibrated modules, and replaces them
with qnn.requanize ops."""
import math

import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import DFPatternCallback, wildcard, is_op, dominates, rewrite


class Requantizer:
    """Removes extraneous qnn.quantize and qnn.dequantize and replaces
    them with qnn.requantize."""

    class RequantizerCallback(DFPatternCallback):
        """First pass that inserts requantize ops, specifically taking
        qnn.dequantize -> qnn.quantize to qnn.requantize
        and
        qnn.dequantize -> int8_op* -> qnn.quantize to requantize -> int8_op*
        """

        def __init__(self):
            super().__init__()

            self.data = wildcard()
            self.dequantize_scale = wildcard()
            self.dequantize_zp = wildcard()

            self.quantize_scale = wildcard()
            self.quantize_zp = wildcard()

            # Ops that are permitted inbetween quantize and dequantize if we are
            # rewriting to requantize
            self.is_int_8_op = (
                is_op("nn.max_pool2d")(wildcard())
                | is_op("nn.max_pool2d")(wildcard())
                | is_op("nn.max_pool3d")(wildcard())
                | is_op("nn.relu")(wildcard())
                | is_op("transpose")(wildcard())
                | is_op("reshape")(wildcard())
                | is_op("nn.pad")(wildcard())
                | is_op("squeeze")(wildcard())
                | is_op("nn.global_avg_pool2d")
                | is_op("nn.batch_flatten")
                | is_op("copy")
                | is_op("mean")
                | is_op("sqrt")
            )

            # All ops in is_int_8_op must also be in self.op_map
            self.op_map = {
                relay.op.get("nn.max_pool2d"): relay.op.nn.max_pool2d,
                relay.op.get("nn.max_pool3d"): relay.op.nn.max_pool3d,
                relay.op.get("transpose"): relay.op.transpose,
                relay.op.get("reshape"): relay.op.reshape,
                relay.op.get("nn.pad"): relay.op.nn.pad,
                relay.op.get("squeeze"): relay.op.squeeze,
                relay.op.get("nn.global_avg_pool2d"): relay.op.nn.global_avg_pool2d,
                relay.op.get("nn.batch_flatten"): relay.op.nn.batch_flatten,
                relay.op.get("copy"): relay.op.copy,
                relay.op.get("mean"): relay.op.mean,
                relay.op.get("sqrt"): relay.op.sqrt,
            }

            # Main pattern -- quantize(is_int_8_op*(dequantize(data))) --
            # (with 1 or more is_int_8_ops)
            self.dequantize = is_op("qnn.dequantize")(
                self.data, self.dequantize_scale, self.dequantize_zp
            )

            self.dominator = dominates(self.dequantize, self.is_int_8_op, self.is_int_8_op)
            self.quantize = is_op("qnn.quantize")(
                self.dominator, self.quantize_scale, self.quantize_zp
            )

            # Pattern with the null path : quantize(dequantize(data)) -- (no is_int_8_op inbetween)
            # We have to do the null path outside the dominator pattern because of pattern matcher
            # limitations
            self.no_path_dequantize = is_op("qnn.dequantize")(
                self.data, self.dequantize_scale, self.dequantize_zp
            )
            self.no_path_quantize = is_op("qnn.quantize")(
                self.no_path_dequantize, self.quantize_scale, self.quantize_zp
            )

            self.pattern = self.quantize | self.no_path_quantize

        def callback(self, pre, post, node_map):
            # Extract data from the pattern
            data = node_map[self.data][0]
            dequantize_scale = node_map[self.dequantize_scale][0]
            deq_zp = node_map[self.dequantize_zp][0]

            quantize_scale = node_map[self.quantize_scale][0]
            quantize_zp = node_map[self.quantize_zp][0]

            # Case where there are no ops in between the dequantize and quantize
            if self.no_path_quantize in node_map:
                axis = node_map[self.no_path_dequantize][0].attrs.axis
                res = relay.qnn.op.requantize(
                    data, dequantize_scale, deq_zp, quantize_scale, quantize_zp, axis=axis
                )
            # Ops inbetween quantize and dequantize are dominated
            elif self.quantize in node_map:

                axis = node_map[self.dequantize][0].attrs.axis
                transformed_data = relay.qnn.op.requantize(
                    data, dequantize_scale, deq_zp, quantize_scale, quantize_zp, axis=axis
                )
                for i in range(len(node_map[self.is_int_8_op]) - 1, -1, -1):
                    call = node_map[self.is_int_8_op][i]
                    # Transform relu into max(zeropoint)
                    if call.op == relay.op.get("nn.relu"):
                        if (
                            quantize_zp.data.asnumpy()
                            == relay.const(0, dtype="int32").data.asnumpy()
                        ):
                            transformed_data = relay.op.nn.relu(transformed_data)
                        else:
                            transformed_data = relay.op.maximum(
                                transformed_data, relay.cast(quantize_zp, "int8")
                            )
                    elif call.op in self.op_map.keys():
                        transformed_data = self.op_map[call.op](transformed_data, **call.attrs)
                    else:
                        raise ValueError(
                            "Uh oh, %s is not copied properly in the requantizer. " % str(call.op)
                        )
                res = transformed_data
            return res

    class RequantizeChainCallback(DFPatternCallback):
        """Folds chains of requantizes into one requantize.
        requantize(scale_a, zp_a, scale_b, zp_b) -> requantize(scale_b, zp_b, scale_c, zp_c) becomes
        requantize(scale_a, zp_a, scale_c, zp_c)
        """

        # Takes a chain of requantizes and turns them into one requantize
        def __init__(self):
            super().__init__()
            self.data = wildcard()
            self.rq_parent_scale_in = wildcard()
            self.rq_parent_zp_in = wildcard()
            self.rq_parent_scale_out = wildcard()
            self.rq_parent_zp_out = wildcard()

            self.rq_child_scale_in = wildcard()
            self.rq_child_zp_in = wildcard()
            self.rq_child_scale_out = wildcard()
            self.rq_child_zp_out = wildcard()

            self.rq_parent = is_op("qnn.requantize")(
                self.data,
                self.rq_parent_scale_in,
                self.rq_parent_zp_in,
                self.rq_parent_scale_out,
                self.rq_parent_zp_out,
            )
            self.rq_child = is_op("qnn.requantize")(
                wildcard(),
                self.rq_child_scale_in,
                self.rq_child_zp_in,
                self.rq_child_scale_out,
                self.rq_child_zp_out,
            )

            self.pattern = dominates(self.rq_parent, self.rq_child, self.rq_child)

        def callback(self, pre, post, node_map):
            data = node_map[self.data][0]
            rq_parent = node_map[self.rq_parent][0]

            rq_parent_scale_in = node_map[self.rq_parent_scale_in][0]
            rq_parent_zp_in = node_map[self.rq_parent_zp_in][0]

            rq_parent_scale_out = node_map[self.rq_parent_scale_out][0]
            rq_parent_zp_out = node_map[self.rq_parent_zp_out][0]

            child_in_scales = node_map[self.rq_child_scale_in]
            child_in_zps = node_map[self.rq_child_zp_in]
            child_out_scales = node_map[self.rq_child_scale_out]
            child_out_zps = node_map[self.rq_child_zp_out]

            len_children = len(node_map[self.rq_child_scale_out])

            # Check to make sure output and input scales and zps match before we apply this
            # transformation
            out_scale = rq_parent_scale_out
            out_zp = rq_parent_zp_out

            for i in range(0, len_children):

                in_scale = child_in_scales[i]
                in_zp = child_in_zps[i]

                assert math.isclose(
                    out_scale.data.asnumpy(), in_scale.data.asnumpy(), rel_tol=1e-05, abs_tol=1e-05
                ) and math.isclose(
                    out_zp.data.asnumpy(), in_zp.data.asnumpy(), rel_tol=1e-05, abs_tol=1e-05
                ), (
                    "Out scales/zps should match in scales/zps. Indicates an internal issue "
                    "in the quantizer somewhere."
                )

                out_scale = child_out_scales[i]
                out_zp = child_out_zps[i]

            parent_axis = rq_parent.attrs["axis"]

            return relay.qnn.op.requantize(
                data, rq_parent_scale_in, rq_parent_zp_in, out_scale, out_zp, axis=parent_axis
            )

    class ConsolidateRequantizeandQuantize(DFPatternCallback):
        """Gets rid of unnecessary requantizes directly following a quantize. Takes
        quantize(scale_a, zp_a) -> requantize(scale_a, zp_a, scale_b, zp_b) to
        quantize(scale_b, zp_b)
        """

        def __init__(self):
            super().__init__()

            self.data = wildcard()
            self.q_scale = wildcard()
            self.q_zp = wildcard()

            self.rq_scale_out = wildcard()
            self.rq_zp_out = wildcard()
            self.rq_scale_in = wildcard()
            self.rq_zp_in = wildcard()

            self.quantize = is_op("qnn.quantize")(self.data, self.q_scale, self.q_zp)
            self.requantize = is_op("qnn.requantize")(
                self.quantize, self.rq_scale_in, self.rq_zp_in, self.rq_scale_out, self.rq_zp_out
            )

            self.pattern = self.requantize

        def callback(self, pre, post, node_map):

            data = node_map[self.data][0]
            requantize = node_map[self.requantize][0]

            q_scale = node_map[self.q_scale][0]
            q_zp = node_map[self.q_zp][0]

            np_q_scale = q_scale.data.asnumpy()
            np_q_zp = q_zp.data.asnumpy()

            rq_scale_in = node_map[self.rq_scale_in][0]
            rq_zp_in = node_map[self.rq_zp_in][0]

            np_rq_scale = rq_scale_in.data.asnumpy()
            np_rq_zp = rq_zp_in.data.asnumpy()
            assert math.isclose(
                np_q_scale, np_rq_scale, rel_tol=1e-05, abs_tol=1e-05
            ) and math.isclose(np_q_zp, np_rq_zp, rel_tol=1e-05, abs_tol=1e-05), (
                "Scales and zps should match between adjacent quantize and requantizes, "
                "indicates a problem earlier in quantization"
            )

            output_scale = node_map[self.rq_scale_out][0]
            output_zp = node_map[self.rq_zp_out][0]

            requantize_axis = requantize.attrs["axis"]
            # Rewrite subgraph to just one quantize
            return relay.qnn.op.quantize(data, output_scale, output_zp, axis=requantize_axis)

    def requantize(self, func):
        """Removes extraneous qnn.quantize and qnn.dequantize ops and replaces them with
        qnn.requantize ops.

        Parameters
        ----------
        func : relay.Function
            Function to requantize.
        """
        # We have to fold scale/zp expressions for requantize to work
        # AlterOpLayout adds some extra ops that mess things up
        optimize = tvm.transform.Sequential(
            [relay.transform.FoldConstant(), relay.transform.EliminateCommonSubexpr()]
        )

        mod = tvm.ir.IRModule.from_expr(func)
        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]):
            rewritten_func = optimize(mod)["main"]

        rewritten_func = rewrite(
            self.RequantizerCallback(), rewritten_func, allow_overlapping_groups=True
        )
        rewritten_func = rewrite(self.RequantizeChainCallback(), rewritten_func)
        rewritten_func = rewrite(self.ConsolidateRequantizeandQuantize(), rewritten_func)

        rewritten_mod = tvm.ir.IRModule.from_expr(rewritten_func)

        return rewritten_mod["main"]
