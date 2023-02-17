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
"""Adreno specific helpers."""
import tvm

from tvm import relay
from tvm.ir import IRModule

acc_dtype = "float32"


def mixed_precision_rule(call_node: "relay.Call", mixed_precision_type: str):
    global acc_dtype
    return [
        relay.transform.mixed_precision.MIXED_PRECISION_ALWAYS,
        acc_dtype,
        mixed_precision_type,
    ]


class AdrenoMixedPrecision(object):
    """Temporarily changes attr of ops to enable FP32 accumulation ."""

    def __init__(self):
        """Saves the required info for RAII pattern usage.
        Parameters
        ----------
        acc_dtype : atr
            accumulation dtype.
        """
        self.older_attr = {}
        self.ops = ["nn.conv2d", "nn.dense"]
        self.attr_key = "FTVMMixedPrecisionConversionType"

    def __enter__(self):
        for op_name in self.ops:
            op = relay.op.get(op_name)
            self.older_attr[op_name] = op.get_attr(self.attr_key)
            op.reset_attr(self.attr_key)
            op.set_attr(self.attr_key, mixed_precision_rule)
        return self

    def __exit__(self, ptype, value, trace):
        for op_name in self.ops:
            op = relay.op.get(op_name)
            op.reset_attr(self.attr_key)
            if self.older_attr[op_name]:
                op.set_attr(self.attr_key, self.older_attr[op_name])


def convert_to_dtype(mod, dtype):
    """Converts the operator datatypes"""

    global acc_dtype
    if dtype in ["float16", "float16_acc32"]:
        acc_dtype = "float16" if dtype == "float16" else "float32"

        mod = IRModule.from_expr(mod)
        with AdrenoMixedPrecision():
            seq = tvm.transform.Sequential(
                [relay.transform.InferType(), relay.transform.ToMixedPrecision()]
            )
            with tvm.transform.PassContext(
                config={"relay.ToMixedPrecision.keep_orig_output_dtype": True}, opt_level=3
            ):
                mod = seq(mod)
    else:
        print("Warn: Invald dtype conversion to ", dtype)
    return mod


@tvm.register_func("adreno.mixed_precision_fp16")
def mixed_precision_hook_fp16(mod, params):
    """TVMC hook api"""

    return convert_to_dtype(mod["main"], "float16")


@tvm.register_func("adreno.mixed_precision_fp16_acc32")
def mixed_precision_hook_fp16_acc32(mod, params):
    """TVMC hook api"""

    return convert_to_dtype(mod["main"], "float16_acc32")
