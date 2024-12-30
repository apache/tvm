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
from tvm import relax
import tvm.testing
from tvm.script.parser import ir as I, relax as R, tir as T
from tvm.relax.transform.legalize_ops import adreno as legalize_adreno
from tvm.ir.module import IRModule
from tvm.relax.expr_functor import PyExprMutator, PyExprVisitor, mutator, visitor


@visitor
class ValidateScope(PyExprVisitor):  # pylint: disable=abstract-method
    def __init__(self, scope_info: dict) -> None:
        self.scope_info = scope_info
        self.matched = True

    def visit(self, mod: IRModule) -> None:
        """Entry point"""
        for _, func in mod.functions_items():
            if isinstance(func, relax.Function):
                self.visit_expr(func)
        return self.matched

    def visit_call_(self, call: relax.Call) -> None:  # pylint: disable=arguments-renamed
        if call.op.name == "relax.call_tir":
            # if call.args[0].name_hint in self.scope_info:
            for idx, arg in enumerate(call.args[1]):
                arg_sinfo = arg.struct_info
                assert isinstance(
                    arg_sinfo, relax.TensorStructInfo
                ), f"Expected TensorStructInfo but git {type(arg_sinfo)}"
                assert (
                    arg_sinfo.vdevice.memory_scope
                    == self.scope_info[call.args[0].name_hint][0][idx]
                ), f"Scope mismatched for argument {idx} in {call.args[0].name_hint}"
            if isinstance(call.sinfo_args[0], relax.TensorStructInfo):
                assert (
                    call.sinfo_args[0].vdevice.memory_scope
                    == self.scope_info[call.args[0].name_hint][1][0]
                ), f"Scope mismatched for return scope: {call.args[0].name_hint}"
            else:
                assert isinstance(
                    call.sinfo_args[0], relax.TupleStructInfo
                ), f"Expected TupleStructInfo but git {type(call.sinfo_args[0])}"
                for idx, sinfo in enumerate(call.sinfo_args[0].fields):
                    assert (
                        sinfo.vdevice.memory_scope
                        == self.scope_info[call.args[0].name_hint][1][idx]
                    ), f"Scope mismatched for return scope for {idx} in {call.args[0].name_hint}"


def verify(mod, expected):
    tgt = tvm.target.Target("opencl --device=adreno", host="llvm")
    skip_ops = [
        "relax.nn.conv2d",
        "relax.nn.max_pool2d",
        "relax.nn.adaptive_avg_pool2d",
        # "relax.nn.layer_norm",
    ]
    with tgt:
        mod = tvm.tir.transform.BindTarget(tvm.target.Target.current(allow_none=False))(mod)
        mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
        mod = tvm.relax.transform.FoldConstant()(mod)
        desired_layouts = {"relax.nn.conv2d": ["NCHW4c", "OIHW4o", "NCHW4c"]}
        mod = tvm.relax.transform.ConvertLayout(desired_layouts)(mod)
        mod = tvm.relax.transform.Normalize()(mod)
        mod = tvm.relax.transform.FoldConstant()(mod)
        mod = tvm.relax.transform.LegalizeOps(skip_ops=skip_ops)(mod)
        mod = tvm.relax.transform.AnnotateTIROpPattern()(mod)
        mod = tvm.relax.backend.adreno.transform.AnnotateCustomMemoryScope(tgt)(mod)
        # There is a possibility of some skipped ops above might not use 5D layouts.
        mod = tvm.relax.transform.LegalizeOps()(mod)
        mod = tvm.relax.transform.LegalizeOps(
            {"relax.nn.conv2d": legalize_adreno.conv2d_NCHWc_OIHWo},
        )(mod)
        # Lets get pattern info for newly legalized ops
        mod = tvm.relax.transform.AnnotateTIROpPattern()(mod)
        mod = tvm.relax.transform.FoldConstant()(mod)
        mod = tvm.relax.transform.FuseOps()(mod)
        mod = tvm.relax.transform.FuseTIR()(mod)
        mod = tvm.relax.transform.DeadCodeElimination()(mod)
        mod = tvm.relax.backend.adreno.transform.FoldVDeviceScopeChange()(mod)
        mod = tvm.relax.transform.DeadCodeElimination()(mod)
        mod = tvm.relax.transform.SpecializePrimFuncBasedOnCallSite()(mod)
        mod = tvm.relax.transform.Normalize()(mod)
    print(mod)
    ValidateScope(expected).visit(mod)

def test_conv2d_conv2d_fallback_to_buffer_conv2d():
    """
        layout_transform (NCHW->NCHW4c)
                  |                      <- texture
                conv2d (1)               <- textures as output
               /         \
            conv2d (2)    conv2d (3)     <- conv2d (2) emits texture, conv2d (3) emits buffer
                 \       /               <- concat shouldn't support textures here
                concatenation
                     |                   <- buffer
               layout_transform (NCHW4c->NCHW)
    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 32, 40, 40), "float32"),
            w1: R.Tensor((96, 32, 2, 2), "float32"),
            w2: R.Tensor((32, 96, 2, 2), "float32"),
            w3: R.Tensor((5, 96, 2, 2), "float32"),
            bias1: R.Tensor((1, 96, 1, 1), "float32"),
            bias2: R.Tensor((1, 32, 1, 1), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv = R.nn.conv2d(x, w1, strides=[2, 2], out_dtype="float32")
                gv1 = R.add(gv, bias1)
                gv2 = R.nn.relu(gv1)
                gv3 = R.nn.conv2d(gv2, w2, strides=[2, 2], out_dtype="float32")
                gv4 = R.add(gv3, bias2)
                gv5 = R.nn.relu(gv4)
                gv6 = R.nn.conv2d(gv2, w3, strides=[2, 2], out_dtype="float32")
                gv7 = R.concat((gv3, gv6), axis=1)
                R.output(gv7)
            return gv7

    Expected = {
        "te_layout_transform": (["global"], ["global.texture-weight"]),
        "te_layout_transform1": (["global"], ["global.texture-weight"]),
        "te_layout_transform2": (["global"], ["global.texture-weight"]),
        "fused_conv2d_NCHWc_OIHWo_opencl_add_relu": (
            ["global.texture-weight", "global.texture-weight", "global.texture-weight"],
            ["global"],
        ),
        "te_layout_transform3": (["global"], ["global.texture-weight"]),
        "conv2d_NCHWc_OIHWo1_opencl": (
            ["global.texture-weight", "global.texture-weight"],
            ["global"],
        ),
        "te_layout_transform4": (["global"], ["global"]),
        "conv2d_opencl": (["global", "global"], ["global"]),
        "te_layout_transform5": (["global"], ["global"]),
        "concatenate": (["global", "global"], ["global"]),
    }
    verify(Input, Expected)

if __name__ == "__main__":
    tvm.testing.main()
