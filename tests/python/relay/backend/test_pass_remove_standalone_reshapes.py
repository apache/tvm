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

# Exercises the RemoveStandaloneReshapes pass.

import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator
import tvm.testing
from tvm.script import tir as T


HOST_DEVICE = tvm.device("cpu")
HOST_TARGET = tvm.target.Target("llvm")

CPU_DEVICE = tvm.device("cpu")
CPU_TARGET = tvm.target.Target("llvm").with_host(HOST_TARGET)

CPU = tvm.target.VirtualDevice(CPU_DEVICE, CPU_TARGET)  # device_type=1


RemoveStandaloneReshapes = tvm._ffi.get_global_func("relay._transform.RemoveStandaloneReshapes")


class MarkReshapeOnlyMutator(ExprMutator):
    """A pass for marking call_lowered as ReshapeOnly where reshapes exist unfused"""

    def __init__(self):
        ExprMutator.__init__(self)

    def visit_call(self, call):
        if isinstance(call.args[0], tvm.ir.GlobalVar) and "reshape" in call.args[0].name_hint:
            # attrs = {"relay_attrs" : {"relay.reshape_only" : 1}}
            dict_attrs = tvm.ir.make_node("DictAttrs", **{"relay.reshape_only": 1})
            attrs = tvm.ir.make_node(
                "relay.attrs.CallLoweredAttrs", **{"metadata": {"relay_attrs": dict_attrs}}
            )
            return relay.Call(call.op, call.args, attrs)
        return super().visit_call(call)


# Reshape should not be removed if its the first layer in the network
def test_first_reshape():
    mod = tvm.ir.IRModule()

    @T.prim_func
    def reshape_primfunc(a: T.handle, d: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        D = T.match_buffer(d, [128, 128])

        for i, j in T.grid(128, 128):
            D[i, j] = A[i, j]

    metatable = {"VirtualDevice": [CPU]}
    reshape_ty = relay.FuncType(
        [
            relay.TensorType((128, 128), "float32"),
        ],
        relay.TensorType((128, 128), "float32"),
    )

    reshape_gv = relay.GlobalVar("reshape", type_annot=reshape_ty)
    mod[reshape_gv] = reshape_primfunc
    mod = tvm.relay.parse(
        """
        #[version = "0.0.5"]
        def @main(%x {virtual_device=meta[VirtualDevice][0]}: Tensor[(128, 128), float32],
                  virtual_device=meta[VirtualDevice][0]) {
          %1 = call_lowered(@reshape, (%x,) );
          let %x_14: Tensor[(128, 128), float32] = on_device(%1, virtual_device=meta[VirtualDevice][0], constrain_result=True);
          %x_14
        }
        """,
        "from_string",
        mod,
        metatable,
    )

    mod["main"] = MarkReshapeOnlyMutator().visit(mod["main"])
    mod = RemoveStandaloneReshapes()(mod)
    reshapes_present = any(["reshape" in gv.name_hint for gv in mod.get_global_vars()])
    assert reshapes_present, "Reshape should have been removed."
    return


# When reshape layer is the last one in the network
def test_last_reshape():
    mod = tvm.ir.IRModule()

    @T.prim_func
    def mul_primfunc(a: T.handle, b: T.handle, d: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        B = T.match_buffer(b, [128, 128])
        D = T.match_buffer(d, [128, 128])

        for i, j, k in T.grid(128, 128, 128):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                D[vi, vj] = A[vi, vk] * B[vj, vk]

    @T.prim_func
    def reshape_primfunc(a: T.handle, d: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        D = T.match_buffer(d, [128, 128])

        for i, j in T.grid(128, 128):
            D[i, j] = A[i, j]

    metatable = {"VirtualDevice": [CPU]}
    mul_ty = relay.FuncType(
        [
            relay.TensorType((128, 128), "float32"),
            relay.TensorType((128, 128), "float32"),
            relay.TensorType((128, 128), "float32"),
        ],
        relay.TensorType((128, 128), "float32"),
    )

    mul_gv = relay.GlobalVar("multiply", type_annot=mul_ty)
    mod[mul_gv] = mul_primfunc
    reshape_ty = relay.FuncType(
        [
            relay.TensorType((128, 128), "float32"),
        ],
        relay.TensorType((128, 128), "float32"),
    )

    reshape_gv = relay.GlobalVar("reshape", type_annot=reshape_ty)
    mod[reshape_gv] = reshape_primfunc
    mod = tvm.relay.parse(
        """
        #[version = "0.0.5"]
        def @main(%x {virtual_device=meta[VirtualDevice][0]}: Tensor[(128, 128), float32],
                  %y {virtual_device=meta[VirtualDevice][0]}: Tensor[(128, 128), float32],
                  %z {virtual_device=meta[VirtualDevice][0]}: Tensor[(128, 128), float32],
                  virtual_device=meta[VirtualDevice][0]) {
          %0 = call_lowered(@multiply, (%x, %y, %z));
          let %x_12: Tensor[(128, 128), float32] = on_device(%0, virtual_device=meta[VirtualDevice][0], constrain_result=True);
          %1 = call_lowered(@reshape, (%x_12,) );
          let %x_14: Tensor[(128, 128), float32] = on_device(%1, virtual_device=meta[VirtualDevice][0], constrain_result=True);
          %x_14
        }
        """,
        "from_string",
        mod,
        metatable,
    )

    # Expected main:
    ##[version = "0.0.5"]
    # def @main(%x /* ty=Tensor[(128, 128), float32] */) -> Tensor[(128, 128), float32] {
    #  %0 = (%x, %y, %z);
    #  %1 = call_lowered(@multiply, %0);
    #  let %x_12: Tensor[(128, 128), float32] = on_device(%1, constrain_result=True);
    #  let %x_14: Tensor[(128, 128), float32] = on_device(%1, constrain_result=True);
    #  %x_14
    # }

    mod["main"] = MarkReshapeOnlyMutator().visit(mod["main"])
    mod = RemoveStandaloneReshapes()(mod)
    reshapes_present = any(["reshape" in gv.name_hint for gv in mod.get_global_vars()])
    assert not reshapes_present, "Reshape should have been removed."
    return


# When reshape layer is not marked as reshape_only
def test_fused_reshape():
    mod = tvm.ir.IRModule()

    @T.prim_func
    def mul_primfunc(a: T.handle, b: T.handle, d: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        B = T.match_buffer(b, [128, 128])
        D = T.match_buffer(d, [128, 128])

        for i, j, k in T.grid(128, 128, 128):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                D[vi, vj] = A[vi, vk] * B[vj, vk]

    @T.prim_func
    def fused_reshape_primfunc(a: T.handle, d: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        D = T.match_buffer(d, [128, 128])

        for i, j in T.grid(128, 128):
            D[i, j] = A[i, j]

    metatable = {"VirtualDevice": [CPU]}
    mul_ty = relay.FuncType(
        [
            relay.TensorType((128, 128), "float32"),
            relay.TensorType((128, 128), "float32"),
            relay.TensorType((128, 128), "float32"),
        ],
        relay.TensorType((128, 128), "float32"),
    )

    mul_gv = relay.GlobalVar("multiply", type_annot=mul_ty)
    mod[mul_gv] = mul_primfunc
    reshape_ty = relay.FuncType(
        [
            relay.TensorType((128, 128), "float32"),
        ],
        relay.TensorType((128, 128), "float32"),
    )

    reshape_gv = relay.GlobalVar("fused_reshape", type_annot=reshape_ty)
    mod[reshape_gv] = fused_reshape_primfunc
    mod = tvm.relay.parse(
        """
        #[version = "0.0.5"]
        def @main(%x {virtual_device=meta[VirtualDevice][0]}: Tensor[(128, 128), float32],
                  %y {virtual_device=meta[VirtualDevice][0]}: Tensor[(128, 128), float32],
                  %z {virtual_device=meta[VirtualDevice][0]}: Tensor[(128, 128), float32],
                  virtual_device=meta[VirtualDevice][0]) {
          %0 = call_lowered(@multiply, (%x, %y, %z));
          let %x_12: Tensor[(128, 128), float32] = on_device(%0, virtual_device=meta[VirtualDevice][0], constrain_result=True);
          %1 = call_lowered(@fused_reshape, (%x_12,) );
          let %x_14: Tensor[(128, 128), float32] = on_device(%1, virtual_device=meta[VirtualDevice][0], constrain_result=True);
          %x_14
        }
        """,
        "from_string",
        mod,
        metatable,
    )

    # Expected main:
    ##[version = "0.0.5"]
    # def @main(%x /* ty=Tensor[(128, 128), float32] */) -> Tensor[(128, 128), float32] {
    #  %0 = (%x, %y, %z);
    #  %1 = call_lowered(@multiply, %0);
    #  let %x_12: Tensor[(128, 128), float32] = on_device(%1, constrain_result=True);
    #  let %x_14: Tensor[(128, 128), float32] = on_device(%1, constrain_result=True);
    #  %x_14
    # }

    mod = RemoveStandaloneReshapes()(mod)
    reshapes_present = any(["reshape" in gv.name_hint for gv in mod.get_global_vars()])
    assert reshapes_present, "Reshape should have been removed."
    return


if __name__ == "__main__":
    tvm.testing.main()
