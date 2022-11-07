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

# Exercises the LowerTE pass.

import tvm
import tvm.testing
import logging

logging.basicConfig()
logger = logging.getLogger("test_pass_lower_te")
logger.setLevel(logging.INFO)

# Since the TE compiler needs a good refactor it has not been exposed as a 'standard' pass
# in relay.transform. For testing grab it directly.
LowerTE = tvm._ffi.get_global_func("relay.tec.LowerTE")


def transform(mod):
    logger.info("Starting module:\n%s", mod)
    host_target = tvm.target.Target("llvm")
    prim_target = tvm.target.Target("llvm", host=host_target)
    ctxt = tvm.transform.PassContext()
    config = tvm.target.make_compilation_config(ctxt, prim_target)
    mod = tvm.relay.transform.PlanDevices(config)(mod)
    mod = tvm.relay.transform.InferType()(mod)
    mod = LowerTE("test", config)(mod)
    mod = tvm.relay.transform.InferType()(mod)
    logger.info("After LowerTE:\n%s", mod)
    return mod


# All attempts to use structural equalty tests against an expected IRModule parsed from
# Relay text were thwarted by the difficulty of setting up the expected call_lower attributes
# with the right GlobalVar instances. So the following assert structural correctness the hard way.


def test_lower_primitive():
    input_mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%a: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
          %0 = fn(%x : Tensor[(5, 7), float32], %y : Tensor[(5, 7), float32], Primitive=1) -> Tensor[(5, 7), float32] {
            add(%x, %y)
          };
          %0(%a, %a)
        }
        """,
        "from_string",
        None,
        None,
    )

    actual_mod = transform(input_mod)

    # Expected:
    #   def @main(%a: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
    #     %0 = (%a, %a);
    #     call_lowered(@test_fused_add, %0, metadata={relay_attrs={Primitive=1},all_prim_fn_vars=[@test_fused_add]})
    #   }
    #   def @test_fused_add = <lowered PrimFunc>

    main = actual_mod["main"]
    call = main.body
    assert call.op.name == "call_lowered"
    assert len(call.args) == 2
    assert call.args[0].name_hint == "test_fused_add"
    assert len(call.args[1].fields) == 2
    assert call.args[1].fields[0].name_hint == "a"
    assert call.args[1].fields[1].name_hint == "a"
    assert call.attrs.metadata["relay_attrs"].Primitive == 1
    assert len(call.attrs.metadata["all_prim_fn_vars"]) == 1
    assert call.attrs.metadata["all_prim_fn_vars"][0].name_hint == "test_fused_add"

    test_fused_add = actual_mod["test_fused_add"]
    assert isinstance(test_fused_add, tvm.tir.PrimFunc)


def test_lower_compiler():
    @tvm._ffi.register_func("relay.ext.test_pass_lower_te")
    def relay_ext_test_pass_lower_te(func):
        return None

    input_mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%a: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
          %0 = fn(%x : Tensor[(5, 7), float32], %y : Tensor[(5, 7), float32], Primitive=1, Compiler="test_pass_lower_te", global_symbol="test_add") -> Tensor[(5, 7), float32] {
            add(%x, %y)
          };
          %0(%a, %a)
        }
        """,
        "from_string",
        None,
        None,
    )

    actual_mod = transform(input_mod)

    # Expected:
    #   def @main(%a : Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
    #     %0 = (%a, %a)
    #     call_lowered(@test_add , %0, metadata={relay_attrs={Primitive=1, Compiler="test_pass_lower_te", global_symbol="test_add"}}, all_prim_fn_vars=[]})
    #   }
    #   def @test_add(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32], Extern=1) -> Tensor[(5, 7), float32] {
    #     add(%x, %y)
    #   }

    main = actual_mod["main"]
    call = main.body
    assert call.op.name == "call_lowered"
    assert len(call.args) == 2
    assert call.args[0].name_hint == "test_add"
    assert len(call.args[1].fields) == 2
    assert call.args[1].fields[0].name_hint == "a"
    assert call.args[1].fields[1].name_hint == "a"
    assert call.attrs.metadata["relay_attrs"].Primitive == 1
    assert call.attrs.metadata["relay_attrs"].Compiler == "test_pass_lower_te"
    assert call.attrs.metadata["relay_attrs"].global_symbol == "test_add"
    assert len(call.attrs.metadata["all_prim_fn_vars"]) == 0

    test_add = actual_mod["test_add"]
    assert isinstance(test_add, tvm.relay.Function)
    assert test_add.attrs["Extern"] == 1


def test_lower_extern():
    input_mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%a: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
          @my_add(%a, %a)
        }
        def @my_add(%x : Tensor[(5, 7), float32], %y : Tensor[(5, 7), float32], Extern=1) -> Tensor[(5, 7), float32] {
          add(%x, %y)
        }
        """,
        "from_string",
        None,
        None,
    )

    actual_mod = transform(input_mod)

    # Expected:
    #   def @main(%a: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
    #     %0 = (%a, %a);
    #     call_lowered(@my_add, %0, metadata={relay_attrs={Extern=1}}, all_prim_fn_vars=[]})
    #   }
    #   def @my_add(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32], Extern=1) -> Tensor[(5, 7), float32] {
    #     add(%x, %y)
    #   }

    main = actual_mod["main"]
    call = main.body
    assert call.op.name == "call_lowered"
    assert len(call.args) == 2
    assert call.args[0].name_hint == "my_add"
    assert len(call.args[1].fields) == 2
    assert call.args[1].fields[0].name_hint == "a"
    assert call.args[1].fields[1].name_hint == "a"
    assert call.attrs.metadata["relay_attrs"].Extern == 1
    assert len(call.attrs.metadata["all_prim_fn_vars"]) == 0

    test_add = actual_mod["my_add"]
    assert isinstance(test_add, tvm.relay.Function)
    assert test_add.attrs["Extern"] == 1


def test_lower_extern_with_dynamic_shape():
    input_mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%a: Tensor[(5, 7), float32]) -> Tensor[(?, ?), float32] {
          @my_dyn(%a, %a)
        }
        def @my_dyn(%x : Tensor[(5, 7), float32], %y : Tensor[(5, 7), float32], Extern=1) -> Tensor[(?, ?), float32] {
          add(%x, %y)
        }
        """,
        "from_string",
        None,
        None,
    )

    actual_mod = transform(input_mod)

    # Expected:
    # def @main(%a: Tensor[(5, 7), float32]) -> Tensor[(?, ?), float32] {
    #   %0 = (%a, %a);
    #   call_lowered(@my_dyn, %0, metadata={prim_shape_fn_var='test_shape_func_add', relay_attrs={Extern=1}, prim_shape_fn_states=[2, 2], prim_shape_fn_num_inputs=2, all_prim_shape_fn_vars=['shape_func_add'], prim_shape_fn_num_outputs=1, all_prim_fn_vars=[]})
    # }
    # def @my_dyn(%x: Tensor[(5, 7), float32] , %y: Tensor[(5, 7), float32] , Extern=1) -> Tensor[(?, ?), float32] {
    #   add(%x, %y)
    # }
    # def @test_shape_func_add = <shape PrimFunc>

    main = actual_mod["main"]
    call = main.body
    assert call.op.name == "call_lowered"
    assert len(call.args) == 2
    assert call.args[0].name_hint == "my_dyn"
    assert len(call.args[1].fields) == 2
    assert call.args[1].fields[0].name_hint == "a"
    assert call.args[1].fields[1].name_hint == "a"
    assert call.attrs.metadata["prim_shape_fn_var"].name_hint == "test_shape_func_add"
    assert call.attrs.metadata["relay_attrs"].Extern == 1
    assert len(call.attrs.metadata["prim_shape_fn_states"]) == 2
    assert call.attrs.metadata["prim_shape_fn_states"][0] == 2
    assert call.attrs.metadata["prim_shape_fn_states"][1] == 2
    assert call.attrs.metadata["prim_shape_fn_num_inputs"] == 2
    assert len(call.attrs.metadata["all_prim_shape_fn_vars"]) == 1
    assert call.attrs.metadata["all_prim_shape_fn_vars"][0].name_hint == "test_shape_func_add"
    assert call.attrs.metadata["prim_shape_fn_num_outputs"] == 1
    assert len(call.attrs.metadata["all_prim_fn_vars"]) == 0

    my_dyn = actual_mod["my_dyn"]
    assert isinstance(my_dyn, tvm.relay.Function)
    assert my_dyn.attrs["Extern"] == 1

    shape_func_add = actual_mod["test_shape_func_add"]
    assert isinstance(shape_func_add, tvm.tir.PrimFunc)


if __name__ == "__main__":
    tvm.testing.main()
