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

import pytest

pytest.importorskip("ethosu.vela")
import numpy as np

import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu import preprocess


def set_func_attr(func, compile_name, symbol_name):
    """
    Helper function to attach attributes to the external function.
    """
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", compile_name)
    func = func.with_attr("global_symbol", symbol_name)
    return func


def test_single_io():
    """
    This test will test the pass wont touch external functions that
    have a single input and a single output.
    """

    def create_graph():
        def create_external_func1(mod_, compiler_name, symbol_name):
            x_int = relay.var("x_int", shape=(10, 10))
            z0 = relay.nn.relu(x_int)
            f1 = relay.Function([x_int], z0)
            f1 = set_func_attr(f1, compiler_name, symbol_name)
            glb_f1 = relay.GlobalVar(symbol_name)
            mod_[glb_f1] = f1
            mod_ = relay.transform.InferType()(mod_)
            return glb_f1, mod_

        mod = tvm.IRModule()
        x = relay.var("x", shape=(10, 10))

        glb_symbol_f1, mod = create_external_func1(mod, "ethos-u", "ethosu_0")
        r = relay.Call(glb_symbol_f1, [x])
        main = relay.Function([x], r)
        mod["main"] = main
        mod = relay.transform.InferType()(mod)
        return mod

    mod = create_graph()
    exp = create_graph()
    mod = preprocess.preprocess_ext_io()(mod)
    assert tvm.ir.structural_equal(mod, exp, map_free_vars=True)


def test_2ins_single_out():
    """
    The test is check two inputs and a single output of external function
    """

    def create_graph():
        def create_external_func1(mod_, compiler_name, symbol_name):
            x_int = relay.var("x_int", shape=(10, 10))
            w0_int = relay.var("w0_int", shape=(10, 10))
            z0 = relay.add(x_int, w0_int)

            f1 = relay.Function([x_int, w0_int], z0)
            f1 = set_func_attr(f1, compiler_name, symbol_name)
            glb_f1 = relay.GlobalVar(symbol_name)
            mod_[glb_f1] = f1
            mod_ = relay.transform.InferType()(mod_)
            return glb_f1, mod_

        mod = tvm.IRModule()

        x = relay.var("x", shape=(10, 10))
        w0 = relay.var("w0", shape=(10, 10))

        glb_symbol_f1, mod = create_external_func1(mod, "ethos-u", "ethosu_0")
        r = relay.Call(glb_symbol_f1, [x, w0])
        main = relay.Function([x, w0], r)
        mod["main"] = main
        mod = relay.transform.InferType()(mod)
        return mod

    def expected():
        def create_external_func1(mod_, compiler_name, symbol_name):
            ifms_int = relay.var("ifms_int", shape=[200])

            # splits
            (x_int_flat, w0_int_flat) = relay.split(ifms_int, [100])
            # reshapes
            x_int = relay.reshape(x_int_flat, newshape=(10, 10))
            w0_int = relay.reshape(w0_int_flat, newshape=(10, 10))

            z0 = relay.add(x_int, w0_int)
            f1 = relay.Function([ifms_int], z0)
            f1 = set_func_attr(f1, compiler_name, symbol_name)
            glb_f1 = relay.GlobalVar(symbol_name)
            mod_[glb_f1] = f1
            mod_ = relay.transform.InferType()(mod_)
            return glb_f1, mod_

        mod = tvm.IRModule()

        x = relay.var("x", shape=(10, 10))
        w0 = relay.var("w0", shape=(10, 10))

        # reshapes
        x_reshaped = relay.reshape(x, newshape=100)
        w0_reshaped = relay.reshape(w0, newshape=100)

        # concat
        ifms = relay.concatenate((x_reshaped, w0_reshaped), 0)

        glb_symbol_f1, mod = create_external_func1(mod, "ethos-u", "ethosu_0")
        r = relay.Call(glb_symbol_f1, [ifms])
        main = relay.Function([x, w0], r)
        mod["main"] = main
        mod = relay.transform.InferType()(mod)
        return mod

    mod = create_graph()
    exp = expected()
    mod = preprocess.preprocess_ext_io()(mod)
    assert tvm.ir.structural_equal(mod, exp, map_free_vars=True)


def test_single_in_2outs():
    """
    The test is to check a single input and two outputs of external function
    """

    def create_graph():
        def create_external_func1(mod_, compiler_name, symbol_name):
            x_int = relay.var("x_int", shape=(10, 10))

            p0 = relay.nn.relu(x_int)
            q0 = relay.tanh(x_int)
            f1_o_tuple = relay.Tuple([p0, q0])

            f1 = relay.Function([x_int], f1_o_tuple)
            f1 = set_func_attr(f1, compiler_name, symbol_name)
            glb_f1 = relay.GlobalVar(symbol_name)
            mod_[glb_f1] = f1
            mod_ = relay.transform.InferType()(mod_)
            return glb_f1, mod_

        mod = tvm.IRModule()
        x = relay.var("x", shape=(10, 10))
        glb_symbol_f1, mod = create_external_func1(mod, "ethos-u", "ethosu_0")
        pq_tuple = relay.Call(glb_symbol_f1, [x])
        p0 = relay.TupleGetItem(pq_tuple, 0)
        q0 = relay.TupleGetItem(pq_tuple, 1)
        r = relay.concatenate((p0, q0), axis=0)
        main = relay.Function([x], r)
        mod["main"] = main
        mod = relay.transform.InferType()(mod)
        return mod

    def expected():
        def create_external_func1(mod_, compiler_name, symbol_name):
            x_int = relay.var("x_int", shape=(10, 10))

            p0 = relay.nn.relu(x_int)
            q0 = relay.tanh(x_int)

            # reshapes
            p0_reshaped = relay.reshape(p0, newshape=100)
            q0_reshaped = relay.reshape(q0, newshape=100)
            ofms = relay.concatenate((p0_reshaped, q0_reshaped), 0)

            f1 = relay.Function([x_int], ofms)
            f1 = set_func_attr(f1, compiler_name, symbol_name)
            glb_f1 = relay.GlobalVar(symbol_name)
            mod_[glb_f1] = f1
            mod_ = relay.transform.InferType()(mod_)
            return glb_f1, mod_

        mod = tvm.IRModule()
        x = relay.var("x", shape=(10, 10))
        glb_symbol_f1, mod = create_external_func1(mod, "ethos-u", "ethosu_0")
        ofms = relay.Call(glb_symbol_f1, [x])

        # splits
        (p0_flat, q0_flat) = relay.split(ofms, [100])
        # reshapes
        p0_flat_reshaped = relay.reshape(p0_flat, newshape=(10, 10))
        q0_flat_reshaped = relay.reshape(q0_flat, newshape=(10, 10))
        # original output
        tuple_out = relay.Tuple([p0_flat_reshaped, q0_flat_reshaped])

        p0 = relay.TupleGetItem(tuple_out, 0)
        q0 = relay.TupleGetItem(tuple_out, 1)
        r = relay.concatenate((p0, q0), axis=0)
        main = relay.Function([x], r)
        mod["main"] = main
        mod = relay.transform.InferType()(mod)
        return mod

    mod = create_graph()
    exp = expected()
    mod = relay.transform.InferType()(mod)
    mod = preprocess.preprocess_ext_io()(mod)
    assert tvm.ir.structural_equal(mod, exp, map_free_vars=True)


def test_4ins_2outs():
    """
    The test is to check a 4 inputs and two outputs of external function.
    This just stand as a general test for multiple ins/outs.
    """

    def create_graph():
        def create_external_func1(mod_, compiler_name, symbol_name):
            x_int = relay.var("x_int", shape=(10, 10))
            w0_int = relay.var("w0_int", shape=(10, 10))
            w1_int = relay.var("w1_int", shape=(10, 10))
            w2_int = relay.var("w2_int", shape=(10, 10))

            z0 = relay.add(x_int, w0_int)
            p0 = relay.subtract(z0, w1_int)
            q0 = relay.multiply(z0, w2_int)
            f1_o_tuple = relay.Tuple([p0, q0])

            f1 = relay.Function([x_int, w0_int, w1_int, w2_int], f1_o_tuple)
            f1 = set_func_attr(f1, compiler_name, symbol_name)
            glb_f1 = relay.GlobalVar(symbol_name)
            mod_[glb_f1] = f1
            mod_ = relay.transform.InferType()(mod_)
            return glb_f1, mod_

        mod = tvm.IRModule()

        x = relay.var("x", shape=(10, 10))
        w0 = relay.var("w0", shape=(10, 10))
        w1 = relay.var("w1", shape=(10, 10))
        w2 = relay.var("w2", shape=(10, 10))

        glb_symbol_f1, mod = create_external_func1(mod, "ethos-u", "ethosu_0")
        pq_tuple = relay.Call(glb_symbol_f1, [x, w0, w1, w2])

        p0 = relay.TupleGetItem(pq_tuple, 0)
        q0 = relay.TupleGetItem(pq_tuple, 1)
        r = relay.concatenate((p0, q0), axis=0)
        main = relay.Function([x, w0, w1, w2], r)
        mod["main"] = main
        mod = relay.transform.InferType()(mod)
        return mod

    def expected():
        def create_external_func1(mod_, compiler_name, symbol_name):
            ifms_int = relay.var("ifms_int", shape=[400])

            # splits
            (x_int_flat, w0_int_flat, w1_int_flat, w2_int_flat) = relay.split(
                ifms_int, [100, 200, 300]
            )
            # reshapes
            x_int = relay.reshape(x_int_flat, newshape=(10, 10))
            w0_int = relay.reshape(w0_int_flat, newshape=(10, 10))
            w1_int = relay.reshape(w1_int_flat, newshape=(10, 10))
            w2_int = relay.reshape(w2_int_flat, newshape=(10, 10))

            z0 = relay.add(x_int, w0_int)
            p0 = relay.subtract(z0, w1_int)
            q0 = relay.multiply(z0, w2_int)
            # f1_o_tuple = relay.Tuple([p0, q0])

            # reshapes
            p0_reshaped = relay.reshape(p0, newshape=100)
            q0_reshaped = relay.reshape(q0, newshape=100)
            ofms = relay.concatenate((p0_reshaped, q0_reshaped), 0)

            f1 = relay.Function([ifms_int], ofms)
            f1 = set_func_attr(f1, compiler_name, symbol_name)
            glb_f1 = relay.GlobalVar(symbol_name)
            mod_[glb_f1] = f1
            mod_ = relay.transform.InferType()(mod_)
            return glb_f1, mod_

        mod = tvm.IRModule()

        x = relay.var("x", shape=(10, 10))
        w0 = relay.var("w0", shape=(10, 10))
        w1 = relay.var("w1", shape=(10, 10))
        w2 = relay.var("w2", shape=(10, 10))

        # reshapes
        x_reshaped = relay.reshape(x, newshape=100)
        w0_reshaped = relay.reshape(w0, newshape=100)
        w1_reshaped = relay.reshape(w1, newshape=100)
        w2_reshaped = relay.reshape(w2, newshape=100)

        # concat
        ifms = relay.concatenate((x_reshaped, w0_reshaped, w1_reshaped, w2_reshaped), 0)

        # call
        glb_func, mod = create_external_func1(mod, "ethos-u", "ethosu_0")
        ofms = relay.Call(glb_func, [ifms])

        # splits
        (p0_flat, q0_flat) = relay.split(ofms, [100])
        # reshapes
        p0_flat_reshaped = relay.reshape(p0_flat, newshape=(10, 10))
        q0_flat_reshaped = relay.reshape(q0_flat, newshape=(10, 10))
        # original output
        tuple_out = relay.Tuple([p0_flat_reshaped, q0_flat_reshaped])

        p0 = relay.TupleGetItem(tuple_out, 0)
        q0 = relay.TupleGetItem(tuple_out, 1)

        r = relay.concatenate((p0, q0), axis=0)
        main = relay.Function([x, w0, w1, w2], r)
        mod["main"] = main
        mod = relay.transform.InferType()(mod)
        return mod

    mod = create_graph()
    exp = expected()
    mod = preprocess.preprocess_ext_io()(mod)
    assert tvm.ir.structural_equal(mod, exp, map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
