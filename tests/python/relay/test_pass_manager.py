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
"""Unit tests for relay pass manager."""
import numpy as np
import pytest

import tvm
from tvm import te
from tvm import relay
from tvm.relay import ExprFunctor
from tvm.relay import Function, Call
from tvm.relay import analysis
from tvm.relay import transform as _transform
from tvm.relay.testing import ctx_list, run_infer_type


def get_var_func():
    shape = (5, 10)
    tp = relay.TensorType(shape, "float32")
    x = relay.var("x", tp)
    gv = relay.GlobalVar("myAbs")
    func = relay.Function([x], relay.abs(x))
    return gv, func


def extract_var_func(mod, name):
    var = mod.get_global_var(name)
    func = mod[var]
    return var, func


def update_func(func):
    # Double the value of Constants and vars.
    class DoubleValues(ExprFunctor):
        def __init__(self):
            ExprFunctor.__init__(self)

        def visit_constant(self, const):
            return relay.add(const, const)

        def visit_var(self, var):
            return relay.add(var, var)

        def visit_call(self, call):
            new_op = self.visit(call.op)
            new_args = [self.visit(arg) for arg in call.args]
            return Call(new_op, new_args, call.attrs)

        def visit_global_var(self, gvar):
            return gvar

        def visit_op(self, op):
            return op

        def visit_function(self, fn):
            new_body = self.visit(fn.body)
            return Function(
                list(fn.params), new_body, fn.ret_type, fn.type_params,
                fn.attrs)

    double_value = DoubleValues()
    return double_value.visit(func)


class OptTester():
    """A helper class for testing the pass manager."""

    def __init__(self, mod):
        if not isinstance(mod, tvm.IRModule):
            raise TypeError("mod is expected to be the type of "
                            "tvm.IRModule")
        self.mod = mod

    def analysis(self):
        """Perform analysis for the current module."""
        pass

    @staticmethod
    def transform(node, ctx=None):
        """Perform optimization on node."""
        if isinstance(node, tvm.IRModule):
            # Add a function to the module and return an updated module.
            gv, func = get_var_func()
            mod = tvm.IRModule({gv: func})
            mod.update(node)
            return mod
        if isinstance(node, relay.Function):
            return update_func(node)

        raise TypeError("Found not supported node type.")


def get_rand(shape, dtype='float32'):
    return tvm.nd.array(np.random.rand(*shape).astype(dtype))


def check_func(func, ref_func):
    func = run_infer_type(func)
    ref_func = run_infer_type(ref_func)
    assert tvm.ir.structural_equal(func, ref_func)


def test_module_pass():
    shape = (5, 10)
    dtype = 'float32'
    tp = relay.TensorType(shape, dtype)
    x = relay.var("x", tp)
    y = relay.var("y", tp)
    v_add = relay.GlobalVar("myAdd")
    func = relay.Function([x, y], x + y)
    mod = tvm.IRModule({v_add: func})

    pass_name = "module_pass_test"
    opt_level = 0
    opt_tester = OptTester(mod)
    pass_ctx = None

    @tvm.transform.module_pass(opt_level=opt_level, name=pass_name)
    def transform(expr, ctx):
        return opt_tester.transform(expr, ctx)

    def test_pass_registration():
        mod_pass = transform
        assert isinstance(mod_pass, tvm.transform.ModulePass)
        pass_info = mod_pass.info
        assert pass_info.name == pass_name
        assert pass_info.opt_level == opt_level

    def test_pass_registration_no_decorator():
        def direct_transform(expr, ctx):
            return opt_tester.transform(expr, ctx)
        mod_pass = tvm.transform.module_pass(direct_transform, opt_level=3)
        assert isinstance(mod_pass, tvm.transform.ModulePass)
        pass_info = mod_pass.info
        assert pass_info.name == "direct_transform"
        assert pass_info.opt_level == 3

    def test_pass_run():
        module_pass = transform
        assert pass_name in str(module_pass)

        updated_mod = module_pass(mod)
        assert isinstance(updated_mod, tvm.IRModule)

        # Check the abs function in the updated module.
        v_abs, myabs = get_var_func()
        new_v_add = updated_mod.get_global_var(v_abs.name_hint)
        new_abs = updated_mod[new_v_add]
        check_func(new_abs, myabs)

        # Check the add function in the updated module.
        v_abs, myabs = get_var_func()
        new_v_add = updated_mod.get_global_var(v_add.name_hint)
        new_add = updated_mod[new_v_add]
        check_func(new_add, func)

        # Check the add function in the python transformed module.
        ret = opt_tester.transform(mod, pass_ctx)
        transformed_v_add = ret.get_global_var(v_add.name_hint)
        transformed_add = mod[transformed_v_add]
        check_func(new_add, transformed_add)

        # Execute the add function.
        x_nd = get_rand(shape, dtype)
        y_nd = get_rand(shape, dtype)
        ref_res = x_nd.asnumpy() + y_nd.asnumpy()
        for target, ctx in ctx_list():
            exe1 = relay.create_executor("graph", ctx=ctx, target=target)
            exe2 = relay.create_executor("debug", ctx=ctx, target=target)
            res1 = exe1.evaluate(new_add)(x_nd, y_nd)
            tvm.testing.assert_allclose(res1.asnumpy(), ref_res, rtol=1e-5)
            res2 = exe2.evaluate(new_add)(x_nd, y_nd)
            tvm.testing.assert_allclose(res2.asnumpy(), ref_res, rtol=1e-5)

    test_pass_registration()
    test_pass_registration_no_decorator
    test_pass_run()


def test_function_class_pass():
    @relay.transform.function_pass(opt_level=1)
    class TestReplaceFunc:
        """Simple test function to replace one argument to another."""
        def __init__(self, new_func):
            self.new_func = new_func

        def transform_function(self, func, mod, ctx):
            return self.new_func

    x = relay.var("x", shape=(10, 20))
    f1 = relay.Function([x], x)
    f2 = relay.Function([x], relay.log(x))
    fpass = TestReplaceFunc(f1)
    assert fpass.info.opt_level == 1
    assert fpass.info.name == "TestReplaceFunc"
    mod = tvm.IRModule.from_expr(f2)
    mod = fpass(mod)
    # wrap in expr
    mod2 = tvm.IRModule.from_expr(f1)
    assert tvm.ir.structural_equal(mod["main"], mod2["main"])


def test_function_pass():
    shape = (10, )
    dtype = 'float32'
    tp = relay.TensorType(shape, dtype)
    x = relay.var("x", tp)
    v_log = relay.GlobalVar("myLog")
    log = relay.Function([x], relay.log(x))
    mod = tvm.IRModule({v_log: log})

    pass_name = "function_pass_test"
    opt_level = 1
    opt_tester = OptTester(mod)
    pass_ctx = None

    @_transform.function_pass(opt_level=opt_level, name=pass_name)
    def transform(expr, mod, ctx):
        return opt_tester.transform(expr, ctx)

    def get_ref_log():
        ref_log = relay.Function([x], relay.log(relay.add(x, x)))
        return ref_log

    def test_pass_registration():
        function_pass = transform
        assert isinstance(function_pass, _transform.FunctionPass)
        pass_info = function_pass.info
        assert pass_info.name == pass_name
        assert pass_info.opt_level == opt_level

    def test_pass_registration_no_decorator():
        def direct_transform(expr, ctx):
            return opt_tester.transform(expr, ctx)
        mod_pass = _transform.function_pass(direct_transform, opt_level=0)
        assert isinstance(mod_pass, _transform.FunctionPass)
        pass_info = mod_pass.info
        assert pass_info.name == "direct_transform"
        assert pass_info.opt_level == 0

    def test_pass_run():
        function_pass = transform
        assert pass_name in str(function_pass)

        updated_mod = function_pass(mod)
        assert isinstance(updated_mod, tvm.IRModule)

        # Check the log function in the updated module.
        new_v_log = updated_mod.get_global_var(v_log.name_hint)
        new_log = updated_mod[new_v_log]
        check_func(new_log, get_ref_log())

        # Check the log function in the python transformed function.
        ret = opt_tester.transform(log, pass_ctx)
        check_func(new_log, ret)

        # Execute the add function.
        x_nd = get_rand(shape, dtype)
        ref_res = np.log(x_nd.asnumpy() * 2)
        for target, ctx in ctx_list():
            exe1 = relay.create_executor("graph", ctx=ctx, target=target)
            exe2 = relay.create_executor("debug", ctx=ctx, target=target)
            res1 = exe1.evaluate(new_log)(x_nd)
            tvm.testing.assert_allclose(res1.asnumpy(), ref_res, rtol=1e-5)
            res2 = exe2.evaluate(new_log)(x_nd)
            tvm.testing.assert_allclose(res2.asnumpy(), ref_res, rtol=1e-5)

    test_pass_registration()
    test_pass_registration_no_decorator()
    test_pass_run()


def test_module_class_pass():
    @tvm.transform.module_pass(opt_level=1)
    class TestPipeline:
        """Simple test function to replace one argument to another."""
        def __init__(self, new_mod, replace):
            self.new_mod = new_mod
            self.replace = replace

        def transform_module(self, mod, ctx):
            if self.replace:
                return self.new_mod
            return mod

    x = relay.var("x", shape=(10, 20))
    m1 = tvm.IRModule.from_expr(relay.Function([x], x))
    m2 = tvm.IRModule.from_expr(relay.Function([x], relay.log(x)))
    fpass = TestPipeline(m2, replace=True)
    assert fpass.info.name == "TestPipeline"
    mod3 = fpass(m1)
    assert mod3.same_as(m2)
    mod4 = TestPipeline(m2, replace=False)(m1)
    assert mod4.same_as(m1)


def test_pass_info():
    info = tvm.transform.PassInfo(opt_level=1, name="xyz")
    assert info.opt_level == 1
    assert info.name == "xyz"


def test_sequential_pass():
    shape = (10, )
    dtype = 'float32'
    tp = relay.TensorType(shape, dtype)
    x = relay.var("x", tp)
    y = relay.var("y", tp)
    v_sub = relay.GlobalVar("mySub")
    sub = relay.Function([x, y], relay.subtract(x, y))

    z = relay.var("z", tp)
    v_log = relay.GlobalVar("myLog")
    log = relay.Function([z], relay.log(z))

    mod = tvm.IRModule({v_sub: sub, v_log: log})

    def get_ref_log():
        ref_log = relay.Function([x], relay.log(relay.add(x, x)))
        return ref_log

    def get_ref_sub():
        ref_sub = relay.Function([x, y],
                                 relay.subtract(
                                     relay.add(x, x), relay.add(y, y)))
        return ref_sub

    def get_ref_abs():
        shape = (5, 10)
        tp = relay.TensorType(shape, "float32")
        a = relay.var("a", tp)
        ref_abs = relay.Function([a], relay.abs(relay.add(a, a)))
        return ref_abs

    # Register a module pass.
    opt_tester = OptTester(mod)
    pass_ctx = None

    @tvm.transform.module_pass(opt_level=1)
    def mod_transform(expr, ctx):
        return opt_tester.transform(expr, ctx)

    module_pass = mod_transform

    # Register a function pass.
    @_transform.function_pass(opt_level=1)
    def func_transform(expr, mod, ctx):
        return opt_tester.transform(expr, ctx)

    function_pass = func_transform

    def test_pass_registration():
        passes = [module_pass, function_pass]
        opt_level = 2
        pass_name = "sequential"
        sequential = tvm.transform.Sequential(passes=passes, opt_level=opt_level)
        pass_info = sequential.info
        assert pass_info.name == pass_name
        assert pass_info.opt_level == opt_level

    def test_no_pass():
        passes = []
        sequential = tvm.transform.Sequential(opt_level=1, passes=passes)
        ret_mod = sequential(mod)
        mod_func = ret_mod[v_sub]
        check_func(sub, mod_func)

    def test_only_module_pass():
        passes = [module_pass]
        sequential = tvm.transform.Sequential(opt_level=1, passes=passes)
        with relay.build_config(required_pass=["mod_transform"]):
            ret_mod = sequential(mod)
        # Check the subtract function.
        sub_var, new_sub = extract_var_func(ret_mod, v_sub.name_hint)
        check_func(new_sub, sub)

        # Check the abs function is added.
        abs_var, abs_func = get_var_func()
        abs_var, new_abs = extract_var_func(ret_mod, abs_var.name_hint)
        check_func(new_abs, abs_func)

    def test_only_function_pass():
        # Check the subtract function.
        passes = [function_pass]
        sequential = tvm.transform.Sequential(opt_level=1, passes=passes)
        with relay.build_config(required_pass=["func_transform"]):
            ret_mod = sequential(mod)
        _, new_sub = extract_var_func(ret_mod, v_sub.name_hint)
        check_func(new_sub, get_ref_sub())

        # Check the log function.
        log_var, new_log = extract_var_func(ret_mod, v_log.name_hint)
        check_func(new_log, get_ref_log())

    def test_multiple_passes():
        # Reset the current module since mod has been polluted by the previous
        # function pass.
        mod = tvm.IRModule({v_sub: sub, v_log: log})
        passes = [module_pass, function_pass]
        sequential = tvm.transform.Sequential(opt_level=1, passes=passes)
        required = ["mod_transform", "func_transform"]
        with relay.build_config(required_pass=required):
            ret_mod = sequential(mod)

        # Check the abs function is added.
        abs_var, abs_func = get_var_func()
        abs_var, new_abs = extract_var_func(ret_mod, abs_var.name_hint)
        check_func(new_abs, get_ref_abs())

        # Check the subtract function is modified correctly.
        _, new_sub = extract_var_func(ret_mod, v_sub.name_hint)
        check_func(new_sub, get_ref_sub())

        # Check the log function is modified correctly.
        _, new_log = extract_var_func(ret_mod, v_log.name_hint)
        check_func(new_log, get_ref_log())

        # Execute the updated subtract function.
        x_nd = get_rand(shape, dtype)
        y_nd = get_rand(shape, dtype)
        ref_res = np.subtract(x_nd.asnumpy() * 2, y_nd.asnumpy() * 2)
        for target, ctx in ctx_list():
            exe1 = relay.create_executor("graph", ctx=ctx, target=target)
            exe2 = relay.create_executor("debug", ctx=ctx, target=target)
            res1 = exe1.evaluate(new_sub)(x_nd, y_nd)
            tvm.testing.assert_allclose(res1.asnumpy(), ref_res, rtol=1e-5)
            res2 = exe2.evaluate(new_sub)(x_nd, y_nd)
            tvm.testing.assert_allclose(res2.asnumpy(), ref_res, rtol=1e-5)

        # Execute the updated abs function.
        x_nd = get_rand((5, 10), dtype)
        ref_res = np.abs(x_nd.asnumpy() * 2)
        for target, ctx in ctx_list():
            exe1 = relay.create_executor("graph", ctx=ctx, target=target)
            exe2 = relay.create_executor("debug", ctx=ctx, target=target)
            res1 = exe1.evaluate(new_abs)(x_nd)
            tvm.testing.assert_allclose(res1.asnumpy(), ref_res, rtol=1e-5)
            res2 = exe2.evaluate(new_abs)(x_nd)
            tvm.testing.assert_allclose(res2.asnumpy(), ref_res, rtol=1e-5)

    test_pass_registration()
    test_no_pass()
    test_only_module_pass()
    test_only_function_pass()
    test_multiple_passes()


def test_sequential_with_scoping():
    shape = (1, 2, 3)
    c_data = np.array(shape).astype("float32")
    tp = relay.TensorType(shape, "float32")
    def before():
        c = relay.const(c_data)
        x = relay.var("x", tp)
        y = relay.add(c, c)
        y = relay.multiply(y, relay.const(2, "float32"))
        y = relay.add(x, y)
        z = relay.add(y, c)
        z1 = relay.add(y, c)
        z2 = relay.add(z, z1)
        return relay.Function([x], z2)

    def expected():
        x = relay.var("x", tp)
        c_folded = (c_data + c_data) * 2
        y = relay.add(x, relay.const(c_folded))
        z = relay.add(y, relay.const(c_data))
        z1 = relay.add(z, z)
        return relay.Function([x], z1)

    seq = tvm.transform.Sequential([
        relay.transform.InferType(),
        relay.transform.FoldConstant(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.AlterOpLayout()
    ])

    mod = tvm.IRModule({"main": before()})
    with relay.build_config(opt_level=3):
        with tvm.target.create("llvm"):
            mod = seq(mod)

    zz = mod["main"]
    zexpected = run_infer_type(expected())
    assert tvm.ir.structural_equal(zz, zexpected)


def test_print_ir(capfd):
    shape = (1, 2, 3)
    tp = relay.TensorType(shape, "float32")
    x = relay.var("x", tp)
    y = relay.add(x, x)
    y = relay.multiply(y, relay.const(2, "float32"))
    func = relay.Function([x], y)

    seq = tvm.transform.Sequential([
        relay.transform.InferType(),
        relay.transform.FoldConstant(),
        tvm.transform.PrintIR(),
        relay.transform.DeadCodeElimination()
    ])

    mod = tvm.IRModule({"main": func})
    with relay.build_config(opt_level=3):
        mod = seq(mod)

    out = capfd.readouterr().err

    assert "PrintIR" in out
    assert "multiply" in out

__TRACE_COUNTER__ = 0

def _tracer(module, info, is_before):
    global __TRACE_COUNTER__
    if bool(is_before):
        __TRACE_COUNTER__ += 1

def test_print_debug_callback():
    global __TRACE_COUNTER__
    shape = (1, 2, 3)
    tp = relay.TensorType(shape, "float32")
    x = relay.var("x", tp)
    y = relay.add(x, x)
    y = relay.multiply(y, relay.const(2, "float32"))
    func = relay.Function([x], y)

    seq = tvm.transform.Sequential([
        relay.transform.InferType(),
        relay.transform.FoldConstant(),
        relay.transform.DeadCodeElimination()
    ])

    assert __TRACE_COUNTER__ == 0
    mod = tvm.IRModule({"main": func})

    with relay.build_config(opt_level=3, trace=_tracer):
        mod = seq(mod)

    assert __TRACE_COUNTER__ == 4


if __name__ == "__main__":
    pytest.main()
