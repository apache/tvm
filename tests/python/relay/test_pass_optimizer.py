"""Unit tests for relay optimizer."""
import numpy as np

import tvm
from tvm import relay
from tvm.relay import ExprFunctor
from tvm.relay import Function, Call, Let, Var, GlobalVar, If, Tuple, TupleGetItem, Constant
from tvm.relay.ir_pass import infer_type, graph_equal
from tvm.relay import create_executor, optimizer
from tvm.relay.testing import ctx_list


def get_var_func():
    shape = (5, 10)
    tp = relay.TensorType(shape, "float32")
    x = relay.var("x", tp)
    gv = relay.GlobalVar("myAbs")
    func = relay.Function([x], relay.abs(x))
    return gv, func


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
    """A helper class for testing the optimizer."""

    def __init__(self, state):
        if not isinstance(state, relay.PassState):
            raise TypeError("state is expected to be the type of "
                            "relay.PassState")
        self.state = state

    def analysis(self):
        """Perform analysis for the current state."""
        pass

    @staticmethod
    def transform(node):
        """Perform optimization on node."""
        if isinstance(node, relay.Module):
            # Add a function to the module and return an updated module.
            gv, func = get_var_func()
            mod = relay.Module({gv: func})
            mod.update(node)
            return mod
        if isinstance(node, relay.Function):
            return update_func(node)
        if isinstance(node, relay.Expr):
            return node


def get_rand(shape, dtype='float32'):
    return tvm.nd.array(np.random.rand(*shape).astype(dtype))


def pass_function(state):
    """This function uses currying. It is designed to be flexible for
    passing Relay nodes at various granularity.
    """
    opt_tester = OptTester(state)

    def _transform(m):
        return opt_tester.transform(m)

    return _transform


def check_func(func, ref_func):
    func = infer_type(func)
    ref_func = infer_type(ref_func)
    assert graph_equal(func, ref_func)


def test_pass_state():
    shape = (5, 10)
    dtype = 'float32'
    tp = relay.TensorType(shape, dtype)
    x = relay.var("x", tp)
    y = relay.var("y", tp)
    gv = relay.GlobalVar("myAdd")
    func = relay.Function([x, y], x + y)

    mod = relay.Module({gv: func})
    state = optimizer.PassState(mod)
    state_func = state.mod[gv]
    check_func(func, state_func)


def test_module_pass():
    shape = (5, 10)
    dtype = 'float32'
    tp = relay.TensorType(shape, dtype)
    x = relay.var("x", tp)
    y = relay.var("y", tp)
    v_add = relay.GlobalVar("myAdd")
    func = relay.Function([x, y], x + y)
    mod = relay.Module({v_add: func})
    state = optimizer.PassState(mod)

    pass_name = "module_pass_test"
    opt_level = 0
    pass_kind = optimizer.PassKind.ModuleKind
    pass_func = pass_function

    def test_pass_registration():
        mod_pass = optimizer.build_pass(pass_name, opt_level, pass_kind,
                                        pass_func)
        assert isinstance(mod_pass, optimizer.ModulePass)
        assert mod_pass.name == pass_name
        assert mod_pass.opt_level == opt_level
        assert mod_pass.pass_kind == pass_kind

    def test_pass_run():
        module_pass = optimizer.ModulePass(pass_name, opt_level, pass_func)
        updated_pass = module_pass.run(state)
        assert isinstance(updated_pass, optimizer.PassState)

        # Check the abs function in the updated module.
        v_abs, myabs = get_var_func()
        new_v_add = updated_pass.mod.get_global_var(v_abs.name_hint)
        new_abs = updated_pass.mod[new_v_add]
        check_func(new_abs, myabs)

        # Check the add function in the updated module.
        v_abs, myabs = get_var_func()
        new_v_add = updated_pass.mod.get_global_var(v_add.name_hint)
        new_add = updated_pass.mod[new_v_add]
        check_func(new_add, func)

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
    test_pass_run()


def test_function_pass():
    shape = (10, )
    dtype = 'float32'
    tp = relay.TensorType(shape, dtype)
    x = relay.var("x", tp)
    v_log = relay.GlobalVar("myLog")
    log = relay.Function([x], relay.log(x))
    mod = relay.Module({v_log: log})
    state = optimizer.PassState(mod)

    pass_name = "function_pass_test"
    opt_level = 1
    pass_kind = optimizer.PassKind.FunctionKind
    pass_func = pass_function

    def get_ref_log():
        ref_log = relay.Function([x], relay.log(relay.add(x, x)))
        return ref_log

    def test_pass_registration():
        function_pass = optimizer.build_pass(pass_name, opt_level, pass_kind,
                                             pass_func)
        assert isinstance(function_pass, optimizer.FunctionPass)
        assert function_pass.name == pass_name
        assert function_pass.opt_level == opt_level
        assert function_pass.pass_kind == pass_kind

    def test_pass_run():
        function_pass = optimizer.FunctionPass(pass_name, opt_level, pass_func)
        updated_pass = function_pass.run(state)
        assert isinstance(updated_pass, optimizer.PassState)

        # Check the log function in the updated module.
        new_v_log = updated_pass.mod.get_global_var(v_log.name_hint)
        new_log = updated_pass.mod[new_v_log]
        check_func(new_log, get_ref_log())

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
    test_pass_run()


def test_pass_optimize():
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

    mod = relay.Module({v_sub: sub, v_log: log})
    state = optimizer.PassState(mod)

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
    module_pass_func = pass_function
    module_pass = optimizer.ModulePass("module_pass", 1, module_pass_func)

    # Register a function pass.
    function_pass_func = pass_function
    function_pass = optimizer.FunctionPass("function_pass", 2,
                                           function_pass_func)

    def test_no_pass():
        passes = []
        ret_state = optimizer.optimize(passes, state)
        state_func = ret_state.mod[v_sub]
        check_func(sub, state_func)

    def extrac_var_func(state, name):
        var = state.mod.get_global_var(name)
        func = state.mod[var]
        return var, func

    def test_only_module_pass():
        passes = [module_pass]
        ret_state = optimizer.optimize(passes, state)
        # Check the subtract function.
        sub_var, new_sub = extrac_var_func(ret_state, v_sub.name_hint)
        check_func(new_sub, sub)

        # Check the abs function is added.
        abs_var, abs_func = get_var_func()
        abs_var, new_abs = extrac_var_func(ret_state, abs_var.name_hint)
        check_func(new_abs, abs_func)

    def test_only_function_pass():
        # Check the subtract function.
        passes = [function_pass]
        ret_state = optimizer.optimize(passes, state)
        sub_var, new_sub = extrac_var_func(ret_state, v_sub.name_hint)
        check_func(new_sub, get_ref_sub())

        # Check the log function.
        log_var, new_log = extrac_var_func(ret_state, v_log.name_hint)
        check_func(new_log, get_ref_log())

    def test_multiple_passes():
        # Reset the pass state since mod has been polluted by the previous
        # function pass.
        mod = relay.Module({v_sub: sub, v_log: log})
        state = optimizer.PassState(mod)
        passes = [module_pass, function_pass]
        ret_state = optimizer.optimize(passes, state)

        # Check the abs function is added.
        abs_var, abs_func = get_var_func()
        abs_var, new_abs = extrac_var_func(ret_state, abs_var.name_hint)
        check_func(new_abs, get_ref_abs())

        # Check the subtract function is modified correctly.
        sub_var, new_sub = extrac_var_func(ret_state, v_sub.name_hint)
        check_func(new_sub, get_ref_sub())

        # Check the log function is modified correctly.
        log_var, new_log = extrac_var_func(ret_state, v_log.name_hint)
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


    test_no_pass()
    test_only_module_pass()
    test_only_function_pass()
    test_multiple_passes()


if __name__ == "__main__":
    test_pass_state()
    test_module_pass()
    test_function_pass()
    test_pass_optimize()
