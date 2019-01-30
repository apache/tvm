"""Unit tests for relay optimizer."""
import numpy as np

import tvm
from tvm import relay
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
            return node
        if isinstance(node, relay.Expr):
            return node


def get_rand(shape, dtype='float32'):
    return tvm.nd.array(np.random.rand(*shape).astype(dtype))


def get_simple_func():
    shape = (5, 10)
    dtype = 'float32'
    tp = relay.TensorType(shape, dtype)
    x = relay.var("x", tp)
    y = relay.var("y", tp)
    func = relay.Function([x, y], x + y)

    func = infer_type(func)
    assert func.checked_type == tp
    ex = create_executor()
    x_nd = get_rand(*shape, dtype)
    y_nd = get_rand(*shape, dtype)
    res = ex.evaluate(func)(x_nd, y_nd)
    np.testing.assert_allclose(res.asnumpy(), x_nd.asnumpy() + y_nd.asnumpy())


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


def module_pass_function(state):
    """This function uses currying. It is designed to be flexible for
    passing Relay nodes at various granularity.
    """
    opt_tester = OptTester(state)

    def _transform(m):
        return opt_tester.transform(m)

    return _transform


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

    pass_name = "test"
    opt_level = 0
    pass_kind = optimizer.PassKind.ModuleKind
    pass_func = module_pass_function

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
    def test_pass_registration():
        pass

    def test_pass_execution():
        pass

    pass


def test_pass_optimize():
    pass


if __name__ == "__main__":
    test_pass_state()
    test_module_pass()
