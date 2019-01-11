import numpy as np

import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay.ir_pass import infer_type
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.op import add
from tvm.relay.module import Module

# @tq, @jr should we put this in testing ns?
def check_rts(expr, args, expected_result, mod=None):
    """
    Check that evaluating `expr` applied to the arguments produces
    `result` on both the evaluator and TVM runtime.

    Parameters
    ----------
    expr:
        The expression to evaluate

    args: list of Expr
        The arguments to supply the expr.

    expected_result:
        The expected result of running the expression.
    """
    intrp = relay.create_executor('debug', mod=mod)
    graph = relay.create_executor('graph', mod=mod)
    eval_result = intrp.evaluate(expr)(*args)
    rts_result = graph.evaluate(expr)(*args)
    tvm.testing.assert_allclose(eval_result.asnumpy(), rts_result.asnumpy())
    tvm.testing.assert_allclose(eval_result.asnumpy(), expected_result)

def test_add_op_scalar():
    """
    Program:
        fn (x, y) {
            return x + y;
        }
    """
    x = relay.var('x', shape=())
    y = relay.var('y', shape=())
    func = relay.Function([x, y], add(x, y))
    x_data = np.array(10.0, dtype='float32')
    y_data = np.array(1.0, dtype='float32')
    check_rts(func, [x_data, y_data], x_data + y_data)

def test_add_op_tensor():
    """
    Program:
        fn (x, y) {
            return x + y;
        }
    """
    x = relay.var('x', shape=(10, 5))
    y = relay.var('y', shape=(10, 5))
    func = relay.Function([x, y], add(x, y))
    x_data = np.random.rand(10, 5).astype('float32')
    y_data = np.random.rand(10, 5).astype('float32')
    check_rts(func, [x_data, y_data], x_data + y_data)

def test_add_op_broadcast():
    """
    Program:
        fn (x, y) {
            return x + y;
        }
    """
    x = relay.var('x', shape=(10, 5))
    y = relay.var('y', shape=(1, 5))
    func = relay.Function([x, y], add(x, y))
    x_data = np.random.rand(10, 5).astype('float32')
    y_data = np.random.rand(1, 5).astype('float32')
    check_rts(func, [x_data, y_data], x_data + y_data)


def test_with_params():
    x = relay.var('x', shape=(10, 5))
    y = relay.var('y', shape=(1, 5))
    z = relay.add(x, y)
    z = relay.exp(z)
    func = relay.Function([x, y], z)
    x_data = np.random.rand(10, 5).astype('float32')
    y_data = np.random.rand(1, 5).astype('float32')
    params = {"y": y_data}
    graph, lib, params = relay.build(func, "llvm", params=params)
    mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    mod.set_input(**params)
    mod.set_input(x=x_data)
    mod.run()
    res = mod.get_output(0).asnumpy()
    ref_res = np.exp(y_data + x_data)
    tvm.testing.assert_allclose(res, ref_res)


def test_plan_memory():
    # it is sufficient to cycle through two memories.

    x = relay.var("x", shape=(10,))
    y = relay.var("x", shape=(1,))
    y2 = relay.exp(y)
    z = relay.add(x, y2)
    z = relay.exp(z)
    z = relay.exp(z)
    z = relay.exp(z)
    z = relay.exp(z)
    z = relay.exp(z)
    func = relay.Function([x, y], z)
    func = relay.ir_pass.infer_type(func)
    func = relay.ir_pass.fuse_ops(func, opt_level=0)
    func = relay.ir_pass.infer_type(func)
    smap = relay.backend._backend.GraphPlanMemory(func)
    storage_ids = set()
    device_types = set()
    for k, v in smap.items():
        assert len(v) == 2
        for x in v[0]:
            storage_ids.add(x.value)
        for x in v[1]:
            device_types.add(x.value)

    # Current rule requires vars have unique storage id
    # because we don't do inplace, we will need another
    # two alternating temporary space.
    assert len(storage_ids) == 4
    assert len(device_types) == 1


if __name__ == "__main__":
    test_plan_memory()
    test_with_params()
    test_add_op_scalar()
    test_add_op_tensor()
    test_add_op_broadcast()
