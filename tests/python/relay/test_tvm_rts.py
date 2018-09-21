import numpy as np

from tvm import relay
from tvm.relay.ir_pass import infer_type
from tvm.relay.eval import evaluate
from tvm.relay.graph_runtime_codegen import evaluate_rts
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.op import add
from tvm.relay.env import Environment

# @tq, @jr should we put this in testing ns?
def check_rts(env, expr, args, expected_result):
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
    eval_result = evaluate(env, expr, *args)
    rts_result = evaluate_rts(env, expr, *args)
    np.testing.assert_allclose(eval_result.asnumpy(), rts_result.asnumpy())

def test_add_op_scalar():
    """
    Program:
        fn (x, y) {
            return x + y;
        }
    """
    env = Environment()
    x = relay.var('x', shape=())
    y = relay.var('y', shape=())
    func = relay.Function([x, y], add(x, y))
    x_data = np.array(10.0, dtype='float32')
    y_data = np.array(1.0, dtype='float32')
    check_rts(env, func, [x_data, y_data], x_data + y_data)

def test_add_op_tensor():
    """
    Program:
        fn (x, y) {
            return x + y;
        }
    """
    env = Environment()
    x = relay.var('x', shape=(10, 5))
    y = relay.var('y', shape=(10, 5))
    func = relay.Function([x, y], add(x, y))
    x_data = np.random.rand(10, 5).astype('float32')
    y_data = np.random.rand(10, 5).astype('float32')
    check_rts(env, func, [x_data, y_data], x_data + y_data)

def test_add_op_broadcast():
    """
    Program:
        fn (x, y) {
            return x + y;
        }
    """
    env = Environment()
    x = relay.var('x', shape=(10, 5))
    y = relay.var('y', shape=(1, 5))
    func = relay.Function([x, y], add(x, y))
    x_data = np.random.rand(10, 5).astype('float32')
    y_data = np.random.rand(1, 5).astype('float32')
    check_rts(env, func, [x_data, y_data], x_data + y_data)

def test_mlp():
    net, params = relay.testing.mlp.get_workload(1, 10)
    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    test_add_op_scalar()
    test_add_op_tensor()
    test_add_op_broadcast()
    test_mlp()