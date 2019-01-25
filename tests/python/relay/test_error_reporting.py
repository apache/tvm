import tvm
from tvm import relay

def check_type_err(expr, msg):
    try:
        expr = relay.ir_pass.infer_type(expr)
        assert False
    except tvm.TVMError as err:
        assert msg in str(err)

def test_too_many_args():
    x = relay.var('x', shape=(10, 10))
    f = relay.Function([x], x)
    y = relay.var('y', shape=(10, 10))
    check_type_err(
        f(x, y),
        "the function is provided too many arguments expected 1, found 2;")

def test_too_few_args():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('y', shape=(10, 10))
    f = relay.Function([x, y], x)
    check_type_err(f(x), "the function is provided too few arguments expected 2, found 1;")

def test_rel_fail():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('y', shape=(11, 10))
    f = relay.Function([x, y], x + y)
    check_type_err(f(x, y), "Incompatible broadcast type TensorType([10, 10], float32) and TensorType([11, 10], float32);")

if __name__ == "__main__":
    test_too_many_args()
    test_too_few_args()
    test_rel_fail()
