import tvm
from tvm import relay
from tvm.relay.ir_pass import dead_code_elimination, alpha_equal
from tvm.relay.ir_builder import convert, IRBuilder
from tvm.relay.op import log, add, equal, subtract

class env:
    def __init__(self):
        self.a = relay.Var("a")
        self.b = relay.Var("b")
        self.c = relay.Var("c")
        self.d = relay.Var("d")
        self.e = relay.Var("e")
        self.x = relay.Var("x")
        self.y = relay.Var("y")
        self.z = relay.Var("z")
        self.shape = tvm.convert([1, 2, 3])
        self.tt = relay.TensorType(self.shape, "float32")
        self.int32 = relay.TensorType([], "int32")
        self.float32 = relay.TensorType([], "float32")
        self.one = convert(1.0)
        self.two = convert(2.0)
        self.three = convert(3.0)

e = env()

def test_let():
    orig = relay.Let(e.x, e.y, e.z, e.tt)
    assert alpha_equal(dead_code_elimination(orig), e.z)

def test_used_let():
    orig = relay.Let(e.a, e.b, relay.Let(e.c, e.d, e.c, e.tt), e.tt)
    assert alpha_equal(dead_code_elimination(orig), relay.Let(e.c, e.d, e.c, e.tt))

def test_chain_unused_let():
    orig = relay.Let(e.a, e.b, relay.Let(e.c, e.d, e.e, e.tt), e.tt)
    assert alpha_equal(dead_code_elimination(orig), e.e)

# make sure we dont infinite loop
def test_recursion():
    """
    Program:
       let f(n: i32, data: f32) -> f32 = {
          if (n == 0) {
              return data;
          } else {
              return f(n - 1, log(data));
          }
       }
       f(2, 10000);
    """
    f = relay.Var("f")
    n = relay.Var("n")
    np = relay.Param(n, e.int32)
    data = relay.Var("data")
    datap = relay.Param(data, e.float32)
    funcbody = relay.If(equal(n, convert(0)), data, f(subtract(n, convert(1.0)), log(data)))
    value = relay.Function([np, datap], e.float32, funcbody, [])
    orig = relay.Let(f, funcbody, f(convert(2.0), convert(10000.0)), e.float32)
    assert alpha_equal(dead_code_elimination(orig), orig)
    assert alpha_equal(dead_code_elimination(relay.Let(f, funcbody, e.three, e.float32)), e.three)

def test_op_let():
    assert alpha_equal(dead_code_elimination(add(relay.Let(e.a, e.one, e.three, e.float32), e.two)), add(e.three, e.two))

def test_if():
    orig = relay.If(convert(True), e.a, e.b)
    assert alpha_equal(dead_code_elimination(orig), e.a)


if __name__ == "__main__":
    test_let()
    test_used_let()
    test_chain_unused_let()
    test_recursion()
    test_op_let()
    test_if()
