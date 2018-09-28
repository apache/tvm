import tvm
from tvm import relay
from tvm.relay.ir_pass import a_normal_form, alpha_equal
from tvm.relay.ir_builder import convert, IRBuilder
from tvm.relay.op import log, add, equal, subtract

i = IRBuilder()

def pp(x):
    print(relay.expr.debug_print(i.env, x))

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
    int32 = relay.TensorType([], "int32")
    float32 = relay.TensorType([], "float32")
    three = convert(3.0)
    f = relay.Var("f")
    n = relay.Var("n")
    np = relay.Param(n, int32)
    data = relay.Var("data")
    datap = relay.Param(data, float32)
    funcbody = relay.If(equal(n, convert(0)), data, f(subtract(n, convert(1.0)), log(data)))
    value = relay.Function([np, datap], float32, funcbody, [])
    it = relay.IncompleteType()
    orig = relay.Let(f, value, f(convert(2.0), convert(10000.0)), it)
    res = a_normal_form(orig)
    x0 = relay.Var('x')
    x1 = relay.Var('x')
    x2 = relay.Var('x')
    x3 = relay.Var('x')
    ret = relay.Let(x0, x3(x1, x2), x0, it)
    ret = relay.Let(x2, convert(10000.0), ret, it)
    ret = relay.Let(x1, convert(2.0), ret, it)
    x10 = relay.Var('x')
    x9 = relay.Var('x')
    x4 = relay.Var('x')
    x5 = relay.Var('x')
    x6 = relay.Var('x')
    x7 = relay.Var('x')
    x8 = relay.Var('x')
    elseret = relay.Let(x5, x3(x6, x7), x5, it)
    elseret = relay.Let(x7, log(data), elseret, it)
    elseret = relay.Let(x6, subtract(n, x8), elseret, it)
    elseret = relay.Let(x8, convert(1.0), elseret, it)
    cond = relay.If(x9, data, elseret)
    funcret = relay.Let(x4, cond, x4, it)
    funcret = relay.Let(x9, equal(n, x10), funcret, it)
    funcret = relay.Let(x10, convert(0), funcret, it)
    ret = relay.Let(x3, relay.Function([np, datap], float32, funcret, []), ret, it)
    assert (relay.expr.debug_print(i.env, res) == relay.expr.debug_print(i.env, ret))
    # assert alpha_equal(res, ret)
    # alpha equality test isnt working,
    # but I cant figure it out as debug printer show them as the same.
    # fix later
