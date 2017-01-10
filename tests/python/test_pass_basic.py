import tvm

def test_verify_ssa():
    x = tvm.Var('x')
    y = tvm.Var()
    z = tvm.make.Evaluate(x + y)
    assert(tvm.ir_pass.VerifySSA(z))


def test_convert_ssa():
    x = tvm.Var('x')
    y = tvm.Var()
    let = tvm.make.Let(x, 1, x + 1)
    z = tvm.make.Evaluate(let + let)
    assert(not tvm.ir_pass.VerifySSA(z))
    z_ssa = tvm.ir_pass.ConvertSSA(z)
    assert(tvm.ir_pass.VerifySSA(z_ssa))
