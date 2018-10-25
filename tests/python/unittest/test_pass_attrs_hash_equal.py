import tvm

def test_attrs_equal():
    x = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3, 4))
    y = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3, 4))
    z = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3,4,1))
    assert tvm.ir_pass.AttrsEqual(x, y)
    assert not tvm.ir_pass.AttrsEqual(x, z)

    dattr = tvm.make.node("DictAttrs", x=1, y=10, name="xyz", padding=(0,0))
    assert not tvm.ir_pass.AttrsEqual(dattr, x)
    dattr2 = tvm.make.node("DictAttrs", x=1, y=10, name="xyz", padding=(0,0))
    assert tvm.ir_pass.AttrsEqual(dattr, dattr2)

    assert tvm.ir_pass.AttrsEqual({"x": x}, {"x": y})
    # array related checks
    assert tvm.ir_pass.AttrsEqual({"x": [x, x]}, {"x": [y, x]})
    assert not tvm.ir_pass.AttrsEqual({"x": [x, 1]}, {"x": [y, 2]})

    n = tvm.var("n")
    assert tvm.ir_pass.AttrsEqual({"x": n+1}, {"x": n+1})





def test_attrs_hash():
    fhash = tvm.ir_pass.AttrsHash
    x = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3, 4))
    y = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3, 4))
    assert fhash({"x": x}) == fhash({"x": y})
    assert fhash({"x": x}) != fhash({"x": [y, 1]})
    assert fhash({"x": [x, 1]}) == fhash({"x": [y, 1]})
    assert fhash({"x": [x, 2]}) == fhash({"x": [y, 2]})


if __name__ == "__main__":
    test_attrs_equal()
    test_attrs_hash()
