import tvm

x = tvm.Var('x')
y = tvm.Var('y')
z = tvm.Var('z')

ys = tvm.arith.intset_range(2, 3)
zs = tvm.arith.intset_range(2, 3)


e0 = (-z)*x+y
e1 = tvm.ir_pass.DeduceBound(x, e0, {y: ys, z: zs})
print(e1)
