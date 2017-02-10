import tvm

x = tvm.Var('x')
y = tvm.Var('y')
z = tvm.Var('z')
a = tvm.Var('a')
b = tvm.Var('b')

e0 = (x*y+z<a+b)
print(type(e0))
e1 = tvm.ir_pass.Deduce(x, e0)
print(e1)
