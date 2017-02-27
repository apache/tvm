import tvm
n = tvm.Var('n')
A = tvm.placeholder((n, ), name='A')
B = tvm.placeholder((n, ), name='B')

T = tvm.compute((n, ), lambda i: A[i]+B[i])
s = tvm.Schedule(T.op)
xo, xi = s[T].split(T.op.axis[0], factor=4)

bounds = tvm.schedule.InferBound(s)
stmt = tvm.schedule.ScheduleOps(s, bounds)
stmt = tvm.ir_pass.LoopPartition(stmt)
print(stmt)

# for (i.outer, 0, n) {
#   for (i.inner, 0, 4) {
#     if (i.inner + (i.outer*4) < n) {
#       compute(i.inner + (i.outer*4)) = (A(i.inner + i.outer*4) + B(i.inner + i.outer*4))
#     }
#   }
# }
