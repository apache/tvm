# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm
from tvm import te

# The following DLDeviceType/TVMDeviceExtType values
# are originally defined in dlpack.h and c_runtime_api.h.
gpu_devices = [2, 4, 7, 8, 10, 11]
other_devices = [1, 3, 9, 12]


def lower(sch, args):
    binds = {}
    arg_list = []
    for x in args:
        if isinstance(x, te.tensor.Tensor):
            buf = tvm.tir.decl_buffer(x.shape, dtype=x.dtype, name=x.name)
            assert x not in binds
            binds[x] = buf
            arg_list.append(buf)
        else:
            raise ValueError("args must be Tensor, Buffer or Var")
    sch = sch.normalize()
    bounds = tvm.te.schedule.InferBound(sch)
    stmt = tvm.te.schedule.ScheduleOps(sch, bounds)
    stmt = tvm.tir.ir_pass.LoopPartition(stmt, False)
    stmt = tvm.tir.ir_pass.StorageFlatten(stmt, binds, 64)
    func = tvm.tir.ir_pass.MakeAPI(stmt, "myadd", arg_list, 0, True)
    return func


# All computations are bound.
# So VerifyMemory pass is expected to succeed.
#
def test_verify_memory_all_bind():
  n = te.var("n")
  A = te.placeholder((n,), name='A')
  B = te.compute(A.shape, lambda i: A[i] + 1.0, name="B")

  # B is bound to threads.
  s = te.create_schedule(B.op)
  bx, tx = s[B].split(B.op.axis[0], factor=64)
  s[B].bind(bx, te.thread_axis("blockIdx.x"))
  s[B].bind(tx, te.thread_axis("threadIdx.x"))

  func = lower(s, [A, B])

  for dev_type in gpu_devices + other_devices:
    assert tvm.tir.ir_pass.VerifyMemory(func, dev_type)


# Computations are not bound.
# So VerifyMemory pass fails when device type is GPU.
#
def test_verify_memory_not_bind():
  n = te.var("n")
  A = te.placeholder((n,), name='A')
  B = te.compute(A.shape, lambda i: A[i] + 1.0, name="B")

  # B is not bound to threads.
  s = te.create_schedule(B.op)

  func = lower(s, [A, B])

  for dev_type in gpu_devices:
    assert not tvm.tir.ir_pass.VerifyMemory(func, dev_type)
  for dev_type in other_devices:
    assert tvm.tir.ir_pass.VerifyMemory(func, dev_type)


# Computations are partially bound.
# So VerifyMemory pass fails when device type is GPU.
#
def test_verify_memory_partially_bind():
  n = te.var("n")
  A = te.placeholder((n,), name='A')
  B = te.compute(A.shape, lambda i: A[i] + 1.0, name="B")
  C = te.compute(B.shape, lambda i: B[i] + 2.0, name="C")
  D = te.compute(C.shape, lambda i: C[i] + 2.0, name="D")

  # C is bound to threads, but B and D are not.
  s = te.create_schedule([B.op, C.op, D.op])
  bx, tx = s[C].split(C.op.axis[0], factor=64)
  s[C].bind(bx, te.thread_axis("blockIdx.x"))
  s[C].bind(tx, te.thread_axis("threadIdx.x"))

  func = lower(s, [A, B, C, D])

  for dev_type in gpu_devices:
    assert not tvm.tir.ir_pass.VerifyMemory(func, dev_type)
  for dev_type in other_devices:
    assert tvm.tir.ir_pass.VerifyMemory(func, dev_type)


if __name__ == "__main__":
  test_verify_memory_all_bind()
  test_verify_memory_not_bind()
  test_verify_memory_partially_bind()

