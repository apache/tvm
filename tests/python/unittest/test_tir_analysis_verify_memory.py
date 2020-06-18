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
import pytest
from tvm import te

# The following DLDeviceType/TVMDeviceExtType values
# are originally defined in dlpack.h and c_runtime_api.h.
gpu_devices = ["cuda", "opencl", "metal", "vulkan"]
other_devices = ["llvm", "ext_dev"]


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

    f = tvm.tir.PrimFunc(arg_list, stmt).with_attr(
        "global_symbol", tvm.runtime.String("test"))
    mod = tvm.IRModule({"test": f})
    return mod


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

  mod = lower(s, [A, B])

  for dev_type in gpu_devices + other_devices:
      binded_mod = tvm.tir.transform.Apply(
          lambda f: f.with_attr("target", tvm.target.create(dev_type)))(mod)
      tvm.tir.analysis.verify_memory(binded_mod)



# Computations are not bound.
# So VerifyMemory pass fails when device type is GPU.
#
def test_verify_memory_not_bind():
  n = te.var("n")
  A = te.placeholder((n,), name='A')
  B = te.compute(A.shape, lambda i: A[i] + 1.0, name="B")

  # B is not bound to threads.
  s = te.create_schedule(B.op)

  mod = lower(s, [A, B])

  for dev_type in gpu_devices:
      binded_mod = tvm.tir.transform.Apply(
          lambda f: f.with_attr("target", tvm.target.create(dev_type)))(mod)
      with pytest.raises(ValueError):
          tvm.tir.analysis.verify_memory(binded_mod)

  for dev_type in other_devices:
      binded_mod = tvm.tir.transform.Apply(
          lambda f: f.with_attr("target", tvm.target.create(dev_type)))(mod)
      tvm.tir.analysis.verify_memory(binded_mod)


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

  mod = lower(s, [A, B, C, D])

  for dev_type in gpu_devices:
      binded_mod = tvm.tir.transform.Apply(
          lambda f: f.with_attr("target", tvm.target.create(dev_type)))(mod)
      with pytest.raises(ValueError):
          tvm.tir.analysis.verify_memory(binded_mod)

  for dev_type in other_devices:
      binded_mod = tvm.tir.transform.Apply(
          lambda f: f.with_attr("target", tvm.target.create(dev_type)))(mod)
      tvm.tir.analysis.verify_memory(binded_mod)



if __name__ == "__main__":
  test_verify_memory_all_bind()
  test_verify_memory_not_bind()
  test_verify_memory_partially_bind()
