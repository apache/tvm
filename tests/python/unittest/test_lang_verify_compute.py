import tvm

def test_verify_compute():
  n = tvm.var("n")
  m = tvm.var("m")
  A = tvm.placeholder((n, m), name='A')
  k = tvm.reduce_axis((0, m), "k")
  k_ = tvm.reduce_axis((0, m-1), "k_")
  f1 = lambda i: tvm.sum(A[i, k], axis=k)
  f2 = lambda i: A[i,0] + 1
  f3 = lambda i: tvm.sum(A[i, k], axis=k) + 1
  f4 = lambda i: A[i,0] * (tvm.sum(A[i, k], axis=k) + 1)
  f5 = lambda i: (tvm.sum(A[i, k], axis=k), A[i,0] + 1)
  f6 = lambda i: (tvm.sum(A[i, k], axis=k), tvm.sum(A[i, k_], axis=k_))

  #
  # Valid compute
  try:
    B = tvm.compute((n,), f1, name="B")
  except tvm._ffi.base.TVMError as ex:
    assert False

  #
  # Valid compute
  try:
    B = tvm.compute((n,), f2, name="B")
  except tvm._ffi.base.TVMError as ex:
    assert False

  #
  # Invalid compute with non top level reduction
  try:
    B = tvm.compute((n,), f3, name="B")
    assert False
  except tvm._ffi.base.TVMError as ex:
    pass

  #
  # Invalid compute with non top level reduction
  try:
    B = tvm.compute((n,), f4, name="B")
    assert False
  except tvm._ffi.base.TVMError as ex:
    pass

  #
  # Invalid compute with reduction and non-reduction batch ops
  try:
    B0, B1 = tvm.compute((n,), f5, name="B")
    assert False
  except tvm._ffi.base.TVMError as ex:
    pass

  #
  # Invalid compute with unequal batch reduction ops
  try:
    B0, B1 = tvm.compute((n,), f6, name="B")
    assert False
  except tvm._ffi.base.TVMError as ex:
    pass


if __name__ == "__main__":
  test_verify_compute()