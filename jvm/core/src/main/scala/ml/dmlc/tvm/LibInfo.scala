package ml.dmlc.tvm

import ml.dmlc.tvm.Base._

import scala.collection.mutable.ArrayBuffer

private[tvm] class LibInfo {
  @native def nativeLibInit(): Int
  @native def tvmGetLastError(): String

  // Function
  @native def tvmFuncListGlobalNames(funcNames: ArrayBuffer[String]): Int
  @native def tvmFuncFree(handle: FunctionHandle): Int

  // NDArray
  @native def tvmArrayFree(handle: TVMArrayHandle): Int
}
