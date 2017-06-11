package ml.dmlc.tvm

import ml.dmlc.tvm.Base._
import ml.dmlc.tvm.types.TVMValue

import scala.collection.mutable.ArrayBuffer

private[tvm] class LibInfo {
  @native def nativeLibInit(): Int
  @native def tvmGetLastError(): String

  // Function
  @native def tvmFuncListGlobalNames(funcNames: ArrayBuffer[String]): Int
  @native def tvmFuncFree(handle: FunctionHandle): Int
  @native def tvmFuncCall(handle: FunctionHandle, args: Array[TVMValue], retVal: TVMValue): Int

  // NDArray
  @native def tvmArrayFree(handle: TVMArrayHandle): Int
}
