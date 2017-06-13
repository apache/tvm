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
  @native def tvmFuncGetGlobal(name: String, handle: RefFunctionHandle): Int
  @native def tvmFuncCall(handle: FunctionHandle, args: Array[TVMValue], retVal: RefTVMValue): Int

  // Module
  @native def tvmModFree(handle: ModuleHandle): Int
  @native def tvmModGetFunction(handle: ModuleHandle, name: String,
                                queryImports: Int, retHandle: RefFunctionHandle): Int

  // NDArray
  @native def tvmArrayFree(handle: TVMArrayHandle): Int
}
