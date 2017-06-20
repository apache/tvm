package ml.dmlc.tvm

import ml.dmlc.tvm.Base._
import ml.dmlc.tvm.types.{TVMContext, TVMType, TVMValue}

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
  @native def tvmModImport(mod: ModuleHandle, dep: ModuleHandle): Int

  // NDArray
  @native def tvmArrayFree(handle: TVMArrayHandle): Int
  @native def tvmArrayAlloc(shape: Array[Long],
                            dtype: TVMType,
                            ctx: TVMContext,
                            refHandle: RefTVMArrayHandle): Int
  @native def tvmArrayGetShape(handle: TVMArrayHandle, shape: ArrayBuffer[Long]): Int
  @native def tvmArrayCopyFromJArray(fromRaw: Array[Float],
                                     from: TVMArrayHandle,
                                     to: TVMArrayHandle): Int
  @native def tvmArrayCopyToJArray(from: TVMArrayHandle, to: Array[Byte]): Int
}
