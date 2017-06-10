package ml.dmlc.tvm

import scala.collection.mutable.ArrayBuffer

private[tvm] class LibInfo {
  @native def nativeLibInit(): Int
  @native def tvmGetLastError(): String
  @native def tvmFuncListGlobalNames(funcNames: ArrayBuffer[String]): Int
}
