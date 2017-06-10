package ml.dmlc.tvm

import ml.dmlc.tvm.Base.{checkCall, _LIB}

import scala.collection.mutable.ArrayBuffer

object Function {
  /**
   * Get list of global functions registered.
   * @return List of global functions names.
   */
  private[tvm] def listGlobalFuncNames(): IndexedSeq[String] = {
    val names = ArrayBuffer.empty[String]
    checkCall(_LIB.tvmFuncListGlobalNames(names))
    names.toIndexedSeq
  }
}
