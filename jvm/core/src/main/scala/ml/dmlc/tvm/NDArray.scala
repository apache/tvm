package ml.dmlc.tvm

import ml.dmlc.tvm.Base._

/**
 * Lightweight NDArray class of TVM runtime.
 */
class NDArray(private val handle: TVMArrayHandle, private val isView: Boolean = false) {
  override protected def finalize(): Unit = {
    if (!isView) {
      checkCall(_LIB.tvmArrayFree(handle))
    }
  }
}
