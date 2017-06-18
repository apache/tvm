package ml.dmlc.tvm

import ml.dmlc.tvm.Base._
import ml.dmlc.tvm.types.{TVMType, TVMContext}

import scala.collection.mutable.ArrayBuffer

/**
 * Lightweight NDArray class of TVM runtime.
 */
class NDArray(private val handle: TVMArrayHandle, private val isView: Boolean = false) {
  override protected def finalize(): Unit = {
    if (!isView) {
      checkCall(_LIB.tvmArrayFree(handle))
    }
  }

  /**
   * Perform an synchronize copy from the array.
   * @param sourceArray The data source we should like to copy from.
   */
  /*
  private def syncCopyfrom(sourceArray: Array[Float]): Unit = {
    require(shape.
    check_call(_LIB.TVMArrayCopyFromTo(
      ctypes.byref(source_tvm_arr), self.handle, None))
  }
  */

  /**
   * Get shape of current NDArray.
   * @return an array representing shape of current ndarray
   */
  def shape: Shape = {
    val data = ArrayBuffer[Long]()
    checkCall(_LIB.tvmArrayGetShape(handle, data))
    Shape(data)
  }
}

object NDArray {
  /**
   * Create an empty array given shape and device
   * @param shape The shape of the array
   * @param dtype The data type of the array
   * @param ctx The context of the array
   * @return The array tvm supported
   */
  def empty(shape: Shape,
            dtype: String = "float32",
            ctx: TVMContext = new TVMContext(1, 0)): NDArray = {
    val refHandle = new RefTVMArrayHandle()
    val t = new TVMType(dtype)
    println(s"Scala t code: ${t.typeCode}, bits: ${t.bits}, lanes: ${t.lanes}")
    checkCall(_LIB.tvmArrayAlloc(shape.toArray, t, ctx, refHandle))
    new NDArray(refHandle.value, false)
  }
}
