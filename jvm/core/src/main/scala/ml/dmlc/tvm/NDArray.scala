package ml.dmlc.tvm

import java.nio.{ByteOrder, ByteBuffer}

import ml.dmlc.tvm.Base._
import ml.dmlc.tvm.types.{TVMType, TVMContext}

import scala.collection.mutable.ArrayBuffer

/**
 * Lightweight NDArray class of TVM runtime.
 */
class NDArray(private[tvm] val handle: TVMArrayHandle,
              private val isView: Boolean = false,
              private val dtype: TVMType = new TVMType("float32")) {
  override protected def finalize(): Unit = {
    if (!isView) {
      checkCall(_LIB.tvmArrayFree(handle))
    }
  }

  def set(source: Array[Float]): Unit = {
    syncCopyFrom(source)
  }

  /**
   * Perform an synchronize copy from the array.
   * @param sourceArray The data source we should like to copy from.
   */
  private def syncCopyFrom(sourceArray: Array[Float]): Unit = {
    require(shape.product == sourceArray.length)
    val tmpArr = NDArray.empty(shape)
    checkCall(_LIB.tvmArrayCopyFromJArray(sourceArray, tmpArr.handle, handle))
    checkCall(_LIB.tvmArrayFree(tmpArr.handle))
  }

  /**
   * Get shape of current NDArray.
   * @return an array representing shape of current ndarray
   */
  def shape: Shape = {
    val data = ArrayBuffer[Long]()
    checkCall(_LIB.tvmArrayGetShape(handle, data))
    Shape(data)
  }

  // Get size of current NDArray.
  def size: Long = shape.product

  def internal: NDArrayInternal = {
    val arrLength = dtype.numOfBytes * size.toInt
    val arr = Array.ofDim[Byte](arrLength)
    checkCall(_LIB.tvmArrayCopyToJArray(handle, arr))
    new NDArrayInternal(arr, dtype)
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
    checkCall(_LIB.tvmArrayAlloc(shape.toArray, t, ctx, refHandle))
    new NDArray(refHandle.value, false, t)
  }
}

private[tvm] class NDArrayInternal (private val internal: Array[Byte], private val dtype: TVMType) {
  private val unitSize = dtype.numOfBytes
  require(internal.length > 0 && internal.length % unitSize == 0,
    s"$dtype size $unitSize cannot divide byte array size ${internal.length}")
  private val units: Array[Array[Byte]] = (
    for (i <- 0 until internal.length / unitSize)
      yield internal.slice(i * unitSize, (i + 1) * unitSize)
    ).toArray

  def getRaw: Array[Byte] = internal
  def toFloatArray: Array[Float] = {
    // TODO
    units.map(wrapBytes(_).getFloat)
  }

  private def wrapBytes(bytes: Array[Byte]): ByteBuffer = {
    val bb = ByteBuffer.wrap(bytes)
    bb.order(ByteOrder.LITTLE_ENDIAN)
    bb
  }
}
