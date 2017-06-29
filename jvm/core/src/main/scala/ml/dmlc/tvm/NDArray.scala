/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.dmlc.tvm

import java.nio.{ByteOrder, ByteBuffer}

import ml.dmlc.tvm.Base._
import ml.dmlc.tvm.types.{TVMType, TVMContext}

import scala.collection.mutable.ArrayBuffer

/**
 * Lightweight NDArray class of TVM runtime.
 */
// scalastyle:off finalize
class NDArray(private[tvm] val handle: TVMArrayHandle,
              private val isView: Boolean = false,
              private val dtype: TVMType = new TVMType("float32")) {
  override protected def finalize(): Unit = {
    if (!isView) {
      checkCall(_LIB.tvmArrayFree(handle))
    }
  }

  def set(source: Array[Double]): Unit = {
    syncCopyFrom(source)
  }

  /**
   * Perform an synchronize copy from the array.
   * @param sourceArray The data source we should like to copy from.
   */
  private def syncCopyFrom(sourceArray: Array[Double]): Unit = {
    require(shape.product == sourceArray.length)
    val tmpArr = NDArray.empty(shape, dtype = this.dtype)

    val nativeArr = Array.ofDim[Byte](sourceArray.length * dtype.numOfBytes)
    dtype.typeCode match {
      case TVMType.INT | TVMType.UINT =>
        dtype.bits match {
          case 8 => (0 until sourceArray.length).foreach(i => nativeArr(i) = sourceArray(i).toByte)
          case 16 => (0 until sourceArray.length).foreach(i =>
            NDArrayInternal.wrapBytes(nativeArr, i * dtype.numOfBytes, dtype.numOfBytes)
              .putShort(sourceArray(i).toShort))
          case 32 => (0 until sourceArray.length).foreach(i =>
            NDArrayInternal.wrapBytes(nativeArr, i * dtype.numOfBytes, dtype.numOfBytes)
              .putInt(sourceArray(i).toLong.toInt))
          case 64 => (0 until sourceArray.length).foreach(i =>
            NDArrayInternal.wrapBytes(nativeArr, i * dtype.numOfBytes, dtype.numOfBytes)
              .putLong(sourceArray(i).toLong))
          case _ => throw new IllegalArgumentException("Do not know how to handle type " + dtype)
        }
      case TVMType.FLOAT =>
        dtype.bits match {
          case 16 => throw new IllegalArgumentException(
            "Currently cannot convert native numerical types to float16")
          case 32 => (0 until sourceArray.length).foreach(i =>
            NDArrayInternal.wrapBytes(nativeArr, i * dtype.numOfBytes, dtype.numOfBytes)
              .putFloat(sourceArray(i).toFloat))
          case 64 => (0 until sourceArray.length).foreach(i =>
            NDArrayInternal.wrapBytes(nativeArr, i * dtype.numOfBytes, dtype.numOfBytes)
              .putDouble(sourceArray(i)))
          case _ => throw new IllegalArgumentException("Do not know how to handle type " + dtype)
        }
    }
    checkCall(_LIB.tvmArrayCopyFromJArray(nativeArr, tmpArr.handle, handle))
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

  /**
   * Return a copied flat java array of current array (row-major).
   * @return  A copy of array content.
   */
  def toArray: Array[Double] = {
    internal.toArray
  }

  def internal: NDArrayInternal = {
    val tmp = NDArray.empty(shape, dtype)
    checkCall(_LIB.tvmArrayCopyFromTo(handle, tmp.handle))

    val arrLength = dtype.numOfBytes * size.toInt
    val arr = Array.ofDim[Byte](arrLength)
    checkCall(_LIB.tvmArrayCopyToJArray(tmp.handle, arr))
    new NDArrayInternal(arr, dtype)
  }
}
// scalastyle:on finalize

object NDArray {
  /**
   * Create an empty array given shape and device
   * @param shape The shape of the array
   * @param dtype The data type of the array
   * @param ctx The context of the array
   * @return The array tvm supported
   */
  def empty(shape: Shape,
            dtype: TVMType = TVMType("float32"),
            ctx: TVMContext = new TVMContext(1, 0)): NDArray = {
    val refHandle = new RefTVMArrayHandle()
    checkCall(_LIB.tvmArrayAlloc(shape.toArray, dtype, ctx, refHandle))
    new NDArray(refHandle.value, false, dtype)
  }

  /**
   * Create an array from source arr.
   * @param arr The source array to be copied from
   * @param shape The shape of the nd array
   * @param dtype The data type of the nd array
   * @param ctx The context of the nd array
   * @return The created nd array
   */
  def array(arr: Array[Double],
            shape: Shape,
            dtype: TVMType = TVMType("float32"),
            ctx: TVMContext = new TVMContext(1, 0)): NDArray = {
    val ndArray = empty(shape, dtype, ctx)
    ndArray.set(arr)
    ndArray
  }
}

private[tvm] object NDArrayInternal {
  def wrapBytes(bytes: Array[Byte]): ByteBuffer = {
    val bb = ByteBuffer.wrap(bytes)
    bb.order(ByteOrder.LITTLE_ENDIAN)
    bb
  }

  def wrapBytes(bytes: Array[Byte], offset: Int, length: Int): ByteBuffer = {
    val bb = ByteBuffer.wrap(bytes, offset, length)
    bb.order(ByteOrder.LITTLE_ENDIAN)
    bb
  }
}

private[tvm] class NDArrayInternal(private val internal: Array[Byte], private val dtype: TVMType) {
  private val unitSize = dtype.numOfBytes
  require(internal.length > 0 && internal.length % unitSize == 0,
    s"$dtype size $unitSize cannot divide byte array size ${internal.length}")

  private val units: Array[Array[Byte]] = (
    for (i <- 0 until internal.length / unitSize)
      yield internal.slice(i * unitSize, (i + 1) * unitSize)
    ).toArray

  def getRaw: Array[Byte] = internal

  def toArray: Array[Double] = {
    dtype.typeCode match {
      case TVMType.INT =>
        dtype.bits match {
          case 8 => internal.map(_.toDouble)
          case 16 => units.map(NDArrayInternal.wrapBytes(_).getShort.toDouble)
          case 32 => units.map(NDArrayInternal.wrapBytes(_).getInt.toDouble)
          case 64 => units.map(NDArrayInternal.wrapBytes(_).getLong.toDouble)
          case _ => throw new IllegalArgumentException("Do not know how to handle type " + dtype)
        }
      case TVMType.UINT =>
        dtype.bits match {
          case 8 => internal.map { x =>
           val i = x.toInt & 0xFF
           i.toDouble
          }
          case 16 => units.map { x =>
            val i = NDArrayInternal.wrapBytes(x).getShort.toInt & 0xFFFF
            i.toDouble
          }
          case 32 => units.map { x =>
            val i = NDArrayInternal.wrapBytes(x).getInt.toLong & 0xFFFFFFFFL
            i.toDouble
          }
          case _ => throw new IllegalArgumentException("Do not know how to handle type " + dtype)
        }
      case TVMType.FLOAT =>
        dtype.bits match {
          case 16 => throw new IllegalArgumentException(
            "Currently cannot convert float16 to native numerical types")
          case 32 => units.map(NDArrayInternal.wrapBytes(_).getFloat.toDouble)
          case 64 => units.map(NDArrayInternal.wrapBytes(_).getDouble)
          case _ => throw new IllegalArgumentException("Do not know how to handle type " + dtype)
        }
    }
  }
}
