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

package ml.dmlc.tvm;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * Lightweight NDArray class of TVM runtime.
 */
public class NDArray extends NDArrayBase {
  private final TVMType dtype;

  NDArray(long handle, boolean isView, TVMType dtype) {
    super(handle, isView);
    this.dtype = dtype;
  }

  @Override protected void finalize() throws Throwable {
    super.finalize();
  }

  /**
   * Copy from a native array.
   * The NDArray type must by float64
   * @param sourceArray the source data
   */
  public void copyFrom(double[] sourceArray) {
    checkCopySize(sourceArray.length);
    if (dtype.typeCode != TVMType.FLOAT || dtype.bits != 64) {
      throw new IllegalArgumentException("Cannot set double[] for " + dtype.toString() + " array");
    }
    byte[] nativeArr = new byte[sourceArray.length * dtype.numOfBytes];
    for (int i = 0; i < sourceArray.length; ++i) {
      wrapBytes(nativeArr, i * dtype.numOfBytes, dtype.numOfBytes).putDouble(sourceArray[i]);
    }
    NDArray tmpArr = empty(shape(), this.dtype);
    Base.checkCall(Base._LIB.tvmArrayCopyFromJArray(nativeArr, tmpArr.handle, handle));
    Base.checkCall(Base._LIB.tvmArrayFree(tmpArr.handle));
  }

  /**
   * Copy from a native array.
   * The NDArray type must by float32
   * @param sourceArray the source data
   */
  public void copyFrom(float[] sourceArray) {
    checkCopySize(sourceArray.length);
    if (dtype.typeCode != TVMType.FLOAT || dtype.bits != 32) {
      throw new IllegalArgumentException("Cannot set float[] for " + dtype.toString() + " array");
    }
    byte[] nativeArr = new byte[sourceArray.length * dtype.numOfBytes];
    for (int i = 0; i < sourceArray.length; ++i) {
      wrapBytes(nativeArr, i * dtype.numOfBytes, dtype.numOfBytes).putFloat(sourceArray[i]);
    }
    NDArray tmpArr = empty(shape(), this.dtype);
    Base.checkCall(Base._LIB.tvmArrayCopyFromJArray(nativeArr, tmpArr.handle, handle));
    Base.checkCall(Base._LIB.tvmArrayFree(tmpArr.handle));
  }

  /**
   * Copy from a native array.
   * The NDArray type must by int64
   * @param sourceArray the source data
   */
  public void copyFrom(long[] sourceArray) {
    checkCopySize(sourceArray.length);
    if (dtype.typeCode != TVMType.INT || dtype.bits != 64) {
      throw new IllegalArgumentException("Cannot set long[] for " + dtype.toString() + " array");
    }
    byte[] nativeArr = new byte[sourceArray.length * dtype.numOfBytes];
    for (int i = 0; i < sourceArray.length; ++i) {
      wrapBytes(nativeArr, i * dtype.numOfBytes, dtype.numOfBytes).putLong(sourceArray[i]);
    }
    NDArray tmpArr = empty(shape(), this.dtype);
    Base.checkCall(Base._LIB.tvmArrayCopyFromJArray(nativeArr, tmpArr.handle, handle));
    Base.checkCall(Base._LIB.tvmArrayFree(tmpArr.handle));
  }

  /**
   * Copy from a native array.
   * The NDArray type must by float32
   * @param sourceArray the source data
   */
  public void copyFrom(int[] sourceArray) {
    checkCopySize(sourceArray.length);
    if (dtype.typeCode != TVMType.INT || dtype.bits != 32) {
      throw new IllegalArgumentException("Cannot set int[] for " + dtype.toString() + " array");
    }
    byte[] nativeArr = new byte[sourceArray.length * dtype.numOfBytes];
    for (int i = 0; i < sourceArray.length; ++i) {
      wrapBytes(nativeArr, i * dtype.numOfBytes, dtype.numOfBytes).putInt(sourceArray[i]);
    }
    NDArray tmpArr = empty(shape(), this.dtype);
    Base.checkCall(Base._LIB.tvmArrayCopyFromJArray(nativeArr, tmpArr.handle, handle));
    Base.checkCall(Base._LIB.tvmArrayFree(tmpArr.handle));
  }

  /**
   * Copy from a native array.
   * The NDArray type must by int16
   * @param sourceArray the source data
   */
  public void copyFrom(short[] sourceArray) {
    checkCopySize(sourceArray.length);
    if (dtype.typeCode != TVMType.INT || dtype.bits != 16) {
      throw new IllegalArgumentException("Cannot set short[] for " + dtype.toString() + " array");
    }
    byte[] nativeArr = new byte[sourceArray.length * dtype.numOfBytes];
    for (int i = 0; i < sourceArray.length; ++i) {
      wrapBytes(nativeArr, i * dtype.numOfBytes, dtype.numOfBytes).putShort(sourceArray[i]);
    }
    NDArray tmpArr = empty(shape(), this.dtype);
    Base.checkCall(Base._LIB.tvmArrayCopyFromJArray(nativeArr, tmpArr.handle, handle));
    Base.checkCall(Base._LIB.tvmArrayFree(tmpArr.handle));
  }

  /**
   * Copy from a native array.
   * The NDArray type must by int8
   * @param sourceArray the source data
   */
  public void copyFrom(byte[] sourceArray) {
    checkCopySize(sourceArray.length);
    if (dtype.typeCode != TVMType.INT || dtype.bits != 8) {
      throw new IllegalArgumentException("Cannot set byte[] for " + dtype.toString() + " array");
    }
    copyFromRaw(sourceArray);
  }

  /**
   * Copy from a native array.
   * The NDArray type must by uint16
   * @param sourceArray the source data
   */
  public void copyFrom(char[] sourceArray) {
    checkCopySize(sourceArray.length);
    if (dtype.typeCode != TVMType.UINT || dtype.bits != 16) {
      throw new IllegalArgumentException("Cannot set char[] for " + dtype.toString() + " array");
    }
    byte[] nativeArr = new byte[sourceArray.length * dtype.numOfBytes];
    for (int i = 0; i < sourceArray.length; ++i) {
      wrapBytes(nativeArr, i * dtype.numOfBytes, dtype.numOfBytes).putChar(sourceArray[i]);
    }
    NDArray tmpArr = empty(shape(), this.dtype);
    Base.checkCall(Base._LIB.tvmArrayCopyFromJArray(nativeArr, tmpArr.handle, handle));
    Base.checkCall(Base._LIB.tvmArrayFree(tmpArr.handle));
  }

  private void checkCopySize(int sourceLength) {
    long arrSize = size();
    if (arrSize != sourceLength) {
      throw new IllegalArgumentException(String.format("Array shape size not match: %d v.s. %d",
        sourceLength, size()));
    }
  }

  /**
   * Copy from a raw byte array.
   * @param sourceArray the source data
   */
  public void copyFromRaw(byte[] sourceArray) {
    NDArray tmpArr = empty(shape(), this.dtype);
    Base.checkCall(Base._LIB.tvmArrayCopyFromJArray(sourceArray, tmpArr.handle, handle));
    Base.checkCall(Base._LIB.tvmArrayFree(tmpArr.handle));
  }

  /**
   * Get shape of current NDArray.
   * @return an array representing shape of current ndarray
   */
  public long[] shape() {
    List<Long> data = new ArrayList<Long>();
    Base.checkCall(Base._LIB.tvmArrayGetShape(handle, data));
    long[] shapeArr = new long[data.size()];
    for (int i = 0; i < shapeArr.length; ++i) {
      shapeArr[i] = data.get(i);
    }
    return shapeArr;
  }

  /**
   * Get total size of current NDArray.
   * @return size of current NDArray.
   */
  public long size() {
    long product = 1L;
    long[] shapeArr = shape();
    for (int i = 0; i < shapeArr.length; ++i) {
      product *= shapeArr[i];
    }
    return product;
  }

  /**
   * Return a copied flat java array of current array (row-major).
   * The NDArray dtype must be float64
   * @return A copy of array content.
   */
  public double[] asDoubleArray() {
    if (dtype.typeCode != TVMType.FLOAT || dtype.bits != 64) {
      throw new IllegalArgumentException(
        "Cannot set convert to double[] for " + dtype.toString() + " array");
    }
    byte[][] units = groupInternalBytes();
    double[] array = new double[units.length];
    for (int i = 0; i < units.length; ++i) {
      array[i] = wrapBytes(units[i]).getDouble();
    }
    return array;
  }

  /**
   * Return a copied flat java array of current array (row-major).
   * The NDArray dtype must be float32
   * @return A copy of array content.
   */
  public float[] asFloatArray() {
    if (dtype.typeCode != TVMType.FLOAT || dtype.bits != 32) {
      throw new IllegalArgumentException(
        "Cannot set convert to float[] for " + dtype.toString() + " array");
    }
    byte[][] units = groupInternalBytes();
    float[] array = new float[units.length];
    for (int i = 0; i < units.length; ++i) {
      array[i] = wrapBytes(units[i]).getFloat();
    }
    return array;
  }

  /**
   * Return a copied flat java array of current array (row-major).
   * The NDArray dtype must be int64
   * @return A copy of array content.
   */
  public long[] asLongArray() {
    if (dtype.typeCode != TVMType.INT || dtype.bits != 64) {
      throw new IllegalArgumentException(
        "Cannot set convert to long[] for " + dtype.toString() + " array");
    }
    byte[][] units = groupInternalBytes();
    long[] array = new long[units.length];
    for (int i = 0; i < units.length; ++i) {
      array[i] = wrapBytes(units[i]).getLong();
    }
    return array;
  }

  /**
   * Return a copied flat java array of current array (row-major).
   * The NDArray dtype must be int32
   * @return A copy of array content.
   */
  public int[] asIntArray() {
    if (dtype.typeCode != TVMType.INT || dtype.bits != 32) {
      throw new IllegalArgumentException(
        "Cannot set convert to int[] for " + dtype.toString() + " array");
    }
    byte[][] units = groupInternalBytes();
    int[] array = new int[units.length];
    for (int i = 0; i < units.length; ++i) {
      array[i] = wrapBytes(units[i]).getInt();
    }
    return array;
  }

  /**
   * Return a copied flat java array of current array (row-major).
   * The NDArray dtype must be int16
   * @return A copy of array content.
   */
  public short[] asShortArray() {
    if (dtype.typeCode != TVMType.INT || dtype.bits != 16) {
      throw new IllegalArgumentException(
        "Cannot set convert to short[] for " + dtype.toString() + " array");
    }
    byte[][] units = groupInternalBytes();
    short[] array = new short[units.length];
    for (int i = 0; i < units.length; ++i) {
      array[i] = wrapBytes(units[i]).getShort();
    }
    return array;
  }

  /**
   * Return a copied flat java array of current array (row-major).
   * The NDArray dtype must be uint16
   * @return A copy of array content.
   */
  public char[] asCharArray() {
    if (dtype.typeCode != TVMType.UINT || dtype.bits != 16) {
      throw new IllegalArgumentException(
        "Cannot set convert to char[] for " + dtype.toString() + " array");
    }
    byte[][] units = groupInternalBytes();
    char[] array = new char[units.length];
    for (int i = 0; i < units.length; ++i) {
      array[i] = wrapBytes(units[i]).getChar();
    }
    return array;
  }

  /**
   * Return a copied flat java array of current array (row-major).
   * The NDArray dtype must be int8
   * @return A copy of array content.
   */
  public byte[] asByteArray() {
    if (dtype.typeCode != TVMType.INT || dtype.bits != 8) {
      throw new IllegalArgumentException(
        "Cannot set convert to byte[] for " + dtype.toString() + " array");
    }
    return internal();
  }

  /**
   * Return a copied internal byte array of current array (row-major).
   * @return A copy of array content.
   */
  public byte[] internal() {
    NDArray tmp = NDArray.empty(shape(), dtype);
    copyTo(tmp);

    int arrLength = dtype.numOfBytes * (int) size();
    byte[] arr = new byte[arrLength];
    Base.checkCall(Base._LIB.tvmArrayCopyToJArray(tmp.handle, arr));
    return arr;
  }

  private byte[][] groupInternalBytes() {
    byte[] raw = internal();
    int unitSize = dtype.numOfBytes;
    if (raw.length <= 0 || raw.length % unitSize != 0) {
      throw new IllegalArgumentException(String.format(
        "%s size %d cannot divide byte array size %d",
        dtype.toString(), unitSize, raw.length));
    }

    int numOfUnits = raw.length / unitSize;
    byte[][] units = new byte[numOfUnits][unitSize];
    for (int i = 0; i < numOfUnits; ++i) {
      System.arraycopy(raw, i * unitSize, units[i], 0, unitSize);
    }
    return units;
  }

  /**
   * Create an empty array given shape, type and device.
   * @param shape The shape of the array.
   * @param dtype The data type of the array.
   * @param ctx The context of the array.
   * @return The array tvm supported.
   */
  public static NDArray empty(long[] shape, TVMType dtype, TVMContext ctx) {
    Base.RefLong refHandle = new Base.RefLong();
    Base.checkCall(Base._LIB.tvmArrayAlloc(
        shape, dtype.typeCode, dtype.bits, dtype.lanes,
        ctx.deviceType, ctx.deviceId, refHandle));
    return new NDArray(refHandle.value, false, dtype);
  }

  /**
   * Create an empty array on cpu given shape and type.
   * @param shape The shape of the array.
   * @param dtype The data type of the array.
   * @return The array tvm supported.
   */
  public static NDArray empty(long[] shape, TVMType dtype) {
    return empty(shape, dtype, new TVMContext(1, 0));
  }

  /**
   * Create an empty float32 array on cpu given shape.
   * @param shape The shape of the array.
   * @return The array tvm supported.
   */
  public static NDArray empty(long[] shape) {
    return empty(shape, new TVMType("float32", 1), new TVMContext(1, 0));
  }

  /**
   * Create an empty float32 array given shape and device.
   * @param shape The shape of the array.
   * @param ctx The context of the array.
   * @return The array tvm supported.
   */
  public static NDArray empty(long[] shape, TVMContext ctx) {
    return empty(shape, new TVMType("float32", 1), ctx);
  }

  private static ByteBuffer wrapBytes(byte[] bytes) {
    ByteBuffer bb = ByteBuffer.wrap(bytes);
    bb.order(ByteOrder.LITTLE_ENDIAN);
    return bb;
  }

  private static ByteBuffer wrapBytes(byte[] bytes, int offset, int length) {
    ByteBuffer bb = ByteBuffer.wrap(bytes, offset, length);
    bb.order(ByteOrder.LITTLE_ENDIAN);
    return bb;
  }
}
