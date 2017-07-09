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

import java.nio.ByteOrder;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import ml.dmlc.tvm.Base.*;
import ml.dmlc.tvm.types.TVMType;
import ml.dmlc.tvm.types.TVMContext;

/**
 * Lightweight NDArray class of TVM runtime.
 */
public class NDArray {
  public final long handle;
  private final boolean isView;
  private final TVMType dtype;

  public NDArray(long handle, boolean isView, TVMType dtype){
    this.handle = handle;
    this.isView = isView;
    this.dtype = dtype;
  }

  public NDArray(long handle) {
    this(handle, false, new TVMType("float32", 1));
  }

  public NDArray(long handle, boolean isView) {
    this(handle, isView, new TVMType("float32", 1));
  }

  @Override
  protected void finalize() {
    if (!isView) {
      Base.checkCall(Base._LIB.tvmArrayFree(handle));
    }
  }

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

  public void copyFrom(byte[] sourceArray) {
    checkCopySize(sourceArray.length);
    if (dtype.typeCode != TVMType.INT || dtype.bits != 8) {
      throw new IllegalArgumentException("Cannot set byte[] for " + dtype.toString() + " array");
    }
    copyFromRaw(sourceArray);
  }

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

  // Get size of current NDArray.
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

  public byte[] asByteArray() {
    if (dtype.typeCode != TVMType.INT || dtype.bits != 8) {
      throw new IllegalArgumentException(
        "Cannot set convert to byte[] for " + dtype.toString() + " array");
    }
    return internal();
  }

  public byte[] internal() {
    NDArray tmp = NDArray.empty(shape(), dtype);
    Base.checkCall(Base._LIB.tvmArrayCopyFromTo(handle, tmp.handle));

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
   * Create an empty array given shape and device
   * @param shape The shape of the array
   * @param dtype The data type of the array
   * @param ctx The context of the array
   * @return The array tvm supported
   */
  public static NDArray empty(long[] shape, TVMType dtype, TVMContext ctx) {
    RefLong refHandle = new RefLong();
    Base.checkCall(Base._LIB.tvmArrayAlloc(shape, dtype, ctx, refHandle));
    return new NDArray(refHandle.value, false, dtype);
  }

  public static NDArray empty(long[] shape, TVMType dtype) {
    return empty(shape, dtype, new TVMContext(1, 0));
  }

  public static NDArray empty(long[] shape) {
    return empty(shape, new TVMType("float32", 1), new TVMContext(1, 0));
  }

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
