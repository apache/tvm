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

  public void set(double[] source) {
    syncCopyFrom(source);
  }

  /**
   * Perform an synchronize copy from the array.
   * @param sourceArray The data source we should like to copy from.
   */
  private void syncCopyFrom(double[] sourceArray) {
    if (size() != sourceArray.length) {
      throw new IllegalArgumentException(String.format("Array shape size not match: %d v.s. %d",
        sourceArray.length, size()));
    }

    NDArray tmpArr = empty(shape(), this.dtype);

    byte[] nativeArr = new byte[sourceArray.length * dtype.numOfBytes];
    switch (dtype.typeCode) {
    case TVMType.INT:
    case TVMType.UINT:
      switch (dtype.bits) {
      case 8:
        for (int i = 0; i < sourceArray.length; ++i) {
          nativeArr[i] = (byte) sourceArray[i];
        }
        break;
      case 16:
        for (int i = 0; i < sourceArray.length; ++i) {
          wrapBytes(nativeArr, i * dtype.numOfBytes, dtype.numOfBytes).putShort(
            (short) sourceArray[i]);
        }
        break;
      case 32:
        for (int i = 0; i < sourceArray.length; ++i) {
          wrapBytes(nativeArr, i * dtype.numOfBytes, dtype.numOfBytes)
            .putInt((int) ((long) sourceArray[i]));
        }
        break;
      case 64:
        for (int i = 0; i < sourceArray.length; ++i) {
          wrapBytes(nativeArr, i * dtype.numOfBytes, dtype.numOfBytes)
            .putLong((long) sourceArray[i]);
        }
        break;
      default:
        throw new IllegalArgumentException("Do not know how to handle type " + dtype);
      }
      break;
    case TVMType.FLOAT:
      switch (dtype.bits) {
      case 16:
        throw new IllegalArgumentException(
          "Currently cannot convert native numerical types to float16");
      case 32:
        for (int i = 0; i < sourceArray.length; ++i) {
          wrapBytes(nativeArr, i * dtype.numOfBytes, dtype.numOfBytes)
            .putFloat((float) sourceArray[i]);
        }
        break;
      case 64:
        for (int i = 0; i < sourceArray.length; ++i) {
          wrapBytes(nativeArr, i * dtype.numOfBytes, dtype.numOfBytes)
            .putDouble(sourceArray[i]);
        }
        break;
      default:
        throw new IllegalArgumentException("Do not know how to handle type " + dtype);
      }
      break;
    default:
      throw new IllegalArgumentException("Do not know how to handle type " + dtype);
    }
    Base.checkCall(Base._LIB.tvmArrayCopyFromJArray(nativeArr, tmpArr.handle, handle));
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
   * @return  A copy of array content.
   */
  public double[] toArray() {
    return internal().toArray();
  }

  private NDArrayInternal internal() {
    NDArray tmp = NDArray.empty(shape(), dtype);
    Base.checkCall(Base._LIB.tvmArrayCopyFromTo(handle, tmp.handle));

    int arrLength = dtype.numOfBytes * (int) size();
    byte[] arr = new byte[arrLength];
    Base.checkCall(Base._LIB.tvmArrayCopyToJArray(tmp.handle, arr));
    return new NDArrayInternal(arr, dtype);
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

  /**
   * Create an array from source arr.
   * @param arr The source array to be copied from
   * @param shape The shape of the nd array
   * @param dtype The data type of the nd array
   * @param ctx The context of the nd array
   * @return The created nd array
   */
  public static NDArray array(double[] arr, long[] shape, TVMType dtype, TVMContext ctx) {
    NDArray ndArray = empty(shape, dtype, ctx);
    ndArray.set(arr);
    return ndArray;
  }

  public static NDArray array(double[] arr, long[] shape, TVMType dtype) {
    NDArray ndArray = empty(shape, dtype);
    ndArray.set(arr);
    return ndArray;
  }

  public static NDArray array(double[] arr, long[] shape) {
    NDArray ndArray = empty(shape);
    ndArray.set(arr);
    return ndArray;
  }

  public static NDArray array(double[] arr, long[] shape, TVMContext ctx) {
    NDArray ndArray = empty(shape, ctx);
    ndArray.set(arr);
    return ndArray;
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

  private static class NDArrayInternal {
    private final byte[] internal;
    private final TVMType dtype;
    private final int unitSize;
    private final byte[][] units;

    public NDArrayInternal(byte[] internal, TVMType dtype) {
      this.internal = internal;
      this.dtype = dtype;

      unitSize = dtype.numOfBytes;
      if (internal.length <= 0 || internal.length % unitSize != 0) {
        throw new IllegalArgumentException(String.format(
          "%s size %d cannot divide byte array size %d",
          dtype.toString(), unitSize, internal.length));
      }

      int numOfUnits = internal.length / unitSize;
      units = new byte[numOfUnits][unitSize];
      for (int i = 0; i < numOfUnits; ++i) {
        System.arraycopy(internal, i * unitSize, units[i], 0, unitSize);
      }
    }

    public double[] toArray() {
      double[] array = new double[units.length];
      switch (dtype.typeCode) {
      case TVMType.INT:
        switch (dtype.bits) {
        case 8:
          for (int i = 0; i < internal.length; ++i) {
            array[i] = internal[i];
          }
          break;
        case 16:
          for (int i = 0; i < units.length; ++i) {
            array[i] = wrapBytes(units[i]).getShort();
          }
          break;
        case 32:
          for (int i = 0; i < units.length; ++i) {
            array[i] = wrapBytes(units[i]).getInt();
          }
          break;
        case 64:
          for (int i = 0; i < units.length; ++i) {
            array[i] = wrapBytes(units[i]).getLong();
          }
          break;
        default:
          throw new IllegalArgumentException("Do not know how to handle type " + dtype);
        }
        break;
      case TVMType.UINT:
        switch (dtype.bits) {
        case 8:
          for (int i = 0; i < internal.length; ++i) {
            array[i] = ((int) internal[i]) & 0xFF;
          }
          break;
        case 16:
          for (int i = 0; i < units.length; ++i) {
            array[i] = ((int) wrapBytes(units[i]).getShort()) & 0xFFFF;
          }
          break;
        case 32:
          for (int i = 0; i < units.length; ++i) {
            array[i] = ((long) wrapBytes(units[i]).getInt()) & 0xFFFFFFFFL;
          }
          break;
        default:
          throw new IllegalArgumentException("Do not know how to handle type " + dtype);
        }
        break;
      case TVMType.FLOAT:
        switch (dtype.bits) {
        case 16:
          throw new IllegalArgumentException(
            "Currently cannot convert float16 to native numerical types");
        case 32:
          for (int i = 0; i < units.length; ++i) {
            array[i] = wrapBytes(units[i]).getFloat();
          }
          break;
        case 64:
          for (int i = 0; i < units.length; ++i) {
            array[i] = wrapBytes(units[i]).getDouble();
          }
          break;
        default:
          throw new IllegalArgumentException("Do not know how to handle type " + dtype);
        }
        break;
      default:
        throw new IllegalArgumentException("Do not know how to handle type " + dtype);
      }
      return array;
    }
  }
}
