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

package org.apache.tvm;

import org.junit.Test;

import static org.junit.Assert.*;

public class NDArrayTest {
  @Test
  public void test_from_float32() {
    NDArray ndarray = NDArray.empty(new long[]{2, 2}, new TVMType("float32"));
    ndarray.copyFrom(new float[]{1, 2, 3, 4});
    assertArrayEquals(new float[]{1f, 2f, 3f, 4f}, ndarray.asFloatArray(), 1e-3f);
    ndarray.release();
  }

  @Test
  public void test_from_float64() {
    NDArray ndarray = NDArray.empty(new long[]{2, 2}, new TVMType("float64"));
    ndarray.copyFrom(new double[]{1, 2, 3, 4});
    assertArrayEquals(new double[]{1.0, 2.0, 3.0, 4.0}, ndarray.asDoubleArray(), 1e-3);
    ndarray.release();
  }

  @Test
  public void test_from_int8() {
    NDArray ndarray = NDArray.empty(new long[]{2, 2}, new TVMType("int8"));
    ndarray.copyFrom(new byte[]{1, 2, 3, 4});
    assertArrayEquals(new byte[]{1, 2, 3, 4}, ndarray.asByteArray());
    ndarray.release();
  }

  @Test
  public void test_from_int16() {
    NDArray ndarray = NDArray.empty(new long[]{2, 2}, new TVMType("int16"));
    ndarray.copyFrom(new short[]{1, 2, 3, 4});
    assertArrayEquals(new short[]{1, 2, 3, 4}, ndarray.asShortArray());
    ndarray.release();
  }

  @Test
  public void test_from_int32() {
    NDArray ndarray = NDArray.empty(new long[]{2, 2}, new TVMType("int32"));
    ndarray.copyFrom(new int[]{1, 2, 3, 4});
    assertArrayEquals(new int[]{1, 2, 3, 4}, ndarray.asIntArray());
    ndarray.release();
  }

  @Test
  public void test_from_int64() {
    NDArray ndarray = NDArray.empty(new long[]{2, 2}, new TVMType("int64"));
    ndarray.copyFrom(new long[]{1, 2, 3, 4});
    assertArrayEquals(new long[]{1, 2, 3, 4}, ndarray.asLongArray());
    ndarray.release();
  }

  @Test
  public void test_from_uint16() {
    NDArray ndarray = NDArray.empty(new long[]{2, 2}, new TVMType("uint16"));
    ndarray.copyFrom(new char[]{65535, 2, 3, 4});
    assertArrayEquals(new char[]{65535, 2, 3, 4}, ndarray.asCharArray());
    ndarray.release();
  }
}
