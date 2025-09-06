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

public class TensorTest {
  @Test
  public void test_from_float32() {
    Tensor tensor = Tensor.empty(new long[]{2, 2}, new TVMType("float32"));
    tensor.copyFrom(new float[]{1, 2, 3, 4});
    assertArrayEquals(new float[]{1f, 2f, 3f, 4f}, tensor.asFloatArray(), 1e-3f);
    tensor.release();
  }

  @Test
  public void test_from_float64() {
    Tensor tensor = Tensor.empty(new long[]{2, 2}, new TVMType("float64"));
    tensor.copyFrom(new double[]{1, 2, 3, 4});
    assertArrayEquals(new double[]{1.0, 2.0, 3.0, 4.0}, tensor.asDoubleArray(), 1e-3);
    tensor.release();
  }

  @Test
  public void test_from_int8() {
    Tensor tensor = Tensor.empty(new long[]{2, 2}, new TVMType("int8"));
    tensor.copyFrom(new byte[]{1, 2, 3, 4});
    assertArrayEquals(new byte[]{1, 2, 3, 4}, tensor.asByteArray());
    tensor.release();
  }

  @Test
  public void test_from_int16() {
    Tensor tensor = Tensor.empty(new long[]{2, 2}, new TVMType("int16"));
    tensor.copyFrom(new short[]{1, 2, 3, 4});
    assertArrayEquals(new short[]{1, 2, 3, 4}, tensor.asShortArray());
    tensor.release();
  }

  @Test
  public void test_from_int32() {
    Tensor tensor = Tensor.empty(new long[]{2, 2}, new TVMType("int32"));
    tensor.copyFrom(new int[]{1, 2, 3, 4});
    assertArrayEquals(new int[]{1, 2, 3, 4}, tensor.asIntArray());
    tensor.release();
  }

  @Test
  public void test_from_int64() {
    Tensor tensor = Tensor.empty(new long[]{2, 2}, new TVMType("int64"));
    tensor.copyFrom(new long[]{1, 2, 3, 4});
    assertArrayEquals(new long[]{1, 2, 3, 4}, tensor.asLongArray());
    tensor.release();
  }

  @Test
  public void test_from_uint16() {
    Tensor tensor = Tensor.empty(new long[]{2, 2}, new TVMType("uint16"));
    tensor.copyFrom(new char[]{65535, 2, 3, 4});
    assertArrayEquals(new char[]{65535, 2, 3, 4}, tensor.asCharArray());
    tensor.release();
  }
}
