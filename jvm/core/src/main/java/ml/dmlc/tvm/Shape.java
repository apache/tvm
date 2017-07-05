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

import java.util.Arrays;
import java.util.Collection;

/**
 * Shape of [[NDArray]] or other data
 */
class Shape {
  private long[] shape;

  public Shape(Collection<Long> dims){
    if (dims == null || dims.size() == 0) {
      throw new IllegalArgumentException("Invalid input dim");
    }
    shape = new long[dims.size()];
    int idx = 0;
    for (Long dim : dims) {
      shape[idx++] = dim;
    }
  }

  public Shape(long... dims) {
    if (dims == null || dims.length == 0) {
      throw new IllegalArgumentException("Invalid input dim");
    }
    shape = dims.clone();
  }

  public int size() {
    return shape.length;
  }

  public Shape slice(int from, int end) {
    if (from < 0 || from > shape.length || end <= from || end > size()) {
      throw new IllegalArgumentException(
        String.format("Invalid slice range (%d, %d)", from, end));
    }
    long[] newShape = Arrays.copyOfRange(shape, from, end);
    return new Shape(newShape);
  }

  public Shape drop(int dim) {
    int size = size();
    if (dim < 0 || dim >= size || size == 1) {
      throw new IllegalArgumentException("Invalid drop dim " + dim);
    }
    long[] newShape = new long[size - 1];
    for (int i = 0; i < size; ++i) {
      if (i != dim) {
        newShape[i] = shape[i];
      }
    }
    return new Shape(newShape);
  }

  public long product() {
    long res = 1;
    for (int i = 0; i < size(); ++i) {
      res *= shape[i];
    }
    return res;
  }

  public long head() {
    return shape[0];
  }

  public long[] toArray() {
    return shape.clone();
  }

  public static Shape concat(Shape s1, Shape s2) {
    long[] newShape = new long[s1.size() + s2.size()];
    System.arraycopy(s1.shape, 0, newShape, 0, s1.size());
    System.arraycopy(s2.shape, 0, newShape, s1.size(), s2.size());
    return new Shape(newShape);
  }

  @Override public String toString() {
    StringBuilder builder = new StringBuilder("(");
    int size = size();
    for (int i = 0; i < size; ++i) {
      builder.append(shape[i]);
      if (i < size - 1) {
        builder.append(", ");
      }
    }
    builder.append(")");
    return builder.toString();
  }

  @Override public boolean equals(Object o) {
    if (o != null && o instanceof Shape) {
      Shape that = (Shape)o;
      return Arrays.equals(shape, that.shape);
    }
    return false;
  }

  @Override public int hashCode() {
    return Arrays.hashCode(shape);
  }
}

