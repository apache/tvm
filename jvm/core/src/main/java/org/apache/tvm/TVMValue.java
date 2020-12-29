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

public class TVMValue {
  public final ArgTypeCode typeCode;

  public TVMValue(ArgTypeCode tc) {
    typeCode = tc;
  }

  public void release() {
  }

  public long asLong() {
    throw new UnsupportedOperationException();
  }

  public double asDouble() {
    throw new UnsupportedOperationException();
  }

  public byte[] asBytes() {
    throw new UnsupportedOperationException();
  }

  public Module asModule() {
    throw new UnsupportedOperationException();
  }

  public Function asFunction() {
    throw new UnsupportedOperationException();
  }

  public NDArrayBase asNDArray() {
    throw new UnsupportedOperationException();
  }

  public String asString() {
    throw new UnsupportedOperationException();
  }

  // easy for JNI to use.
  long asHandle() {
    throw new UnsupportedOperationException();
  }
}
