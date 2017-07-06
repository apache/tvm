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

package ml.dmlc.tvm.types;

import ml.dmlc.tvm.Module;
import ml.dmlc.tvm.NDArray;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class TVMValue {
  public final TypeCode typeCode;
  public TVMValue(TypeCode tc) {
    typeCode = tc;
  }

  public long asLong() {
    throw new NotImplementedException();
  }

  public double asDouble() {
    throw new NotImplementedException();
  }

  public Module asModule() {
    throw new NotImplementedException();
  }

  public NDArray asNDArray() {
    throw new NotImplementedException();
  }

  public String asString() {
    throw new NotImplementedException();
  }
}
