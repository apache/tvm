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

// Type code used in API calls
public enum ArgTypeCode {
  INT(0), UINT(1), FLOAT(2), HANDLE(3), NULL(4), TVM_TYPE(5),
  DLDEVICE(6), ARRAY_HANDLE(7), NODE_HANDLE(8), MODULE_HANDLE(9),
  FUNC_HANDLE(10), STR(11), BYTES(12), NDARRAY_CONTAINER(13);

  public final int id;

  private ArgTypeCode(int id) {
    this.id = id;
  }

  @Override
  public String toString() {
    return String.valueOf(id);
  }
}
