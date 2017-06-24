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

package ml.dmlc.tvm.types

// Type code used in API calls
object TypeCode extends Enumeration {
  type TypeCode = Value
  val INT = Value(0)
  val UINT = Value(1)
  val FLOAT = Value(2)
  val HANDLE = Value(3)
  val NULL = Value(4)
  val TVM_TYPE = Value(5)
  val TVM_CONTEXT = Value(6)
  val ARRAY_HANDLE = Value(7)
  val NODE_HANDLE = Value(8)
  val MODULE_HANDLE = Value(9)
  val FUNC_HANDLE = Value(10)
  val STR = Value(11)
  val BYTES = Value(12)
}
