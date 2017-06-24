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

import ml.dmlc.tvm.Base._
import ml.dmlc.tvm.{NDArray, Module}
import ml.dmlc.tvm.types.TypeCode._

private[tvm] object TVMValue {
  implicit def fromInt(x: Int): TVMValue = new TVMValueLong(x)
  implicit def fromLong(x: Long): TVMValue = new TVMValueLong(x)
  implicit def fromDouble(x: Double): TVMValue = new TVMValueDouble(x)
  implicit def fromFloat(x: Float): TVMValue = new TVMValueDouble(x)
  implicit def fromString(x: String): TVMValue = new TVMValueString(x)
  implicit def fromModule(x: Module): TVMValue = new TVMValueModuleHandle(x.handle)
  implicit def fromNDArray(x: NDArray): TVMValue = new TVMValueNDArrayHandle(x.handle)
}

private[tvm] sealed class TVMValue(val argType: TypeCode) {
  // easy for JNI to use
  val argTypeId = argType.id
}

private[tvm] sealed class TVMValueLong(val value: Long) extends TVMValue(INT)
private[tvm] sealed class TVMValueDouble(val value: Double) extends TVMValue(FLOAT)
private[tvm] sealed class TVMValueString(val value: String) extends TVMValue(STR)
private[tvm] sealed class TVMValueModuleHandle(
  val value: ModuleHandle) extends TVMValue(MODULE_HANDLE)
private[tvm] sealed class TVMValueNDArrayHandle(
  val value: TVMArrayHandle) extends TVMValue(ARRAY_HANDLE)
private[tvm] sealed class TVMValueNull extends TVMValue(NULL)
