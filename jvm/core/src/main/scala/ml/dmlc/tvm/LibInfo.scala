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

package ml.dmlc.tvm

import ml.dmlc.tvm.Base._
import ml.dmlc.tvm.types.{TVMContext, TVMType, TVMValue}

import scala.collection.mutable.ArrayBuffer

private[tvm] class LibInfo {
  @native def nativeLibInit(): Int
  @native def tvmGetLastError(): String

  // Function
  @native def tvmFuncListGlobalNames(funcNames: ArrayBuffer[String]): Int
  @native def tvmFuncFree(handle: FunctionHandle): Int
  @native def tvmFuncGetGlobal(name: String, handle: RefFunctionHandle): Int
  @native def tvmFuncCall(handle: FunctionHandle, args: Array[TVMValue], retVal: RefTVMValue): Int

  // Module
  @native def tvmModFree(handle: ModuleHandle): Int
  @native def tvmModGetFunction(handle: ModuleHandle, name: String,
                                queryImports: Int, retHandle: RefFunctionHandle): Int
  @native def tvmModImport(mod: ModuleHandle, dep: ModuleHandle): Int

  // NDArray
  @native def tvmArrayFree(handle: TVMArrayHandle): Int
  @native def tvmArrayAlloc(shape: Array[Long],
                            dtype: TVMType,
                            ctx: TVMContext,
                            refHandle: RefTVMArrayHandle): Int
  @native def tvmArrayGetShape(handle: TVMArrayHandle, shape: ArrayBuffer[Long]): Int
  @native def tvmArrayCopyFromJArray(fromRaw: Array[Byte],
                                     from: TVMArrayHandle,
                                     to: TVMArrayHandle): Int
  @native def tvmArrayCopyToJArray(from: TVMArrayHandle, to: Array[Byte]): Int

  // TVMContext
  @native def tvmSynchronize(ctx: TVMContext): Int
}
