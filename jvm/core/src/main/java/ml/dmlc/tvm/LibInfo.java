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

import ml.dmlc.tvm.Base.*;
import ml.dmlc.tvm.types.TVMContext;
import ml.dmlc.tvm.types.TVMType;

import java.util.List;

public class LibInfo {
  native public int nativeLibInit(String tvmLibFile);
  native public int shutdown();

  native public String tvmGetLastError();

  // Function
  native public void tvmFuncPushArgLong(long arg);
  native public void tvmFuncPushArgDouble(double arg);
  native public void tvmFuncPushArgString(String arg);
  native public void tvmFuncPushArgHandle(long arg, int argType);

  native public int tvmFuncListGlobalNames(List<String> funcNames);
  native public int tvmFuncFree(long handle);
  native public int tvmFuncGetGlobal(String name, RefLong handle);
  native public int tvmFuncCall(long handle, RefTVMValue retVal);

  // Module
  native public int tvmModFree(long handle);
  native public int tvmModGetFunction(long handle, String name,
                               int queryImports, RefLong retHandle);
  native public int tvmModImport(long mod, long dep);

  // NDArray
  native public int tvmArrayFree(long handle);
  native public int tvmArrayAlloc(long[] shape, TVMType dtype, TVMContext ctx, RefLong refHandle);
  native public int tvmArrayGetShape(long handle, List<Long> shape);
  native public int tvmArrayCopyFromTo(long from, long to);
  native public int tvmArrayCopyFromJArray(byte[] fromRaw, long from, long to);
  native public int tvmArrayCopyToJArray(long from, byte[] to);

  // TVMContext
  native public int tvmSynchronize(TVMContext ctx);
}
