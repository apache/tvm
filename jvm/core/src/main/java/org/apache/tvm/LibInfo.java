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

import java.util.List;

class LibInfo {
  native int nativeLibInit(String tvmLibFile);

  native int shutdown();

  native String tvmGetLastError();

  // Function
  native void tvmFuncPushArgLong(long arg);

  native void tvmFuncPushArgDouble(double arg);

  native void tvmFuncPushArgString(String arg);

  native void tvmFuncPushArgBytes(byte[] arg);

  native void tvmFuncPushArgHandle(long arg, int argType);

  native void tvmFuncPushArgDevice(Device device);

  native int tvmFuncListGlobalNames(List<String> funcNames);

  native int tvmFuncFree(long handle);

  native int tvmFuncGetGlobal(String name, Base.RefLong handle);

  native int tvmFuncCall(long handle, Base.RefTVMValue retVal);

  native int tvmFuncCreateFromCFunc(Function.Callback function, Base.RefLong handle);

  native int tvmFuncRegisterGlobal(String name, long handle, int override);

  // Module
  native int tvmModFree(long handle);

  native int tvmModGetFunction(long handle, String name,
                                      int queryImports, Base.RefLong retHandle);

  native int tvmModImport(long mod, long dep);

  // NDArray
  native int tvmArrayFree(long handle);

  native int tvmArrayAlloc(long[] shape, int dtypeCode, int dtypeBits, int dtypeLanes,
      int deviceType, int deviceId, Base.RefLong refHandle);

  native int tvmArrayGetShape(long handle, List<Long> shape);

  native int tvmArrayCopyFromTo(long from, long to);

  native int tvmArrayCopyFromJArray(byte[] fromRaw, long from, long to);

  native int tvmArrayCopyToJArray(long from, byte[] to);

  // Device
  native int tvmSynchronize(int deviceType, int deviceId);
}
