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

import java.util.List;

class LibInfo {
  public native int nativeLibInit(String tvmLibFile);

  public native int shutdown();

  public native String tvmGetLastError();

  // Function
  public native void tvmFuncPushArgLong(long arg);

  public native void tvmFuncPushArgDouble(double arg);

  public native void tvmFuncPushArgString(String arg);

  public native void tvmFuncPushArgHandle(long arg, int argType);

  public native int tvmFuncListGlobalNames(List<String> funcNames);

  public native int tvmFuncFree(long handle);

  public native int tvmFuncGetGlobal(String name, Base.RefLong handle);

  public native int tvmFuncCall(long handle, Base.RefTVMValue retVal);

  // Module
  public native int tvmModFree(long handle);

  public native int tvmModGetFunction(long handle, String name,
                                      int queryImports, Base.RefLong retHandle);

  public native int tvmModImport(long mod, long dep);

  // NDArray
  public native int tvmArrayFree(long handle);

  public native int tvmArrayAlloc(long[] shape,
                                  int dtypeCode,
                                  int dtypeBits,
                                  int dtypeLanes,
                                  int deviceType,
                                  int deviceId,
                                  Base.RefLong refHandle);

  public native int tvmArrayGetShape(long handle, List<Long> shape);

  public native int tvmArrayCopyFromTo(long from, long to);

  public native int tvmArrayCopyFromJArray(byte[] fromRaw, long from, long to);

  public native int tvmArrayCopyToJArray(long from, byte[] to);

  // TVMContext
  public native int tvmSynchronize(int deviceType, int deviceId);
}
