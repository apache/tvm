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

  native String tvmFFIGetLastError();

  // Object
  native int tvmFFIObjectFree(long handle);

  // Function
  native void tvmFFIFunctionPushArgLong(long arg);

  native void tvmFFIFunctionPushArgDouble(double arg);

  native void tvmFFIFunctionPushArgString(String arg);

  native void tvmFFIFunctionPushArgBytes(byte[] arg);

  native void tvmFFIFunctionPushArgHandle(long arg, int argTypeIndex);

  native void tvmFFIFunctionPushArgDevice(Device device);

  native int tvmFFIFunctionListGlobalNames(List<String> funcNames);

  native int tvmFFIFunctionGetGlobal(String name, Base.RefLong handle);

  native int tvmFFIFunctionSetGlobal(String name, long handle, int override);

  native int tvmFFIFunctionCall(long handle, Base.RefTVMValue retVal);

  native int tvmFFIFunctionCreateFromCallback(Function.Callback function, Base.RefLong handle);

  // Tensor
  native int tvmFFIDLTensorGetShape(long handle, List<Long> shape);

  native int tvmFFIDLTensorCopyFromTo(long from, long to);

  native int tvmFFIDLTensorCopyFromJArray(byte[] fromRaw, long to);

  native int tvmFFIDLTensorCopyToJArray(long from, byte[] to);

  // the following functions are binded to keep things simpler
  // One possibility is to enhance FFI to support shape directly
  // so we do not need to run this binding through JNI
  // Device
  native int tvmSynchronize(int deviceType, int deviceId);

  native int tvmTensorEmpty(long[] shape, int dtypeCode, int dtypeBits,
                             int dtypeLanes, int deviceType, int deviceId,
                             Base.RefLong handle);
}
