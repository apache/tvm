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
public class TypeIndex {
  public static final int kTVMFFINone = 0;
  public static final int kTVMFFIInt = 1;
  public static final int kTVMFFIBool = 2;
  public static final int kTVMFFIFloat = 3;
  public static final int kTVMFFIOpaquePtr = 4;
  public static final int kTVMFFIDataType = 5;
  public static final int kTVMFFIDevice = 6;
  public static final int kTVMFFIDLTensorPtr = 7;
  public static final int kTVMFFIRawStr = 8;
  public static final int kTVMFFIByteArrayPtr = 9;
  public static final int kTVMFFIObjectRValueRef = 10;
  public static final int kTVMFFIStaticObjectBegin = 64;
  public static final int kTVMFFIObject = 64;
  public static final int kTVMFFIStr = 65;
  public static final int kTVMFFIBytes = 66;
  public static final int kTVMFFIError = 67;
  public static final int kTVMFFIFunction = 68;
  public static final int kTVMFFIShape = 70;
  public static final int kTVMFFITensor = 71;
  public static final int kTVMFFIArray = 72;
  public static final int kTVMFFIMap = 73;
  public static final int kTVMFFIModule = 73;
}
