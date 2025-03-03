/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/**
 * Types for C API.
 */

/** A pointer to points to the raw address space. */
export type Pointer = number;

/** A pointer offset, need to add a base address to get a valid ptr. */
export type PtrOffset = number;

// -- TVM runtime C API --
/**
 * const char *TVMGetLastError();
 */
export type FTVMGetLastError = () => Pointer;

/**
 * void TVMAPISetLastError(const char* msg);
 */
export type FTVMAPISetLastError = (msg: Pointer) => void;

/**
 * int TVMModGetFunction(TVMModuleHandle mod,
 *                       const char* func_name,
 *                       int query_imports,
 *                       TVMFunctionHandle *out);
 */
export type FTVMModGetFunction = (
  mod: Pointer, funcName: Pointer, queryImports: number, out: Pointer) => number;
/**
 * int TVMModImport(TVMModuleHandle mod,
 *                  TVMModuleHandle dep);
 */
export type FTVMModImport = (mod: Pointer, dep: Pointer) => number;

/**
 * int TVMModFree(TVMModuleHandle mod);
 */
export type FTVMModFree = (mod: Pointer) => number;

/**
 * int TVMFuncFree(TVMFunctionHandle func);
 */
export type FTVMFuncFree = (func: Pointer) => number;

/**
 * int TVMFuncCall(TVMFunctionHandle func,
 *                 TVMValue* arg_values,
 *                 int* type_codes,
 *                 int num_args,
 *                 TVMValue* ret_val,
 *                 int* ret_type_code);
 */
export type FTVMFuncCall = (
  func: Pointer, argValues: Pointer, typeCode: Pointer,
  nargs: number, retValue: Pointer, retCode: Pointer) => number;

/**
 * int TVMCFuncSetReturn(TVMRetValueHandle ret,
 *                       TVMValue* value,
 *                       int* type_code,
 *                       int num_ret);
 */
export type FTVMCFuncSetReturn = (
  ret: Pointer, value: Pointer, typeCode: Pointer, numRet: number) => number;

/**
 * int TVMCbArgToReturn(TVMValue* value, int* code);
 */
export type FTVMCbArgToReturn = (value: Pointer, code: Pointer) => number;

/**
 * int TVMFuncListGlobalNames(int* outSize, const char*** outArray);
 */
export type FTVMFuncListGlobalNames = (outSize: Pointer, outArray: Pointer) => number;

/**
 * int TVMFuncRegisterGlobal(
 *    const char* name, TVMFunctionHandle f, int override);
 */
export type FTVMFuncRegisterGlobal = (
  name: Pointer, f: Pointer, override: number) => number;

/**
 *int TVMFuncGetGlobal(const char* name, TVMFunctionHandle* out);
    */
export type FTVMFuncGetGlobal = (name: Pointer, out: Pointer) => number;

/**
 * int TVMArrayAlloc(const tvm_index_t* shape,
 *                   int ndim,
 *                   int dtype_code,
 *                   int dtype_bits,
 *                   int dtype_lanes,
 *                   int device_type,
 *                   int device_id,
 *                   TVMArrayHandle* out);
 */
export type FTVMArrayAlloc = (
  shape: Pointer, ndim: number,
  dtypeCode: number, dtypeBits: number,
  dtypeLanes: number, deviceType: number, deviceId: number,
  out: Pointer) => number;

/**
 * int TVMArrayFree(TVMArrayHandle handle);
 */
export type FTVMArrayFree = (handle: Pointer) => number;

/**
 * int TVMArrayCopyFromBytes(TVMArrayHandle handle,
 *                           void* data,
 *                           size_t nbytes);
 */
export type FTVMArrayCopyFromBytes = (
  handle: Pointer, data: Pointer, nbytes: number) => number;

/**
 * int TVMArrayCopyToBytes(TVMArrayHandle handle,
 *                         void* data,
 *                         size_t nbytes);
 */
export type FTVMArrayCopyToBytes = (
  handle: Pointer, data: Pointer, nbytes: number) => number;

/**
 * int TVMArrayCopyFromTo(TVMArrayHandle from,
 *                        TVMArrayHandle to,
 *                        TVMStreamHandle stream);
 */
export type FTVMArrayCopyFromTo = (
  from: Pointer, to: Pointer, stream: Pointer) => number;

/**
 * int TVMSynchronize(int device_type, int device_id, TVMStreamHandle stream);
 */
export type FTVMSynchronize = (
  deviceType: number, deviceId: number, stream: Pointer) => number;

/**
 * typedef int (*TVMBackendPackedCFunc)(TVMValue* args,
 *                                      int* type_codes,
 *                                      int num_args,
 *                                      TVMValue* out_ret_value,
 *                                      int* out_ret_tcode);
 */
export type FTVMBackendPackedCFunc = (
  argValues: Pointer, argCodes: Pointer, nargs: number,
  outValue: Pointer, outCode: Pointer) => number;


/**
 * int TVMObjectFree(TVMObjectHandle obj);
 */
export type FTVMObjectFree = (obj: Pointer) => number;

/**
 * int TVMObjectGetTypeIndex(TVMObjectHandle obj, unsigned* out_tindex);
 */
export type FTVMObjectGetTypeIndex = (obj: Pointer, out_tindex: Pointer) => number;

/**
 * int TVMObjectTypeIndex2Key(unsigned tindex, char** out_type_key);
 */
export type FTVMObjectTypeIndex2Key = (type_index: number, out_type_key: Pointer) => number;

/**
 * int TVMObjectTypeKey2Index(const char* type_key, unsigned* out_tindex);
 */
export type FTVMObjectTypeKey2Index = (type_key: Pointer, out_tindex: Pointer) => number;

// -- TVM Wasm Auxiliary C API --

/** void* TVMWasmAllocSpace(int size); */
export type FTVMWasmAllocSpace = (size: number) => Pointer;

/** void TVMWasmFreeSpace(void* data); */
export type FTVMWasmFreeSpace = (ptr: Pointer) => void;

/**
 * int TVMWasmPackedCFunc(TVMValue* args,
 *                        int* type_codes,
 *                        int num_args,
 *                        TVMRetValueHandle ret,
 *                        void* resource_handle);
 */
export type FTVMWasmPackedCFunc = (
  args: Pointer, typeCodes: Pointer, nargs: number,
  ret: Pointer, resourceHandle: Pointer) => number;

/**
 * int TVMWasmFuncCreateFromCFunc(void* resource_handle,
 *                                TVMFunctionHandle *out);
 */
export type FTVMWasmFuncCreateFromCFunc = (
  resource: Pointer, out: Pointer) => number;

/**
 * void TVMWasmPackedCFuncFinalizer(void* resource_handle);
 */
export type FTVMWasmPackedCFuncFinalizer = (resourceHandle: Pointer) => void;

/**
 * Size of common data types.
 */
export const enum SizeOf {
  U8 = 1,
  U16 = 2,
  I32 = 4,
  I64 = 8,
  F32 = 4,
  F64 = 8,
  TVMValue = 8,
  DLDataType = I32,
  DLDevice = I32 + I32,
}

/**
 * Argument Type code in TVM FFI.
 */
export const enum ArgTypeCode {
  Int = 0,
  UInt = 1,
  Float = 2,
  TVMOpaqueHandle = 3,
  Null = 4,
  TVMDataType = 5,
  DLDevice = 6,
  TVMDLTensorHandle = 7,
  TVMObjectHandle = 8,
  TVMModuleHandle = 9,
  TVMPackedFuncHandle = 10,
  TVMStr = 11,
  TVMBytes = 12,
  TVMNDArrayHandle = 13,
  TVMObjectRValueRefArg = 14,
  TVMArgBool = 15,
}
