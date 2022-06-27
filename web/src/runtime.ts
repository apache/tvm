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
 * TVM JS Wasm Runtime library.
 */
import { Pointer, PtrOffset, SizeOf, ArgTypeCode } from "./ctypes";
import { Disposable } from "./types";
import { Memory, CachedCallStack } from "./memory";
import { assert, StringToUint8Array } from "./support";
import { Environment } from "./environment";
import { WebGPUContext } from "./webgpu";

import * as compact from "./compact";
import * as ctypes from "./ctypes";

/**
 * Type for PackedFunc inthe TVMRuntime.
 */
export type PackedFunc = ((...args: any) => any) &
  Disposable & { _tvmPackedCell: PackedFuncCell };

/**
 * @internal
 * FFI Library wrapper, maintains most runtime states.
 */
class FFILibrary implements Disposable {
  wasm32: boolean;
  memory: Memory;
  exports: Record<string, Function>;
  webGPUContext?: WebGPUContext;
  private wasmInstance: WebAssembly.Instance;
  private recycledCallStacks: Array<CachedCallStack> = [];

  constructor(
    wasmInstance: WebAssembly.Instance,
    imports: Record<string, any>
  ) {
    this.wasmInstance = wasmInstance;
    this.memory = new Memory(this.detectWasmMemory(this.wasmInstance, imports));
    assert(
      this.wasmInstance.exports !== undefined,
      "Expect the library module contains exports"
    );
    this.exports = this.wasmInstance.exports as Record<string, Function>;
    this.wasm32 = this.memory.wasm32;
    this.validateInstance();
  }

  dispose(): void {
    while (this.recycledCallStacks.length != 0) {
      (this.recycledCallStacks.pop() as Disposable).dispose();
    }
  }

  sizeofPtr(): number {
    return this.memory.sizeofPtr();
  }

  checkCall(code: number): void {
    if (code != 0) {
      const msgPtr = (this.exports
        .TVMGetLastError as ctypes.FTVMGetLastError)();
      throw new Error("TVMError: " + this.memory.loadCString(msgPtr));
    }
  }

  getOrAllocCallStack(): CachedCallStack {
    if (this.recycledCallStacks.length != 0) {
      return this.recycledCallStacks.pop() as CachedCallStack;
    }
    return new CachedCallStack(
      this.memory,
      this.exports.TVMWasmAllocSpace as ctypes.FTVMWasmAllocSpace,
      this.exports.TVMWasmFreeSpace as ctypes.FTVMWasmFreeSpace
    );
  }

  recycleCallStack(callstack: CachedCallStack): void {
    callstack.reset();
    this.recycledCallStacks.push(callstack);
  }

  private validateInstance(): void {
    this.checkExports(["TVMWasmAllocSpace", "TVMWasmFreeSpace", "TVMFuncFree"]);
  }

  private checkExports(funcNames: Array<string>): void {
    const missList = [];
    for (const name of funcNames) {
      const f = this.exports[name];
      if (!(f instanceof Function)) {
        missList.push(name);
      }
    }
    if (missList.length != 0) {
      throw new Error("Cannot find " + missList + " in exports");
    }
  }

  private detectWasmMemory(
    instance: WebAssembly.Instance,
    imports: Record<string, any>
  ): WebAssembly.Memory {
    if (instance.exports.memory instanceof WebAssembly.Memory) {
      return instance.exports.memory;
    }
    if (imports.env && imports.env.memory instanceof WebAssembly.Memory) {
      return imports.env.memory;
    }

    throw new Error(
      "Cannt detect wasm memory from imports " +
        imports +
        " or exports" +
        instance.exports
    );
  }
}

/**
 * A typed scalar constant used to represent a typed number
 * argument to PackedFunc calls.
 */
export class Scalar {
  /** The value. */
  value: number;
  /** The data type of the scalar. */
  dtype: string;

  constructor(value: number, dtype: string) {
    this.value = value;
    this.dtype = dtype;
  }
}

/**
 * Cell holds the PackedFunc object.
 */
class PackedFuncCell implements Disposable {
  handle: Pointer;
  private lib: FFILibrary;

  constructor(handle: Pointer, lib: FFILibrary) {
    this.handle = handle;
    this.lib = lib;
  }

  dispose(): void {
    if (this.handle != 0) {
      this.lib.checkCall(
        (this.lib.exports.TVMFuncFree as ctypes.FTVMFuncFree)(this.handle)
      );
      this.handle = 0;
    }
  }
}

const DeviceEnumToStr: Record<number, string> = {
  1: "cpu",
  2: "cuda",
  4: "opencl",
  8: "metal",
  15: "webgpu"
};

const DeviceStrToEnum: Record<string, number> = {
  cpu: 1,
  cuda: 2,
  cl: 4,
  opencl: 4,
  vulkan: 7,
  metal: 8,
  webgpu: 15
};

/**
 * Represent a runtime context where a NDArray can reside.
 */
export class DLDevice {
  /** The device type code of the device. */
  deviceType: number;
  /** The device index. */
  deviceId: number;

  private lib: FFILibrary;

  constructor(deviceType: number | string, deviceId: number, lib: FFILibrary) {
    const tp = typeof deviceType;
    if (tp == "string") {
      this.deviceType = DeviceStrToEnum[deviceType];
      if (this.deviceType == undefined) {
        throw new Error("Cannot recogonize deviceType " + deviceType);
      }
    } else if (tp == "number") {
      this.deviceType = deviceType as number;
    } else {
      throw new Error("Cannot take type " + tp + " as deviceType");
    }
    this.deviceId = deviceId;
    this.lib = lib;
  }

  /**
   * Synchronize the device
   */
  async sync(): Promise<void> {
    if (this.deviceType == DeviceStrToEnum.webgpu) {
      assert(this.lib.webGPUContext !== undefined);
      await this.lib.webGPUContext.sync();
    }
  }

  toString(): string {
    return (
      DeviceEnumToStr[this.deviceType] + "(" + this.deviceId.toString() + ")"
    );
  }
}
/**
 * The data type code in DLDataType
 */
export const enum DLDataTypeCode {
  Int = 0,
  UInt = 1,
  Float = 2,
  OpaqueHandle = 3
}

const DLDataTypeCodeToStr: Record<number, string> = {
  0: "int",
  1: "uint",
  2: "float",
  3: "handle",
};

/**
 * Runtime data type of NDArray.
 */
export class DLDataType {
  /** The type code */
  code: number;
  /** Number of bits in the data type. */
  bits: number;
  /** Number of vector lanes. */
  lanes: number;

  constructor(code: number, bits: number, lanes: number) {
    this.code = code;
    this.bits = bits;
    this.lanes = lanes;
  }

  toString(): string {
    const ret = DLDataTypeCodeToStr[this.code] + this.bits.toString();
    if (this.lanes != 1) {
      return ret + "x" + this.lanes.toString();
    } else {
      return ret;
    }
  }

  numStorageBytes(): number {
    return (this.bits * this.lanes + 7) >> 3;
  }
}

/**
 * n-dimnesional array.
 */
export class NDArray implements Disposable {
  /** Internal array handle. */
  handle: Pointer;
  /** Number of dimensions. */
  ndim: number;
  /** Data type of the array. */
  dtype: string;
  /** Shape of the array. */
  shape: Array<number>;
  /** Device of the array. */
  device: DLDevice;
  /** Whether it is a temporary view that can become invalid after the call. */
  private isView: boolean;
  private byteOffset: number;
  private dltensor: Pointer;
  private dataPtr: Pointer;
  private lib: FFILibrary;
  private dlDataType: DLDataType;

  constructor(handle: Pointer, isView: boolean, lib: FFILibrary) {
    this.handle = handle;
    this.isView = isView;
    this.lib = lib;

    if (this.isView) {
      this.dltensor = handle;
    } else {
      this.dltensor = this.getDLTensorFromArrayHandle(this.handle);
    }
    // constant offsets.
    const arrayOffsetData = 0;
    const arrayOffsetContext = arrayOffsetData + this.lib.sizeofPtr();
    const arrayOffsetDevType = arrayOffsetContext;
    const arrayOffsetDevId = arrayOffsetContext + SizeOf.I32;
    const arrayOffsetNdim = arrayOffsetContext + SizeOf.DLDevice;
    const arrayOffsetDtype = arrayOffsetNdim + SizeOf.I32;
    const arrayOffsetDtypeCode = arrayOffsetDtype;
    const arrayOffsetDtypeBits = arrayOffsetDtype + SizeOf.U8;
    const arrayOffsetDtypeLanes = arrayOffsetDtypeBits + SizeOf.U8;
    const arrayOffsetShape = arrayOffsetDtype + SizeOf.DLDataType;
    const arrayOffsetStrides = arrayOffsetShape + this.lib.sizeofPtr();
    const arrayOffsetByteOffset = arrayOffsetStrides + this.lib.sizeofPtr();
    // dataPtr
    this.dataPtr = lib.memory.loadPointer(this.dltensor);
    // ndim
    this.ndim = lib.memory.loadI32(this.dltensor + arrayOffsetNdim);
    // shape
    const cshapePtr = lib.memory.loadPointer(this.dltensor + arrayOffsetShape);
    this.shape = [];
    for (let i = 0; i < this.ndim; ++i) {
      this.shape.push(lib.memory.loadI64(cshapePtr + i * SizeOf.I64));
    }
    // dtype
    const code = lib.memory.loadU8(this.dltensor + arrayOffsetDtypeCode);
    const bits = lib.memory.loadU8(this.dltensor + arrayOffsetDtypeBits);
    const lanes = lib.memory.loadU16(this.dltensor + arrayOffsetDtypeLanes);
    this.dlDataType = new DLDataType(code, bits, lanes);
    this.dtype = this.dlDataType.toString();

    // device
    const deviceType = lib.memory.loadI32(this.dltensor + arrayOffsetDevType);
    const deviceId = lib.memory.loadI32(this.dltensor + arrayOffsetDevId);
    this.device = new DLDevice(deviceType, deviceId, lib);

    // byte_offset
    this.byteOffset = lib.memory.loadI64(this.dltensor + arrayOffsetByteOffset);
  }

  dispose(): void {
    if (this.handle != 0 && !this.isView) {
      this.lib.checkCall(
        (this.lib.exports.TVMArrayFree as ctypes.FTVMArrayFree)(this.handle)
      );
      this.handle = 0;
    }
  }
  /**
   * Copy data from another NDArray or javascript array.
   * The number of elements must match.
   *
   * @param data The source data array.
   * @returns this
   */
  copyFrom(data: NDArray | Array<number> | Float32Array): this {
    if (data instanceof NDArray) {
      this.lib.checkCall(
        (this.lib.exports.TVMArrayCopyFromTo as ctypes.FTVMArrayCopyFromTo)(
          data.handle,
          this.handle,
          0
        )
      );
      return this;
    } else {
      const size = this.shape.reduce((a, b) => {
        return a * b;
      }, 1);
      if (data.length != size) {
        throw new Error(
          "data size and shape mismatch data.length" +
            data.length +
            " vs " +
            size
        );
      }
      let buffer: ArrayBuffer;
      if (this.dtype == "float32") {
        buffer = Float32Array.from(data).buffer;
      } else if (this.dtype == "float64") {
        buffer = Float64Array.from(data).buffer;
      } else if (this.dtype == "int32") {
        buffer = Int32Array.from(data).buffer;
      } else if (this.dtype == "int8") {
        buffer = Int8Array.from(data).buffer;
      } else if (this.dtype == "uint8") {
        buffer = Uint8Array.from(data).buffer;
      } else {
        throw new Error("Unsupported data type " + this.dtype);
      }
      return this.copyFromRawBytes(new Uint8Array(buffer));
    }
  }
  /**
   * Copy data from raw bytes.
   * @param data Uint8Array of bytes.
   * @returns this
   */
  copyFromRawBytes(data: Uint8Array): this {
    const size = this.shape.reduce((a, b) => {
      return a * b;
    }, 1);
    const nbytes = this.dlDataType.numStorageBytes() * size;
    if (nbytes != data.length) {
      throw new Error("Expect the data's length equals nbytes=" + nbytes);
    }

    const stack = this.lib.getOrAllocCallStack();

    const tempOffset = stack.allocRawBytes(nbytes);
    const tempPtr = stack.ptrFromOffset(tempOffset);
    this.lib.memory.storeRawBytes(tempPtr, data);
    this.lib.checkCall(
      (this.lib.exports.TVMArrayCopyFromBytes as ctypes.FTVMArrayCopyFromBytes)(
        this.handle,
        tempPtr,
        nbytes
      )
    );

    this.lib.recycleCallStack(stack);
    return this;
  }
  /**
   * Return a copied Uint8Array of the raw bytes in the NDArray.
   * @returns The result array.
   */
  toRawBytes(): Uint8Array {
    if (this.device.deviceType != DeviceStrToEnum.cpu) {
      throw new Error("Can only synchronize copy for GPU array, use copyfrom instead.");
    }
    const size = this.shape.reduce((a, b) => {
      return a * b;
    }, 1);

    const nbytes = this.dlDataType.numStorageBytes() * size;
    const stack = this.lib.getOrAllocCallStack();

    const tempOffset = stack.allocRawBytes(nbytes);
    const tempPtr = stack.ptrFromOffset(tempOffset);
    this.lib.checkCall(
      (this.lib.exports.TVMArrayCopyToBytes as ctypes.FTVMArrayCopyToBytes)(
        this.handle,
        tempPtr,
        nbytes
      )
    );
    const ret = this.lib.memory.loadRawBytes(tempPtr, nbytes);

    this.lib.recycleCallStack(stack);
    return ret;
  }

  /**
   * Return a TypedArray copy of the NDArray, the specific type depends on
   * the dtype of the NDArray.
   * @returns The result array.
   */
  toArray(): Float32Array | Float64Array | Int32Array | Int8Array | Uint8Array {
    const stype = this.dtype;
    if (stype == "float32") {
      return new Float32Array(this.toRawBytes().buffer);
    } else if (stype == "float64") {
      return new Float64Array(this.toRawBytes().buffer);
    } else if (stype == "int32") {
      return new Int32Array(this.toRawBytes().buffer);
    } else if (stype == "int8") {
      return new Int8Array(this.toRawBytes().buffer);
    } else if (stype == "uint8") {
      return new Uint8Array(this.toRawBytes().buffer);
    } else {
      throw new Error("Unsupported data type " + this.dtype);
    }
  }

  private getDLTensorFromArrayHandle(handle: Pointer): Pointer {
    // Note: this depends on the NDArray C ABI.
    // keep this function in case of ABI change.
    return handle;
  }
}

/**
 * Runtime Module.
 */
export class Module implements Disposable {
  handle: Pointer;
  private lib: FFILibrary;
  private makePackedFunc: (ptr: Pointer) => PackedFunc;

  constructor(
    handle: Pointer,
    lib: FFILibrary,
    makePackedFunc: (ptr: Pointer) => PackedFunc
  ) {
    this.handle = handle;
    this.lib = lib;
    this.makePackedFunc = makePackedFunc;
  }

  dispose(): void {
    if (this.handle != 0) {
      this.lib.checkCall(
        (this.lib.exports.TVMModFree as ctypes.FTVMModFree)(this.handle)
      );
      this.handle = 0;
    }
  }

  /**
   * Get a function in the module.
   * @param name The name of the function.
   * @returns The result function.
   */
  getFunction(name: string): PackedFunc {
    const stack = this.lib.getOrAllocCallStack();
    const nameOffset = stack.allocRawBytes(name.length + 1);
    stack.storeRawBytes(nameOffset, StringToUint8Array(name));

    const outOffset = stack.allocPtrArray(1);
    const outPtr = stack.ptrFromOffset(outOffset);

    stack.commitToWasmMemory(outOffset);

    this.lib.checkCall(
      (this.lib.exports.TVMModGetFunction as ctypes.FTVMModGetFunction)(
        this.handle,
        stack.ptrFromOffset(nameOffset),
        1,
        outPtr
      )
    );
    const handle = this.lib.memory.loadPointer(outPtr);
    this.lib.recycleCallStack(stack);
    if (handle == 0) {
      throw Error("Cannot find function " + name);
    }
    const ret = this.makePackedFunc(handle);
    return ret;
  }

  /**
   * Import another module into the current runtime module.
   * @param mod The module to be imported.
   */
  importModule(mod: Module): void {
    this.lib.checkCall(
      (this.lib.exports.TVMModImport as ctypes.FTVMModImport)(
        this.handle,
        mod.handle
      )
    );
  }
}

/**
 *  Graph executor.
 *
 *  This is a thin wrapper of the underlying TVM module.
 *  you can also directly call set_input, run, and get_output
 *  of underlying module functions
 */
class GraphExecutor implements Disposable {
  module: Module;
  private packedSetInput: PackedFunc;
  private packedRun: PackedFunc;
  private packedGetOutput: PackedFunc;
  private packedLoadParams: PackedFunc;

  /**
   * COnstructor
   * @param module The underlying module.
   */
  constructor(module: Module) {
    this.module = module;
    this.packedSetInput = module.getFunction("set_input");
    this.packedRun = module.getFunction("run");
    this.packedGetOutput = module.getFunction("get_output");
    this.packedLoadParams = module.getFunction("load_params");
  }

  dispose(): void {
    this.packedSetInput.dispose();
    this.packedRun.dispose();
    this.packedGetOutput.dispose();
  }

  /**
   * Set input to the executor.
   *
   * @param key The input key.
   * @param value The value to get set.
   */
  setInput(key: number | string, value: NDArray): void {
    if (typeof key == "number") {
      this.packedSetInput(new Scalar(key, "int32"), value);
    } else {
      this.packedSetInput(key, value);

    }
  }

  /**
   * Execute the underlying graph.
   */
  run(): void {
    this.packedRun();
  }

  /**
   * Get index-th output.
   * @param index The index number.
   * @param out The optional output storage parameters.
   * @returns The output array.
   */
  getOutput(index: number, out: NDArray | undefined = undefined): NDArray {
    if (out !== undefined) {
      this.packedGetOutput(new Scalar(index, "int32"), out)
      return out;
    } else {
      return this.packedGetOutput(new Scalar(index, "int32"));
    }
  }

  /**
   * Load parameters from parameter binary.
   * @param paramBinary The parameter binary.
   */
  loadParams(paramBinary: Uint8Array): void {
    this.packedLoadParams(paramBinary);
  }

  /**
   * Benchmark stable execution of the graph(without data copy).
   * @params dev The device to sync during each run.
   * @number The number of times to compute the average.
   * @repeat The number of times to repeat the run.
   */
  async benchmarkRuns(dev: DLDevice, number=10, repeat=4): Promise<number[]> {
    // Skip first run as it can involve GPU warmup and module loading time.
    const perf = compact.getPerformance();
    const results = [];
    this.run();
    await dev.sync();
    for (let k = 0; k < repeat; ++k) {
      const tstart = perf.now();
      for (let i = 0; i < number; ++i) {
        this.run();
      }
      await dev.sync();
      const tend = perf.now();
      results.push((tend - tstart) / number);
    }
    return results;
  }
}

/** Code used as the first argument of the async callback. */
const enum AyncCallbackCode {
  kReturn = 4,
  kException = 5,
}

/**
 * TVM runtime instance.
 */
export class Instance implements Disposable {
  memory: Memory;
  exports: Record<string, Function>;
  private lib: FFILibrary;
  private env: Environment;

  /**
   * Internal function(registered by the runtime)
   */
  private wasmCreateLibraryModule?: PackedFunc &
    ((getFunc: PackedFunc, getGlobal: PackedFunc) => PackedFunc);

  /**
   * Constructor
   *
   * importObject can also be a {@link LibraryProvider} object,
   * a WASI object, or an object containing wasmLibraryProvider field.
   *
   * @param wasmModule The input module or instance.
   * @param importObject The imports to initialize the wasmInstance if it is not provided.
   * @param wasmInstance Additional wasm instance argument for deferred construction.
   * @param env Directly specified environment module.
   *
   * @see Please use the async version {@link instantiate} when targeting browsers.
   */
  constructor(
    wasmModule: WebAssembly.Module,
    importObject: Record<string, any> = {},
    wasmInstance?: WebAssembly.Instance,
    env?: Environment
  ) {
    if (wasmInstance instanceof WebAssembly.Instance) {
      assert(
        env instanceof Environment,
        "env must be provided when passing in instance"
      );
    } else {
      assert(env === undefined);
      env = new Environment(importObject);
      wasmInstance = new WebAssembly.Instance(wasmModule, env.imports);
    }

    env.start(wasmInstance);
    this.env = env;
    this.lib = new FFILibrary(wasmInstance, env.imports);
    this.memory = this.lib.memory;
    this.exports = this.lib.exports;
    this.registerEnvGlobalPackedFuncs();
  }

  dispose(): void {
    this.lib.dispose();
  }
  /**
   * Get system-wide library module in the wasm.
   * System lib is a global module that contains self register functions in startup.
   * @returns The system library module.
   */
  systemLib(): Module {
    const getSysLib = this.getGlobalFunc("runtime.SystemLib");
    const mod = getSysLib() as Module;
    getSysLib.dispose();
    return mod;
  }
  /**
   * List all the global function names registered in the runtime.
   * @returns The name list.
   */
  listGlobalFuncNames(): Array<string> {
    const stack = this.lib.getOrAllocCallStack();

    const outSizeOffset = stack.allocPtrArray(2);

    const outSizePtr = stack.ptrFromOffset(outSizeOffset);
    const outArrayPtr = stack.ptrFromOffset(
      outSizeOffset + this.lib.sizeofPtr()
    );

    this.lib.checkCall(
      (this.exports.TVMFuncListGlobalNames as ctypes.FTVMFuncListGlobalNames)(
        outSizePtr,
        outArrayPtr
      )
    );

    const size = this.memory.loadI32(outSizePtr);
    const array = this.memory.loadPointer(outArrayPtr);
    const names: Array<string> = [];

    for (let i = 0; i < size; ++i) {
      names.push(
        this.memory.loadCString(
          this.memory.loadPointer(array + this.lib.sizeofPtr() * i)
        )
      );
    }

    this.lib.recycleCallStack(stack);
    return names;
  }

  /**
   * Register function to be global function in tvm runtime.
   * @param name The name of the function.
   * @param f function to be registered.
   * @param override Whether overwrite function in existing registry.
   */
  registerFunc(
    name: string,
    func: PackedFunc | Function,
    override = false
  ): void {
    const packedFunc = this.toPackedFunc(func);
    const ioverride = override ? 1 : 0;

    const stack = this.lib.getOrAllocCallStack();
    const nameOffset = stack.allocRawBytes(name.length + 1);
    stack.storeRawBytes(nameOffset, StringToUint8Array(name));
    stack.commitToWasmMemory();

    this.lib.checkCall(
      (this.lib.exports.TVMFuncRegisterGlobal as ctypes.FTVMFuncRegisterGlobal)(
        stack.ptrFromOffset(nameOffset),
        packedFunc._tvmPackedCell.handle,
        ioverride
      )
    );
  }

  /**
   * Get global PackedFunc from the runtime.
   * @param name The name of the function.
   * @returns The result function.
   */
  getGlobalFunc(name: string): PackedFunc {
    const stack = this.lib.getOrAllocCallStack();
    const nameOffset = stack.allocRawBytes(name.length + 1);
    stack.storeRawBytes(nameOffset, StringToUint8Array(name));
    const outOffset = stack.allocPtrArray(1);
    const outPtr = stack.ptrFromOffset(outOffset);

    stack.commitToWasmMemory(outOffset);

    this.lib.checkCall(
      (this.exports.TVMFuncGetGlobal as ctypes.FTVMFuncGetGlobal)(
        stack.ptrFromOffset(nameOffset),
        outPtr
      )
    );
    const handle = this.memory.loadPointer(outPtr);
    this.lib.recycleCallStack(stack);
    if (handle == 0) {
      throw Error("Cannot find global function " + name);
    }
    const ret = this.makePackedFunc(handle);
    return ret;
  }

  /**
   * Check if func is PackedFunc.
   *
   * @param func The input.
   * @returns The check result.
   */
  isPackedFunc(func: unknown): boolean {
    // eslint-disable-next-line no-prototype-builtins
    return typeof func == "function" && func.hasOwnProperty("_tvmPackedCell");
  }

  /**
   * Convert func to PackedFunc
   *
   * @param func Input function.
   * @returns The converted function.
   */
  toPackedFunc(func: Function): PackedFunc {
    if (this.isPackedFunc(func)) return func as PackedFunc;
    return this.createPackedFuncFromCFunc(this.wrapJSFuncAsPackedCFunc(func));
  }

  /**
   * Convert dtype to {@link DLDataType}
   *
   * @param dtype The input dtype string or DLDataType.
   * @returns The converted result.
   */
  toDLDataType(dtype: string | DLDataType): DLDataType {
    if (dtype instanceof DLDataType) return dtype;
    if (typeof dtype == "string") {
      let pattern = dtype;
      let code,
        bits = 32,
        lanes = 1;
      if (pattern.substring(0, 5) == "float") {
        pattern = pattern.substring(5, pattern.length);
        code = DLDataTypeCode.Float;
      } else if (pattern.substring(0, 3) == "int") {
        pattern = pattern.substring(3, pattern.length);
        code = DLDataTypeCode.Int;
      } else if (pattern.substring(0, 4) == "uint") {
        pattern = pattern.substring(4, pattern.length);
        code = DLDataTypeCode.UInt;
      } else if (pattern.substring(0, 6) == "handle") {
        pattern = pattern.substring(5, pattern.length);
        code = DLDataTypeCode.OpaqueHandle;
        bits = 64;
      } else {
        throw new Error("Unknown dtype " + dtype);
      }

      const arr = pattern.split("x");
      if (arr.length >= 1) {
        const parsed = parseInt(arr[0]);
        if (parsed + "" == arr[0]) {
          bits = parsed;
        }
      }
      if (arr.length >= 2) {
        lanes = parseInt(arr[1]);
      }
      return new DLDataType(code, bits, lanes);
    } else {
      throw new Error("Unknown dtype " + dtype);
    }
  }

  /**
   * Create a new {@link Scalar} that can be passed to a PackedFunc.
   * @param value The number value.
   * @param dtype The dtype string.
   * @returns The created scalar.
   */
  scalar(value: number, dtype: string): Scalar {
    return new Scalar(value, dtype);
  }

  /**
   * Create a new {@link DLDevice}
   * @param deviceType The device type.
   * @param deviceId The device index.
   * @returns The created device.
   */
  device(deviceType: number | string, deviceId = 0): DLDevice {
    return new DLDevice(deviceType, deviceId, this.lib);
  }

  /**
   * Create a new cpu {@link DLDevice}
   * @param deviceId The device index.
   */
  cpu(deviceId = 0): DLDevice {
    return this.device("cpu", deviceId);
  }

  /**
   * Create a new webgpu {@link DLDevice}
   * @param deviceId The device index.
   */
  webgpu(deviceId = 0): DLDevice {
    return this.device("webgpu", deviceId);
  }

  /**
   * Create an empty {@link NDArray} with given shape and dtype.
   *
   * @param shape The shape of the array.
   * @param dtype The data type of the array.
   * @param dev The device of the ndarray.
   * @returns The created ndarray.
   */
  empty(
    shape: Array<number> | number,
    dtype: string | DLDataType = "float32",
    dev: DLDevice = this.device("cpu", 0)
  ): NDArray {
    dtype = this.toDLDataType(dtype);
    shape = typeof shape == "number" ? [shape] : shape;

    const stack = this.lib.getOrAllocCallStack();
    const shapeOffset = stack.allocRawBytes(shape.length * SizeOf.I64);
    for (let i = 0; i < shape.length; ++i) {
      stack.storeI64(shapeOffset + i * SizeOf.I64, shape[i]);
    }

    const outOffset = stack.allocPtrArray(1);
    const outPtr = stack.ptrFromOffset(outOffset);
    stack.commitToWasmMemory(outOffset);

    this.lib.checkCall(
      (this.exports.TVMArrayAlloc as ctypes.FTVMArrayAlloc)(
        stack.ptrFromOffset(shapeOffset),
        shape.length,
        dtype.code,
        dtype.bits,
        dtype.lanes,
        dev.deviceType,
        dev.deviceId,
        outPtr
      )
    );
    const ret = new NDArray(this.memory.loadPointer(outPtr), false, this.lib);
    this.lib.recycleCallStack(stack);
    return ret;
  }

  /**
   * Create a new graph executor.
   *
   * @param graphJson The graph executor json file.
   * @param lib The underlying library.
   * @param dev The execution device of the graph.
   */
  createGraphExecutor(graphJson: string, lib: Module, dev: DLDevice): GraphExecutor {
    const fcreate = this.getGlobalFunc('tvm.graph_executor.create');
    const module = fcreate(
      graphJson,
      lib,
      this.scalar(dev.deviceType, "int32"),
      this.scalar(dev.deviceId, "int32")) as Module;
    return new GraphExecutor(module);
  }


  /**
   * Register an asyncfunction to be global function in the server.
   * @param name The name of the function.
   * @param func function to be registered.
   * @param override Whether overwrite function in existing registry.
   *
   * @note The async function will only be used for serving remote calls in the rpc.
   */
  registerAsyncServerFunc(
    name: string,
    func: Function,
    override = false
  ): void {
    const asyncVariant = (...args: Array<any>): void => {
      const fargs = args.slice(0, args.length - 1);
      const callback = args[args.length - 1] as PackedFunc;
      const promise: Promise<any> = func(...fargs);
      promise.then((rv: any) => {
        callback(this.scalar(AyncCallbackCode.kReturn, "int32"), rv);
      });
    };
    this.registerFunc("__async." + name, asyncVariant, override);
  }

  /**
   * Initialize webgpu in the runtime.
   * @param device The given GPU device.
   */
  initWebGPU(device: GPUDevice): void {
    const webGPUContext = new WebGPUContext(
      this.memory, device
    );
    this.registerFunc("wasm.WebGPUDeviceAPI", (name: string) => {
      return webGPUContext.getDeviceAPI(name);
    });
    this.registerFunc("wasm.WebGPUCreateShader", (info: string, data: Uint8Array) => {
      return webGPUContext.createShader(info, data);
    });
    this.registerAsyncServerFunc("wasm.WebGPUWaitForTasks", async () => {
      await webGPUContext.sync();
    });
    this.lib.webGPUContext = webGPUContext;
  }

  /** Register global packed functions needed by the backend to the env. */
  private registerEnvGlobalPackedFuncs(): void {
    // Register the timer function to enable the time_evaluator.
    const perf = compact.getPerformance();

    // Helper function to time the finvoke
    const timeExecution = async (
      finvoke: PackedFunc,
      dev: DLDevice,
      nstep: number,
      repeat: number,
      minRepeatMs: number,
      cooldownIntervalMs: number,
      repeatsToCooldown: number
    ): Promise<Uint8Array> => {
      finvoke(this.scalar(1, "int32"));
      await dev.sync();
      const result = [];
      let setupNumber: number = nstep;

      for (let i = 0; i < repeat; ++i) {
        let durationMs = 0.0;
        do {
          if (durationMs > 0.0) {
            let golden_ratio = 1.618;
            setupNumber = Math.floor(
              Math.max(minRepeatMs / (durationMs / setupNumber) + 1, setupNumber * golden_ratio)
            );
          }
          const tstart: number = perf.now();
          finvoke(this.scalar(setupNumber, "int32"));
          await dev.sync();
          const tend: number = perf.now();

          durationMs = tend - tstart;
        } while (durationMs < minRepeatMs);
        const speed = durationMs / setupNumber / 1000;
        result.push(speed);
        if (cooldownIntervalMs > 0.0 && (i % repeatsToCooldown) == 0 ) {
          await new Promise(r => setTimeout(r, cooldownIntervalMs));
        }
      }
      const ret = new Float64Array(result.length);
      ret.set(result);
      return new Uint8Array(ret.buffer);
    };

    const addOne = async (x: number): Promise<number> => {
      await new Promise(resolve => setTimeout(resolve, 100));
      return x + 1;
    };

    this.registerAsyncServerFunc("wasm.TimeExecution", timeExecution);
    this.registerAsyncServerFunc("testing.asyncAddOne", addOne);
  }

  private createPackedFuncFromCFunc(
    func: ctypes.FTVMWasmPackedCFunc
  ): PackedFunc {
    let findex = this.env.packedCFuncTable.length;
    if (this.env.packedCFuncTableFreeId.length != 0) {
      findex = this.env.packedCFuncTableFreeId.pop() as number;
    } else {
      this.env.packedCFuncTable.push(undefined);
    }
    this.env.packedCFuncTable[findex] = func;

    const stack = this.lib.getOrAllocCallStack();
    const outOffset = stack.allocPtrArray(1);
    const outPtr = stack.ptrFromOffset(outOffset);
    this.lib.checkCall(
      (this.exports
        .TVMWasmFuncCreateFromCFunc as ctypes.FTVMWasmFuncCreateFromCFunc)(
        findex,
        outPtr
      )
    );
    const ret = this.makePackedFunc(this.memory.loadPointer(outPtr));
    this.lib.recycleCallStack(stack);
    return ret;
  }

  /**
   * Set packed function arguments into the location indicated by argsValue and argsCode.
   * Allocate new temporary space from the stack if necessary.
   *
   * @parma stack The call stack
   * @param args  The input arguments.
   * @param argsValue The offset of argsValue.
   * @param argsCode The offset of argsCode.
   */
  setPackedArguments(
    stack: CachedCallStack,
    args: Array<any>,
    argsValue: PtrOffset,
    argsCode: PtrOffset
  ): void {
    for (let i = 0; i < args.length; ++i) {
      let val = args[i];
      const tp = typeof val;
      const valueOffset = argsValue + i * SizeOf.TVMValue;
      const codeOffset = argsCode + i * SizeOf.I32;
      if (val instanceof NDArray) {
        stack.storePtr(valueOffset, val.handle);
        stack.storeI32(codeOffset, ArgTypeCode.TVMNDArrayHandle);
      } else if (val instanceof Scalar) {
        if (val.dtype.startsWith("int") || val.dtype.startsWith("uint")) {
          stack.storeI64(valueOffset, val.value);
          stack.storeI32(codeOffset, ArgTypeCode.Int);
        } else if (val.dtype.startsWith("float")) {
          stack.storeF64(valueOffset, val.value);
          stack.storeI32(codeOffset, ArgTypeCode.Float);
        } else {
          assert(val.dtype == "handle", "Expect handle");
          stack.storePtr(valueOffset, val.value);
          stack.storeI32(codeOffset, ArgTypeCode.TVMOpaqueHandle);
        }
      } else if (val instanceof DLDevice) {
        stack.storeI32(valueOffset, val.deviceType);
        stack.storeI32(valueOffset + SizeOf.I32, val.deviceType);
        stack.storeI32(codeOffset, ArgTypeCode.DLDevice);
      } else if (tp == "number") {
        stack.storeF64(valueOffset, val);
        stack.storeI32(codeOffset, ArgTypeCode.Float);
        // eslint-disable-next-line no-prototype-builtins
      } else if (tp == "function" && val.hasOwnProperty("_tvmPackedCell")) {
        stack.storePtr(valueOffset, val._tvmPackedCell.handle);
        stack.storeI32(codeOffset, ArgTypeCode.TVMPackedFuncHandle);
      } else if (val === null || val == undefined) {
        stack.storePtr(valueOffset, 0);
        stack.storeI32(codeOffset, ArgTypeCode.Null);
      } else if (tp == "string") {
        stack.allocThenSetArgString(valueOffset, val);
        stack.storeI32(codeOffset, ArgTypeCode.TVMStr);
      } else if (val instanceof Uint8Array) {
        stack.allocThenSetArgBytes(valueOffset, val);
        stack.storeI32(codeOffset, ArgTypeCode.TVMBytes);
      } else if (val instanceof Function) {
        val = this.toPackedFunc(val);
        stack.tempArgs.push(val);
        stack.storePtr(valueOffset, val._tvmPackedCell.handle);
        stack.storeI32(codeOffset, ArgTypeCode.TVMPackedFuncHandle);
      } else if (val instanceof Module) {
        stack.storePtr(valueOffset, val.handle);
        stack.storeI32(codeOffset, ArgTypeCode.TVMModuleHandle);
      } else {
        throw new Error("Unsupported argument type " + tp);
      }
    }
  }

  private wrapJSFuncAsPackedCFunc(func: Function): ctypes.FTVMWasmPackedCFunc {
    const lib = this.lib;
    return (
      argValues: Pointer,
      argCodes: Pointer,
      nargs: number,
      ret: Pointer,
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      _handle: Pointer
    ): number => {
      const jsArgs = [];
      for (let i = 0; i < nargs; ++i) {
        const valuePtr = argValues + i * SizeOf.TVMValue;
        const codePtr = argCodes + i * SizeOf.I32;
        let tcode = lib.memory.loadI32(codePtr);

        if (
          tcode == ArgTypeCode.TVMObjectHandle ||
          tcode == ArgTypeCode.TVMObjectRValueRefArg ||
          tcode == ArgTypeCode.TVMPackedFuncHandle ||
          tcode == ArgTypeCode.TVMNDArrayHandle ||
          tcode == ArgTypeCode.TVMModuleHandle
        ) {
          lib.checkCall(
            (lib.exports.TVMCbArgToReturn as ctypes.FTVMCbArgToReturn)(
              valuePtr,
              codePtr
            )
          );
        }
        tcode = lib.memory.loadI32(codePtr);
        jsArgs.push(this.retValueToJS(valuePtr, tcode, true));
      }

      const rv = func(...jsArgs);

      if (rv !== undefined && rv !== null) {
        const stack = lib.getOrAllocCallStack();
        const valueOffset = stack.allocRawBytes(SizeOf.TVMValue);
        const codeOffset = stack.allocRawBytes(SizeOf.I32);
        this.setPackedArguments(stack, [rv], valueOffset, codeOffset);
        const valuePtr = stack.ptrFromOffset(valueOffset);
        const codePtr = stack.ptrFromOffset(codeOffset);
        stack.commitToWasmMemory();
        lib.checkCall(
          (lib.exports.TVMCFuncSetReturn as ctypes.FTVMCFuncSetReturn)(
            ret,
            valuePtr,
            codePtr,
            1
          )
        );
        lib.recycleCallStack(stack);
      }
      return 0;
    };
  }

  private makePackedFunc(handle: Pointer): PackedFunc {
    const cell = new PackedFuncCell(handle, this.lib);

    const packedFunc = (...args: any): any => {
      const stack = this.lib.getOrAllocCallStack();

      const valueOffset = stack.allocRawBytes(SizeOf.TVMValue * args.length);
      const tcodeOffset = stack.allocRawBytes(SizeOf.I32 * args.length);

      this.setPackedArguments(stack, args, valueOffset, tcodeOffset);

      const rvalueOffset = stack.allocRawBytes(SizeOf.TVMValue);
      const rcodeOffset = stack.allocRawBytes(SizeOf.I32);
      const rvaluePtr = stack.ptrFromOffset(rvalueOffset);
      const rcodePtr = stack.ptrFromOffset(rcodeOffset);

      // commit to wasm memory, till rvalueOffset (the return value don't need to be committed)
      stack.commitToWasmMemory(rvalueOffset);

      this.lib.checkCall(
        (this.exports.TVMFuncCall as ctypes.FTVMFuncCall)(
          handle,
          stack.ptrFromOffset(valueOffset),
          stack.ptrFromOffset(tcodeOffset),
          args.length,
          rvaluePtr,
          rcodePtr
        )
      );

      const ret = this.retValueToJS(rvaluePtr, this.memory.loadI32(rcodePtr), false);
      this.lib.recycleCallStack(stack);
      return ret;
    };
    // Attach attributes to the function type.
    // This is because javascript do not allow us to overload call.
    const ret: any = packedFunc;
    ret.dispose = (): void => {
      cell.dispose();
    };
    ret._tvmPackedCell = cell;
    return ret as PackedFunc;
  }

  private retValueToJS(rvaluePtr: Pointer, tcode: number, callbackArg: boolean): any {
    switch (tcode) {
      case ArgTypeCode.Int:
      case ArgTypeCode.UInt:
        return this.memory.loadI64(rvaluePtr);
      case ArgTypeCode.Float:
        return this.memory.loadF64(rvaluePtr);
        case ArgTypeCode.TVMOpaqueHandle: {
          return this.memory.loadPointer(rvaluePtr);
        }
      case ArgTypeCode.TVMNDArrayHandle: {
        return new NDArray(this.memory.loadPointer(rvaluePtr), false, this.lib);
      }
      case ArgTypeCode.TVMDLTensorHandle: {
        assert(callbackArg);
        return new NDArray(this.memory.loadPointer(rvaluePtr), true, this.lib);
      }
      case ArgTypeCode.TVMPackedFuncHandle: {
        return this.makePackedFunc(this.memory.loadPointer(rvaluePtr));
      }
      case ArgTypeCode.TVMModuleHandle: {
        return new Module(
          this.memory.loadPointer(rvaluePtr),
          this.lib,
          (ptr: Pointer) => {
            return this.makePackedFunc(ptr);
          }
        );
      }
      case ArgTypeCode.Null: return undefined;
      case ArgTypeCode.DLDevice: {
        const deviceType = this.memory.loadI32(rvaluePtr);
        const deviceId = this.memory.loadI32(rvaluePtr + SizeOf.I32);
        return this.device(deviceType, deviceId);
      }
      case ArgTypeCode.TVMStr: {
        const ret = this.memory.loadCString(this.memory.loadPointer(rvaluePtr));
        return ret;
      }
      case ArgTypeCode.TVMBytes: {
        return this.memory.loadTVMBytes(this.memory.loadPointer(rvaluePtr));
      }
      default:
        throw new Error("Unsupported return type code=" + tcode);
    }
  }
}

/**
 * Asynchrously instantiate a new {@link Instance}.
 *
 * importObject can also be a {@link LibraryProvider} object,
 * a WASI object, or an object containing wasmLibraryProvider field.
 * We can take benefit of syslib implementations from the Emscripten
 * by passing its generated js Module as the imports.
 *
 * @param bufferSource The source to be compiled.
 * @param importObject The import objects.
 * @param logger The system logger.
 */
export function instantiate(
  bufferSource: ArrayBuffer,
  importObject: Record<string, any> = {},
  logger: (msg: string) => void = console.log
): Promise<Instance> {
  const env = new Environment(importObject, logger);

  return WebAssembly.instantiate(bufferSource, env.imports).then(
    (result: WebAssembly.WebAssemblyInstantiatedSource): Instance => {
      return new Instance(result.module, {}, result.instance, env);
    }
  );
}
