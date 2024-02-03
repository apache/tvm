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
 * Classes to manipulate Wasm memories.
 */
import { Pointer, PtrOffset, SizeOf } from "./ctypes";
import { Disposable } from "./types";
import { assert, StringToUint8Array } from "./support";

import * as ctypes from "./ctypes";

/**
 * Wasm Memory wrapper to perform JS side raw memory access.
 */
export class Memory {
  memory: WebAssembly.Memory;
  wasm32 = true;
  private buffer: ArrayBuffer | SharedArrayBuffer;
  private viewU8: Uint8Array;
  private viewU16: Uint16Array;
  private viewI32: Int32Array;
  private viewU32: Uint32Array;
  private viewF32: Float32Array;
  private viewF64: Float64Array;

  constructor(memory: WebAssembly.Memory) {
    this.memory = memory;
    this.buffer = this.memory.buffer;
    this.viewU8 = new Uint8Array(this.buffer);
    this.viewU16 = new Uint16Array(this.buffer);
    this.viewI32 = new Int32Array(this.buffer);
    this.viewU32 = new Uint32Array(this.buffer);
    this.viewF32 = new Float32Array(this.buffer);
    this.viewF64 = new Float64Array(this.buffer);
  }

  loadU8(ptr: Pointer): number {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    return this.viewU8[ptr >> 0];
  }

  loadU16(ptr: Pointer): number {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    return this.viewU16[ptr >> 1];
  }

  loadU32(ptr: Pointer): number {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    return this.viewU32[ptr >> 2];
  }

  loadI32(ptr: Pointer): number {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    return this.viewI32[ptr >> 2];
  }

  loadI64(ptr: Pointer): number {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    const base = ptr >> 2;
    // assumes little endian, for now truncate high.
    return this.viewI32[base];
  }

  loadF32(ptr: Pointer): number {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    return this.viewF32[ptr >> 2];
  }

  loadF64(ptr: Pointer): number {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    return this.viewF64[ptr >> 3];
  }

  loadPointer(ptr: Pointer): Pointer {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    if (this.wasm32) {
      return this.loadU32(ptr);
    } else {
      return this.loadI64(ptr);
    }
  }
  loadUSize(ptr: Pointer): Pointer {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    if (this.wasm32) {
      return this.loadU32(ptr);
    } else {
      return this.loadI64(ptr);
    }
  }
  sizeofPtr(): number {
    return this.wasm32 ? SizeOf.I32 : SizeOf.I64;
  }
  /**
   * Load raw bytes from ptr.
   * @param ptr The head address
   * @param numBytes The number
   */
  loadRawBytes(ptr: Pointer, numBytes: number): Uint8Array {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    const result = new Uint8Array(numBytes);
    result.set(this.viewU8.slice(ptr, ptr + numBytes));
    return result;
  }
  /**
   * Load TVMByteArray from ptr.
   *
   * @param ptr The address of the header.
   */
  loadTVMBytes(ptr: Pointer): Uint8Array {
    const data = this.loadPointer(ptr);
    const length = this.loadUSize(ptr + this.sizeofPtr());
    return this.loadRawBytes(data, length);
  }
  /**
   * Load null-terminated C-string from ptr.
   * @param ptr The head address
   */
  loadCString(ptr: Pointer): string {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    // NOTE: the views are still valid for read.
    const ret = [];
    let ch = 1;
    while (ch != 0) {
      ch = this.viewU8[ptr];
      if (ch != 0) {
        ret.push(String.fromCharCode(ch));
      }
      ++ptr;
    }
    return ret.join("");
  }
  /**
   * Store raw bytes to the ptr.
   * @param ptr The head address.
   * @param bytes The bytes content.
   */
  storeRawBytes(ptr: Pointer, bytes: Uint8Array): void {
    if (this.buffer != this.memory.buffer) {
      this.updateViews();
    }
    this.viewU8.set(bytes, ptr);
  }

  /**
   * Update memory view after the memory growth.
   */
  private updateViews(): void {
    this.buffer = this.memory.buffer;
    this.viewU8 = new Uint8Array(this.buffer);
    this.viewU16 = new Uint16Array(this.buffer);
    this.viewI32 = new Int32Array(this.buffer);
    this.viewU32 = new Uint32Array(this.buffer);
    this.viewF32 = new Float32Array(this.buffer);
    this.viewF64 = new Float64Array(this.buffer);
  }
}

/**
 * Auxiliary call stack for the FFI calls.
 *
 * Lifecyle of a call stack.
 * - Calls into allocXX to allocate space, mixed with storeXXX to store data.
 * - Calls into ptrFromOffset, no further allocation(as ptrFromOffset can change),
 *   can still call into storeXX
 * - Calls into commitToWasmMemory once.
 * - reset.
 */
export class CachedCallStack implements Disposable {
  /** List of temporay arguments that can be disposed during reset. */
  tempArgs: Array<Disposable> = [];

  private memory: Memory;
  private cAllocSpace: ctypes.FTVMWasmAllocSpace;
  private cFreeSpace: ctypes.FTVMWasmFreeSpace;

  private buffer: ArrayBuffer;
  private viewU8: Uint8Array;
  private viewI32: Int32Array;
  private viewU32: Uint32Array;
  private viewF64: Float64Array;

  private stackTop: PtrOffset = 0;
  private basePtr: Pointer = 0;

  private addressToSetTargetValue: Array<[PtrOffset, PtrOffset]> = [];

  constructor(
    memory: Memory,
    allocSpace: ctypes.FTVMWasmAllocSpace,
    freeSpace: ctypes.FTVMWasmFreeSpace
  ) {
    const initCallStackSize = 128;
    this.memory = memory;
    this.cAllocSpace = allocSpace;
    this.cFreeSpace = freeSpace;
    this.buffer = new ArrayBuffer(initCallStackSize);
    this.basePtr = this.cAllocSpace(initCallStackSize);
    this.viewU8 = new Uint8Array(this.buffer);
    this.viewI32 = new Int32Array(this.buffer);
    this.viewU32 = new Uint32Array(this.buffer);
    this.viewF64 = new Float64Array(this.buffer);
    this.updateViews();
  }

  dispose(): void {
    if (this.basePtr != 0) {
      this.cFreeSpace(this.basePtr);
      this.basePtr = 0;
    }
  }
  /**
   * Rest the call stack so that it can be reused again.
   */
  reset(): void {
    this.stackTop = 0;
    assert(this.addressToSetTargetValue.length === 0);
    while (this.tempArgs.length != 0) {
      (this.tempArgs.pop() as Disposable).dispose();
    }
  }

  /**
   * Commit all the cached data to WasmMemory.
   * This function can only be called once.
   * No further store function should be called.
   *
   * @param nbytes Number of bytes to be stored.
   */
  commitToWasmMemory(nbytes: number = this.stackTop): void {
    // commit all pointer values.
    while (this.addressToSetTargetValue.length != 0) {
      const [targetOffset, valueOffset] = this.addressToSetTargetValue.pop() as [
        number,
        number
      ];
      this.storePtr(targetOffset, this.ptrFromOffset(valueOffset));
    }
    this.memory.storeRawBytes(this.basePtr, this.viewU8.slice(0, nbytes));
  }

  /**
   * Allocate space by number of bytes
   * @param nbytes Number of bytes.
   * @note This function always allocate space that aligns to 64bit.
   */
  allocRawBytes(nbytes: number): PtrOffset {
    // always aligns to 64bit
    nbytes = ((nbytes + 7) >> 3) << 3;

    if (this.stackTop + nbytes > this.buffer.byteLength) {
      const newSize = Math.max(
        this.buffer.byteLength * 2,
        this.stackTop + nbytes
      );
      const oldU8 = this.viewU8;
      this.buffer = new ArrayBuffer(newSize);
      this.updateViews();
      this.viewU8.set(oldU8);
      if (this.basePtr != 0) {
        this.cFreeSpace(this.basePtr);
      }
      this.basePtr = this.cAllocSpace(newSize);
    }
    const retOffset = this.stackTop;
    this.stackTop += nbytes;
    return retOffset;
  }

  /**
   * Allocate space for pointers.
   * @param count Number of pointers.
   * @returns The allocated pointer array.
   */
  allocPtrArray(count: number): PtrOffset {
    return this.allocRawBytes(this.memory.sizeofPtr() * count);
  }

  /**
   * Get the real pointer from offset values.
   * Note that the returned value becomes obsolete if alloc is called on the stack.
   * @param offset The allocated offset.
   */
  ptrFromOffset(offset: PtrOffset): Pointer {
    return this.basePtr + offset;
  }

  // Store APIs
  storePtr(offset: PtrOffset, value: Pointer): void {
    if (this.memory.wasm32) {
      this.storeU32(offset, value);
    } else {
      this.storeI64(offset, value);
    }
  }

  storeUSize(offset: PtrOffset, value: Pointer): void {
    if (this.memory.wasm32) {
      this.storeU32(offset, value);
    } else {
      this.storeI64(offset, value);
    }
  }

  storeI32(offset: PtrOffset, value: number): void {
    this.viewI32[offset >> 2] = value;
  }

  storeU32(offset: PtrOffset, value: number): void {
    this.viewU32[offset >> 2] = value;
  }

  storeI64(offset: PtrOffset, value: number): void {
    // For now, just store as 32bit
    // NOTE: wasm always uses little endian.
    const low = value & 0xffffffff;
    const base = offset >> 2;
    this.viewI32[base] = low;
    // sign extend
    this.viewI32[base + 1] = value < 0 ? -1 : 0;
  }

  storeF64(offset: PtrOffset, value: number): void {
    this.viewF64[offset >> 3] = value;
  }

  storeRawBytes(offset: PtrOffset, bytes: Uint8Array): void {
    this.viewU8.set(bytes, offset);
  }

  /**
   * Allocate then set C-String pointer to the offset.
   * This function will call into allocBytes to allocate necessary data.
   * The address won't be set immediately(because the possible change of basePtr)
   * and will be filled when we commit the data.
   *
   * @param offset The offset to set ot data pointer.
   * @param data The string content.
   */
  allocThenSetArgString(offset: PtrOffset, data: string): void {
    const strOffset = this.allocRawBytes(data.length + 1);
    this.storeRawBytes(strOffset, StringToUint8Array(data));
    this.addressToSetTargetValue.push([offset, strOffset]);
  }
  /**
   * Allocate then set the argument location with a TVMByteArray.
   * Allocate new temporary space for bytes.
   *
   * @param offset The offset to set ot data pointer.
   * @param data The string content.
   */
  allocThenSetArgBytes(offset: PtrOffset, data: Uint8Array): void {
    // Note: size of size_t equals sizeof ptr.
    const headerOffset = this.allocRawBytes(this.memory.sizeofPtr() * 2);
    const dataOffset = this.allocRawBytes(data.length);
    this.storeRawBytes(dataOffset, data);
    this.storeUSize(headerOffset + this.memory.sizeofPtr(), data.length);

    this.addressToSetTargetValue.push([offset, headerOffset]);
    this.addressToSetTargetValue.push([headerOffset, dataOffset]);
  }

  /**
   * Update internal cache views.
   */
  private updateViews(): void {
    this.viewU8 = new Uint8Array(this.buffer);
    this.viewI32 = new Int32Array(this.buffer);
    this.viewU32 = new Uint32Array(this.buffer);
    this.viewF64 = new Float64Array(this.buffer);
  }
}
