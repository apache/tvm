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
const { CachedCallStack, Memory } = require("../../src/memory");

test("loadU64 reads unsigned values and rejects unsafe integers", () => {
  const wasmMemory = new WebAssembly.Memory({ initial: 1 });
  const memory = new Memory(wasmMemory);
  const words = new Uint32Array(wasmMemory.buffer);

  words.set([0x80000001, 1], 2);
  expect(memory.loadU64(8)).toBe(0x180000001);

  words.set([0, 0x200000], 2);
  expect(() => memory.loadU64(8)).toThrow("Cannot represent uint64 value");
});

test("loadRawBytes returns an owned copy", () => {
  const wasmMemory = new WebAssembly.Memory({ initial: 1 });
  const memory = new Memory(wasmMemory);
  const source = new Uint8Array(wasmMemory.buffer, 8, 4);
  source.set([1, 2, 3, 4]);

  const result = memory.loadRawBytes(8, 4);

  expect(Array.from(result)).toEqual([1, 2, 3, 4]);
  expect(result.buffer).not.toBe(wasmMemory.buffer);

  result[0] = 10;
  source[1] = 20;
  expect(Array.from(result)).toEqual([10, 2, 3, 4]);
  expect(Array.from(source)).toEqual([1, 20, 3, 4]);
});

test("loadRawBytes preserves the requested length at the end of memory", () => {
  const wasmMemory = new WebAssembly.Memory({ initial: 1 });
  const memory = new Memory(wasmMemory);
  const source = new Uint8Array(wasmMemory.buffer);
  source.set([5, 6], source.length - 2);

  const result = memory.loadRawBytes(source.length - 2, 4);

  expect(Array.from(result)).toEqual([5, 6, 0, 0]);
});

test("viewRawBytes returns a borrowed Wasm memory view", () => {
  const wasmMemory = new WebAssembly.Memory({ initial: 1 });
  const memory = new Memory(wasmMemory);
  const source = new Uint8Array(wasmMemory.buffer, 8, 4);
  source.set([1, 2, 3, 4]);

  const result = memory.viewRawBytes(8, 4);

  expect(Array.from(result)).toEqual([1, 2, 3, 4]);
  expect(result.buffer).toBe(wasmMemory.buffer);
  source[0] = 10;
  result[1] = 20;
  expect(Array.from(result)).toEqual([10, 20, 3, 4]);
  expect(Array.from(source)).toEqual([10, 20, 3, 4]);
});

test("viewRawBytes refreshes its backing view after memory growth", () => {
  const wasmMemory = new WebAssembly.Memory({ initial: 1, maximum: 2 });
  const memory = new Memory(wasmMemory);
  const oldBuffer = wasmMemory.buffer;

  wasmMemory.grow(1);
  const source = new Uint8Array(wasmMemory.buffer, 65536, 4);
  source.set([5, 6, 7, 8]);
  const result = memory.viewRawBytes(65536, 4);

  expect(result.buffer).toBe(wasmMemory.buffer);
  expect(result.buffer).not.toBe(oldBuffer);
  expect(Array.from(result)).toEqual([5, 6, 7, 8]);
});

test("viewRawBytes supports shared Wasm memory", () => {
  const wasmMemory = new WebAssembly.Memory({
    initial: 1,
    maximum: 2,
    shared: true,
  });
  const memory = new Memory(wasmMemory);
  const source = new Uint8Array(wasmMemory.buffer, 16, 4);
  source.set([1, 2, 3, 4]);

  const result = memory.viewRawBytes(16, 4);

  expect(result.buffer).toBe(wasmMemory.buffer);
  expect(result.buffer).toBeInstanceOf(SharedArrayBuffer);
  expect(Array.from(result)).toEqual([1, 2, 3, 4]);
});

test.each([
  [-1, 1, "pointer"],
  [0.5, 1, "pointer"],
  [Number.MAX_SAFE_INTEGER + 1, 1, "pointer"],
  [0, -1, "byte length"],
  [0, 0.5, "byte length"],
  [0, Number.MAX_SAFE_INTEGER + 1, "byte length"],
  [65536, 1, "exceeds memory size"],
  [65535, 2, "exceeds memory size"],
  [Number.MAX_SAFE_INTEGER, 1, "exceeds memory size"],
])("viewRawBytes rejects invalid range (%p, %p)", (ptr, nbytes, message) => {
  const memory = new Memory(new WebAssembly.Memory({ initial: 1 }));
  expect(() => memory.viewRawBytes(ptr, nbytes)).toThrow(message);
});

test("CachedCallStack commits a view of its cached bytes", () => {
  const memory = {
    wasm32: true,
    sizeofPtr: () => 4,
    storeRawBytes: jest.fn(),
  };
  const stack = new CachedCallStack(memory, () => 1024, () => {});
  const offset = stack.allocRawBytes(4);
  stack.storeRawBytes(offset, new Uint8Array([1, 2, 3, 4]));

  stack.commitToWasmMemory(4);

  expect(memory.storeRawBytes).toHaveBeenCalledTimes(1);
  const [ptr, bytes] = memory.storeRawBytes.mock.calls[0];
  expect(ptr).toBe(1024);
  expect(Array.from(bytes)).toEqual([1, 2, 3, 4]);
  expect(bytes.buffer).toBe(stack.buffer);
});
