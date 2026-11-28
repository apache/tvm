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

const { OPFSStore } = require("../../src/opfs_store.ts");

const largeSize = 2179989504;

function createStore(file) {
  const store = new OPFSStore("test");
  jest.spyOn(store, "getStoredEntry").mockResolvedValue({
    payloadHandle: { getFile: async () => file },
    record: { nbytes: file.size },
  });
  return store;
}

test("a small OPFS payload uses the direct blob read", async () => {
  const blob = new Blob([Uint8Array.of(1, 2, 3)]);
  const stream = jest.spyOn(blob, "stream");
  const store = createStore(blob);
  expect(new Uint8Array(await store.readArrayBuffer("weights"))).toEqual(Uint8Array.of(1, 2, 3));
  expect(stream).not.toHaveBeenCalled();
});

function mockLargePayload(chunks) {
  const reader = {
    read: jest.fn(),
    cancel: jest.fn().mockResolvedValue(undefined),
    releaseLock: jest.fn(),
  };
  for (const chunk of chunks) {
    reader.read.mockResolvedValueOnce({ done: false, value: chunk });
  }
  reader.read.mockResolvedValue({ done: true });
  const blob = {
    size: largeSize,
    arrayBuffer: jest.fn(),
    stream: () => ({ getReader: () => reader }),
  };
  // Keep the exact file size and offsets without allocating a multi-GiB test buffer.
  const payload = { byteLength: largeSize, buffer: { byteLength: largeSize }, set: jest.fn() };
  const Uint8ArrayConstructor = Uint8Array;
  const constructor = jest.spyOn(globalThis, "Uint8Array").mockImplementation((...args) =>
    args[0] === largeSize ? payload : new Uint8ArrayConstructor(...args)
  );
  return { store: createStore(blob), blob, reader, payload, constructor };
}

test("an OPFS payload over 2 GiB is read from the blob stream", async () => {
  const chunks = [{ byteLength: 2147483648 }, { byteLength: 32505856 }];
  const { store, blob, reader, payload, constructor } = mockLargePayload(chunks);
  try {
    expect(await store.readArrayBuffer("weights")).toBe(payload.buffer);
    expect(payload.set.mock.calls).toEqual([[chunks[0], 0], [chunks[1], 2147483648]]);
    expect(blob.arrayBuffer).not.toHaveBeenCalled();
    expect(reader.releaseLock).toHaveBeenCalledTimes(1);
    expect(reader.cancel).not.toHaveBeenCalled();
  } finally {
    constructor.mockRestore();
  }
});

test("a truncated OPFS stream fails without returning zero-padded bytes", async () => {
  const { store, reader, constructor } = mockLargePayload([{ byteLength: 24 }]);
  try {
    await expect(store.readArrayBuffer("weights")).rejects.toThrow("Incomplete payload read");
    expect(reader.cancel).toHaveBeenCalledTimes(1);
    expect(reader.releaseLock).toHaveBeenCalledTimes(1);
  } finally {
    constructor.mockRestore();
  }
});
