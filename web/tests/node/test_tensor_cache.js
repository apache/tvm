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
const { getTensorCacheRecordBytes } = require("../../src/artifact_cache");

test("tensor-cache record is a borrowed ArrayBuffer view", () => {
  const shard = new Uint8Array([1, 2, 3, 4, 5, 6]).buffer;

  const record = getTensorCacheRecordBytes(shard, {
    byteOffset: 2,
    nbytes: 3,
  });

  expect(Array.from(record)).toEqual([3, 4, 5]);
  expect(record.buffer).toBe(shard);
  expect(record.byteOffset).toBe(2);
});

test("tensor-cache record respects a Uint8Array shard offset", () => {
  const backing = new Uint8Array([90, 91, 1, 2, 3, 4, 92]);
  const shard = backing.subarray(2, 6);

  const record = getTensorCacheRecordBytes(shard, {
    byteOffset: 1,
    nbytes: 2,
  });

  expect(Array.from(record)).toEqual([2, 3]);
  expect(record.buffer).toBe(backing.buffer);
  expect(record.byteOffset).toBe(shard.byteOffset + 1);
});

test("tensor-cache record may cover the full shard", () => {
  const shard = new Uint8Array([1, 2, 3, 4]);

  const record = getTensorCacheRecordBytes(shard, {
    byteOffset: 0,
    nbytes: shard.byteLength,
  });

  expect(record).toEqual(shard);
  expect(record.buffer).toBe(shard.buffer);
});

test("tensor-cache record may be empty at the end of the shard", () => {
  const shard = new Uint8Array([1, 2, 3, 4]);

  const record = getTensorCacheRecordBytes(shard, {
    byteOffset: shard.byteLength,
    nbytes: 0,
  });

  expect(record.byteLength).toBe(0);
  expect(record.byteOffset).toBe(shard.byteOffset + shard.byteLength);
});

test.each([
  [{ byteOffset: -1, nbytes: 1 }, "byteOffset"],
  [{ byteOffset: 0.5, nbytes: 1 }, "byteOffset"],
  [{ byteOffset: Number.MAX_SAFE_INTEGER + 1, nbytes: 1 }, "byteOffset"],
  [{ byteOffset: 0, nbytes: -1 }, "nbytes"],
  [{ byteOffset: 0, nbytes: 0.5 }, "nbytes"],
  [{ byteOffset: 0, nbytes: Number.MAX_SAFE_INTEGER + 1 }, "nbytes"],
  [{ byteOffset: 5, nbytes: 0 }, "exceeds shard size"],
  [{ byteOffset: 3, nbytes: 2 }, "exceeds shard size"],
  [{ byteOffset: Number.MAX_SAFE_INTEGER, nbytes: 1 }, "exceeds shard size"],
])("tensor-cache record rejects invalid range %j", (range, message) => {
  expect(() => getTensorCacheRecordBytes(new ArrayBuffer(4), range)).toThrow(
    message,
  );
});
