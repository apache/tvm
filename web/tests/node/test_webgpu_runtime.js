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
jest.mock("../../src/compact", () => ({}));
const { DLDevice, Tensor } = require("../../src/runtime");

function tokenTensor(byteOffset = 0) {
  return Object.assign(Object.create(Tensor.prototype), {
    handle: 1, dataPtr: 10, byteOffset, dtype: "int32", shape: [2],
    device: { deviceType: 15 },
  });
}

test("tensor readback includes view offsets and checks tensor bounds", () => {
  const context = {
    createSampledTokenReadbackRing: jest.fn(() => 1),
    submitSampledTokenReadbackRing: jest.fn(() => 1),
    submitSampledTokenReadbackRingSources: jest.fn(() => 2),
    disposeSampledTokenReadbackRing: jest.fn(),
  };
  const ring = new DLDevice("webgpu", 0, { webGPUContext: context })
    .createSampledTokenReadbackRing({ maxTokensPerBatch: 2 });
  const from = tokenTensor(8);
  expect(ring.submit(from, 1, 1)).toBe(1);
  expect(context.submitSampledTokenReadbackRing).toHaveBeenCalledWith(1, 10, 3, 1, undefined);
  expect(ring.submitTokens([from, tokenTensor(4)])).toBe(2);
  expect(context.submitSampledTokenReadbackRingSources).toHaveBeenCalledWith(1, [
    { from: 10, fromTokenOffset: 2, tokenCount: 1 },
    { from: 10, fromTokenOffset: 1, tokenCount: 1 },
  ]);
  expect(() => ring.submit(from, 2, 1)).toThrow("token range");
  from.dtype = "float32";
  expect(() => ring.submit(from, 1)).toThrow("int32 WebGPU tensor");
  ring.dispose();
  ring.dispose();
  expect(context.disposeSampledTokenReadbackRing).toHaveBeenCalledTimes(1);
  expect(() => ring.submitTokens([])).toThrow("disposed");
});

test("readback rings reject CPU devices", () => {
  expect(() => new DLDevice("cpu", 0, {}).createSampledTokenReadbackRing({maxTokensPerBatch: 1}))
    .toThrow("only available on WebGPU");
});

test("public nonblocking submission delegates only for WebGPU devices", () => {
  const context = { submitPendingCommands: jest.fn().mockReturnValueOnce(true).mockReturnValueOnce(false) };
  const lib = { webGPUContext: context };
  const gpu = new DLDevice("webgpu", 0, lib);
  expect(gpu.submitPendingCommands()).toBe(true);
  expect(gpu.submitPendingCommands()).toBe(false);
  expect(context.submitPendingCommands).toHaveBeenCalledTimes(2);
  expect(() => new DLDevice("cpu", 0, lib).submitPendingCommands()).toThrow("only available on WebGPU");
  expect(context.submitPendingCommands).toHaveBeenCalledTimes(2);
});
