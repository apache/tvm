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
const { WebGPUContext } = require("../../src/webgpu");

global.GPUBufferUsage = {
  MAP_READ: 1 << 0,
  COPY_DST: 1 << 1,
  COPY_SRC: 1 << 2,
  STORAGE: 1 << 3,
  UNIFORM: 1 << 4,
};
global.GPUMapMode = {
  READ: 1,
};
global.GPUShaderStage = {
  COMPUTE: 1,
};

function createDeferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function createMockDevice({
  mapAsync = () => Promise.resolve(),
  onSubmittedWorkDone = () => Promise.resolve(),
} = {}) {
  const events = [];
  const encoders = [];
  const buffers = [];

  const queue = {
    submit: jest.fn(() => events.push("submit")),
    writeBuffer: jest.fn(() => events.push("writeBuffer")),
    onSubmittedWorkDone: jest.fn(onSubmittedWorkDone),
  };

  const device = {
    queue,
    createCommandEncoder: jest.fn(() => {
      const commands = [];
      const encoderId = encoders.length;
      const encoder = {
        commands,
        beginComputePass: jest.fn(() => ({
          setPipeline: jest.fn(),
          setBindGroup: jest.fn(),
          dispatchWorkgroups: jest.fn(() => {
            commands.push("compute");
            events.push("compute");
          }),
          end: jest.fn(),
        })),
        copyBufferToBuffer: jest.fn((src, srcOffset, dst, dstOffset, size) => {
          if (src.data && dst.data) {
            new Uint8Array(dst.data, dstOffset, size).set(new Uint8Array(src.data, srcOffset, size));
          }
          commands.push("copy");
          events.push("copy");
        }),
        finish: jest.fn(() => {
          const commandBuffer = { encoderId, commands: commands.slice() };
          events.push("finish");
          return commandBuffer;
        }),
      };
      encoders.push(encoder);
      return encoder;
    }),
    createBuffer: jest.fn((descriptor) => {
      const mappedData = new ArrayBuffer(descriptor.size);
      const buffer = {
        data: mappedData,
        size: descriptor.size,
        destroy: jest.fn(() => events.push("destroy")),
        mapAsync: jest.fn(mapAsync),
        getMappedRange: jest.fn((offset = 0, size = descriptor.size - offset) =>
          mappedData.slice(offset, offset + size)),
        unmap: jest.fn(),
      };
      buffers.push(buffer);
      return buffer;
    }),
    createBindGroupLayout: jest.fn(() => ({})),
    createPipelineLayout: jest.fn(() => ({})),
    createShaderModule: jest.fn(() => ({})),
    createComputePipeline: jest.fn(() => ({})),
    createBindGroup: jest.fn(() => ({})),
    pushErrorScope: jest.fn(),
    popErrorScope: jest.fn(() => Promise.resolve(null)),
    destroy: jest.fn(),
  };

  return { device, queue, events, encoders, buffers };
}

function createContext(deviceOptions) {
  const gpu = createMockDevice(deviceOptions);
  const memory = {
    loadRawBytes: jest.fn(),
    viewRawBytes: jest.fn(),
    storeRawBytes: jest.fn(),
  };
  const context = new WebGPUContext(memory, gpu.device);
  const allocate = context.getDeviceAPI("deviceAllocDataSpace");

  return {
    ...gpu,
    context,
    memory,
    source: allocate(64),
    destination: allocate(64),
  };
}

test("compute dispatches and GPU copies share one submission", async () => {
  const { context, device, queue, encoders, source, destination } = createContext();
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const shader = context.createShader(
    {
      name: "main",
      arg_types: [],
      launch_param_tags: [],
    },
    "@compute @workgroup_size(1) fn main() {}"
  );

  shader();
  copyWithinGPU(source, 0, destination, 0, 16);
  copyWithinGPU(destination, 16, source, 32, 16);

  expect(device.createCommandEncoder).toHaveBeenCalledTimes(1);
  expect(queue.submit).not.toHaveBeenCalled();
  expect(encoders[0].commands).toEqual(["compute", "copy", "copy"]);

  await context.sync();

  expect(encoders[0].finish).toHaveBeenCalledTimes(1);
  expect(queue.submit).toHaveBeenCalledTimes(1);
  expect(queue.submit.mock.calls[0][0]).toEqual([
    {
      encoderId: 0,
      commands: encoders[0].commands,
    },
  ]);
  expect(queue.onSubmittedWorkDone).toHaveBeenCalledTimes(1);

  await context.sync();
  expect(queue.submit).toHaveBeenCalledTimes(1);
});

test("pending WebGPU commands can be submitted without waiting", () => {
  const { context, queue } = createContext();
  const shader = context.createShader(
    {
      name: "main",
      arg_types: [],
      launch_param_tags: [],
    },
    "@compute @workgroup_size(1) fn main() {}"
  );
  shader();
  expect(context.submitPendingCommands()).toBe(true);
  expect(queue.submit).toHaveBeenCalledTimes(1);
  expect(queue.onSubmittedWorkDone).not.toHaveBeenCalled();

  expect(context.submitPendingCommands()).toBe(false);
  expect(queue.submit).toHaveBeenCalledTimes(1);
});

test("sampled-token ring copy shares the pending compute submission", async () => {
  const { context, device, queue, encoders, source } = createContext();
  const shader = context.createShader(
    {
      name: "main",
      arg_types: [],
      launch_param_tags: [],
    },
    "@compute @workgroup_size(1) fn main() {}"
  );
  const ringId = context.createSampledTokenReadbackRing({
    slotCount: 2,
    maxTokensPerBatch: 1,
  });

  shader();
  const batchSeq = context.submitSampledTokenReadbackRing(ringId, source, 0, 1);

  expect(batchSeq).toBe(1);
  expect(device.createCommandEncoder).toHaveBeenCalledTimes(1);
  expect(encoders[0].commands).toEqual(["compute", "copy"]);
  expect(queue.submit).toHaveBeenCalledTimes(1);
  expect(queue.submit.mock.calls[0][0]).toEqual([
    {
      encoderId: 0,
      commands: encoders[0].commands,
    },
  ]);

  const batches = await context.waitSampledTokenReadbackRing(ringId);
  expect(batches).toHaveLength(1);
  expect(batches[0].batchSeq).toBe(1);
  expect(batches[0].tokenCount).toBe(1);
  context.disposeSampledTokenReadbackRing(ringId);
});

test("a readback ring submit with an unknown source leaves the ring usable", async () => {
  const { context, source } = createContext();
  const ringId = context.createSampledTokenReadbackRing({
    slotCount: 1,
    maxTokensPerBatch: 1,
  });

  expect(() =>
    context.submitSampledTokenReadbackRing(ringId, 0x7fff0000, 0, 1),
  ).toThrow();

  // The failed submit consumed neither the slot nor a batch sequence.
  expect(context.submitSampledTokenReadbackRing(ringId, source, 0, 1)).toBe(1);
  const batches = await context.waitSampledTokenReadbackRing(ringId);
  expect(batches.map((batch) => batch.batchSeq)).toEqual([1]);
  context.disposeSampledTokenReadbackRing(ringId);
});

test("invalid readback ranges leave slots and sequence numbers available", async () => {
  const { context, source } = createContext();
  const ringId = context.createSampledTokenReadbackRing({ slotCount: 1, maxTokensPerBatch: 2 });
  for (const [offset, count] of [[-1, 1], [0, 0], [0.5, 1], [16, 1], [15, 2], [0, 3]]) {
    expect(() => context.submitSampledTokenReadbackRing(ringId, source, offset, count)).toThrow();
  }
  expect(() => context.submitSampledTokenReadbackRing(
    ringId, source, 0, 2, new Int32Array(1),
  )).toThrow();
  expect(context.submitSampledTokenReadbackRing(ringId, source, 15, 1)).toBe(1);
  expect((await context.waitSampledTokenReadbackRing(ringId))[0].batchSeq).toBe(1);
  context.disposeSampledTokenReadbackRing(ringId);
});

test("a full ring becomes reusable after draining its oldest batch", async () => {
  const { context, source } = createContext();
  const ringId = context.createSampledTokenReadbackRing({ slotCount: 1, maxTokensPerBatch: 1 });
  expect(context.submitSampledTokenReadbackRing(ringId, source, 0, 1)).toBe(1);
  expect(() => context.submitSampledTokenReadbackRing(ringId, source, 0, 1)).toThrow("full");
  await context.waitSampledTokenReadbackRing(ringId);
  expect(context.submitSampledTokenReadbackRing(ringId, source, 0, 1)).toBe(2);
  await context.waitSampledTokenReadbackRing(ringId);
  context.disposeSampledTokenReadbackRing(ringId);
});

test("disposal wakes a waiter and defers buffer destruction until mapping settles", async () => {
  const mapped = createDeferred();
  const { context, source, buffers } = createContext({ mapAsync: () => mapped.promise });
  const ringId = context.createSampledTokenReadbackRing({ slotCount: 1, maxTokensPerBatch: 1 });
  context.submitSampledTokenReadbackRing(ringId, source, 0, 1);
  const waiting = context.waitSampledTokenReadbackRing(ringId);
  context.disposeSampledTokenReadbackRing(ringId);
  expect(buffers[2].destroy).not.toHaveBeenCalled();
  await expect(waiting).resolves.toEqual([]);
  mapped.resolve();
  await context.sync();
  expect(buffers[2].unmap).toHaveBeenCalledTimes(1);
  expect(buffers[2].destroy).toHaveBeenCalledTimes(1);
});

test("submission failure rejects an existing ring waiter", async () => {
  const mapped = createDeferred();
  const { context, source, queue } = createContext({ mapAsync: () => mapped.promise });
  const ringId = context.createSampledTokenReadbackRing({ slotCount: 2, maxTokensPerBatch: 1 });
  context.submitSampledTokenReadbackRing(ringId, source, 0, 1);
  const waiting = context.waitSampledTokenReadbackRing(ringId);
  queue.submit.mockImplementationOnce(() => { throw new Error("submission failed"); });
  expect(() => context.submitSampledTokenReadbackRing(ringId, source, 0, 1))
    .toThrow("submission failed");
  await expect(waiting).rejects.toThrow("submission failed");
  expect(() => context.pollSampledTokenReadbackRing(ringId)).toThrow("submission failed");
  context.disposeSampledTokenReadbackRing(ringId);
  mapped.resolve();
  await context.sync();
});

test("a readback ring map failure is reported through the ring", async () => {
  const unhandled = [];
  const onUnhandled = (reason) => unhandled.push(reason);
  process.on("unhandledRejection", onUnhandled);
  try {
    const { context, source } = createContext({
      mapAsync: () => Promise.reject(new Error("map failed")),
    });
    const ringId = context.createSampledTokenReadbackRing({
      slotCount: 1,
      maxTokensPerBatch: 1,
    });
    context.submitSampledTokenReadbackRing(ringId, source, 0, 1);

    await expect(context.waitSampledTokenReadbackRing(ringId)).rejects.toThrow(
      "map failed",
    );
    expect(() => context.pollSampledTokenReadbackRing(ringId)).toThrow("map failed");
    await context.sync();
    await new Promise((resolve) => setImmediate(resolve));
    expect(unhandled).toEqual([]);
    context.disposeSampledTokenReadbackRing(ringId);
  } finally {
    process.off("unhandledRejection", onUnhandled);
  }
});

test.each([1, 4, 16])(
  "sampled-token ring concatenates K=%i token buffers in one submission",
  async (batchSize) => {
    const { context, device, queue, encoders, buffers } = createContext();
    const allocate = context.getDeviceAPI("deviceAllocDataSpace");
    const tokenSources = Array.from({ length: batchSize }, () => allocate(64));
    const ringId = context.createSampledTokenReadbackRing({
      slotCount: 2,
      maxTokensPerBatch: batchSize,
    });

    const batchSeq = context.submitSampledTokenReadbackRingSources(
      ringId,
      tokenSources.map((from) => ({ from, fromTokenOffset: 0, tokenCount: 1 })),
    );

    expect(batchSeq).toBe(1);
    expect(device.createCommandEncoder).toHaveBeenCalledTimes(1);
    expect(encoders[0].commands).toEqual(Array(batchSize).fill("copy"));
    expect(encoders[0].copyBufferToBuffer.mock.calls).toEqual(
      Array.from({ length: batchSize }, (_, index) => [
        buffers[2 + index],
        0,
        buffers[2 + batchSize],
        index * 4,
        4,
      ]),
    );
    expect(queue.submit).toHaveBeenCalledTimes(1);

    const batches = await context.waitSampledTokenReadbackRing(ringId);
    expect(batches).toHaveLength(1);
    expect(batches[0].tokenCount).toBe(batchSize);
    expect(Array.from(batches[0].tokens)).toEqual(Array(batchSize).fill(0));
    context.disposeSampledTokenReadbackRing(ringId);
  },
);

test("sampled-token ring rejects an empty token-buffer batch", () => {
  const { context } = createContext();
  const ringId = context.createSampledTokenReadbackRing({
    slotCount: 2,
    maxTokensPerBatch: 3,
  });

  expect(() =>
    context.submitSampledTokenReadbackRingSources(ringId, [])
  ).toThrow("At least one sampled-token source is required.");
  context.disposeSampledTokenReadbackRing(ringId);
});

// Guards out-of-order map completion (seen in Firefox): waiting must not
// return an empty batch list while the next batch in sequence is pending.
test("ring wait returns batches in sequence when maps resolve out of order", async () => {
  const resolvers = [];
  const { context, source } = createContext({
    mapAsync: () => new Promise((resolve) => resolvers.push(resolve)),
  });
  const ringId = context.createSampledTokenReadbackRing({
    slotCount: 2,
    maxTokensPerBatch: 1,
  });
  context.submitSampledTokenReadbackRing(ringId, source, 0, 1);
  context.submitSampledTokenReadbackRing(ringId, source, 0, 1);
  const waiting = context.waitSampledTokenReadbackRing(ringId);
  resolvers[1]();
  await new Promise((resolve) => setTimeout(resolve, 0));
  resolvers[0]();
  const batches = await waiting;
  expect(batches.map((batch) => batch.batchSeq)).toEqual([1, 2]);
  context.disposeSampledTokenReadbackRing(ringId);
});

// Guards readback into a caller's array, which a caller reuses across
// batches: a batch is copied into it, and a ring disposed while a slot maps
// must not write into an array its owner may already have reused.
test("ring copies a batch into the caller's array only while the ring lives", async () => {
  const resolvers = [];
  const { context, source, buffers } = createContext({
    mapAsync: () => new Promise((resolve) => resolvers.push(resolve)),
  });
  const ringId = context.createSampledTokenReadbackRing({
    slotCount: 2,
    maxTokensPerBatch: 2,
  });
  const into = new Int32Array(4).fill(-1);
  expect(() =>
    context.submitSampledTokenReadbackRing(ringId, source, 0, 2, new Int32Array(1)),
  ).toThrow("destination length");
  context.submitSampledTokenReadbackRing(ringId, source, 0, 2, into);
  new Int32Array(buffers[2].data).set([7, 8]);
  resolvers[0]();
  const [batch] = await context.waitSampledTokenReadbackRing(ringId);
  expect(batch.tokens.buffer).toBe(into.buffer);
  expect(Array.from(into)).toEqual([7, 8, -1, -1]);

  into.fill(-1);
  context.submitSampledTokenReadbackRing(ringId, source, 0, 2, into);
  context.disposeSampledTokenReadbackRing(ringId);
  resolvers[1]();
  await context.sync();
  expect(Array.from(into)).toEqual([-1, -1, -1, -1]);
});

test("a host write flushes pending GPU copies before writeBuffer", () => {
  const { context, queue, events, source, destination } = createContext();
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const rawBytes = new Uint8Array([1, 2, 3, 4]);

  copyWithinGPU(source, 0, destination, 0, rawBytes.length);
  expect(queue.submit).not.toHaveBeenCalled();

  context.copyRawBytesToBuffer(rawBytes, destination, 4, rawBytes.length);

  expect(queue.submit).toHaveBeenCalledTimes(1);
  expect(queue.writeBuffer).toHaveBeenCalledTimes(1);
  expect(events).toEqual(["copy", "finish", "submit", "writeBuffer"]);
});

test("an aligned CPU to GPU copy writes the requested bytes", () => {
  const { context, device, queue, memory, destination } = createContext();
  const copyToGPU = context.getDeviceAPI("deviceCopyToGPU");
  const wasmMemory = new WebAssembly.Memory({ initial: 1 });
  const rawBytes = new Uint8Array(wasmMemory.buffer, 128, 8);
  rawBytes.set([1, 2, 3, 4, 5, 6, 7, 8]);
  memory.viewRawBytes.mockReturnValue(rawBytes);

  copyToGPU(128, destination, 12, rawBytes.length);

  expect(memory.viewRawBytes).toHaveBeenCalledWith(128, rawBytes.length);
  expect(memory.loadRawBytes).not.toHaveBeenCalled();
  expect(queue.writeBuffer.mock.calls[0][2].buffer).toBe(wasmMemory.buffer);
  expect(queue.writeBuffer).toHaveBeenCalledWith(
    device.createBuffer.mock.results[1].value,
    12,
    rawBytes,
    0,
    rawBytes.length
  );
});

test("an unaligned CPU to GPU copy pads the write to four bytes", () => {
  const { context, device, queue, memory, destination } = createContext();
  const copyToGPU = context.getDeviceAPI("deviceCopyToGPU");
  const rawBytes = new Uint8Array([1, 2, 3]);
  memory.viewRawBytes.mockReturnValue(rawBytes);

  copyToGPU(256, destination, 4, rawBytes.length);

  expect(memory.viewRawBytes).toHaveBeenCalledWith(256, rawBytes.length);
  expect(memory.loadRawBytes).not.toHaveBeenCalled();
  expect(queue.writeBuffer).toHaveBeenCalledTimes(1);
  const [buffer, toOffset, data, dataOffset, nbytes] =
    queue.writeBuffer.mock.calls[0];
  expect(buffer).toBe(device.createBuffer.mock.results[1].value);
  expect(toOffset).toBe(4);
  expect(Array.from(data)).toEqual([1, 2, 3, 0]);
  expect(dataOffset).toBe(0);
  expect(nbytes).toBe(4);
});

test("a non-four-byte GPU allocation is rounded up for padded writes", () => {
  const { context, device } = createContext();
  const allocate = context.getDeviceAPI("deviceAllocDataSpace");

  allocate(3);

  expect(device.createBuffer).toHaveBeenLastCalledWith({
    size: 4,
    usage: GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  expect(context.currAllocatedBytes).toBe(64 + 64 + 4);
});

test.each([
  [-1, "destination offset"],
  [0.5, "destination offset"],
  [2, "destination offset"],
])("a CPU to GPU copy rejects invalid offset %p", (offset, message) => {
  const { context, memory, destination } = createContext();
  const copyToGPU = context.getDeviceAPI("deviceCopyToGPU");

  expect(() => copyToGPU(128, destination, offset, 4)).toThrow(message);
  expect(memory.viewRawBytes).not.toHaveBeenCalled();
});

test.each([
  [-1, "source offset"],
  [0.5, "source offset"],
  [2, "source offset"],
])("a GPU readback rejects invalid offset %p", (offset, message) => {
  const { context, source } = createContext();
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");

  expect(() => copyFromGPU(source, offset, 128, 4)).toThrow(message);
});

test("an unaligned GPU readback copies four bytes and stores the logical bytes", async () => {
  const {
    context,
    device,
    memory,
    source,
  } = createContext();
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");
  device.createBuffer.mockImplementationOnce((descriptor) => {
    const mappedData = new Uint8Array([1, 2, 3, 99]).buffer;
    return {
      size: descriptor.size,
      destroy: jest.fn(),
      mapAsync: jest.fn(() => Promise.resolve()),
      getMappedRange: jest.fn(() => mappedData),
      unmap: jest.fn(),
    };
  });

  copyFromGPU(source, 0, 128, 3);
  await context.sync();

  const copyEncoder = device.createCommandEncoder.mock.results[0].value;
  expect(copyEncoder.copyBufferToBuffer).toHaveBeenCalledWith(
    device.createBuffer.mock.results[0].value,
    0,
    device.createBuffer.mock.results[2].value,
    0,
    4,
  );
  expect(memory.storeRawBytes).toHaveBeenCalledTimes(1);
  expect(memory.storeRawBytes.mock.calls[0][0]).toBe(128);
  expect(Array.from(memory.storeRawBytes.mock.calls[0][1])).toEqual([1, 2, 3]);
});

test("a GPU readback flushes pending copies before its own submission", async () => {
  const {
    context,
    queue,
    events,
    encoders,
    memory,
    source,
    destination,
  } = createContext();
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");

  copyWithinGPU(source, 0, destination, 0, 16);
  copyFromGPU(destination, 0, 128, 16);

  expect(queue.submit).toHaveBeenCalledTimes(2);
  expect(encoders).toHaveLength(2);
  expect(encoders[0].commands).toEqual(["copy"]);
  expect(encoders[1].commands).toEqual(["copy"]);
  expect(events).toEqual(["copy", "finish", "submit", "copy", "finish", "submit"]);

  await context.sync();

  expect(memory.storeRawBytes).toHaveBeenCalledTimes(1);
  expect(memory.storeRawBytes.mock.calls[0][0]).toBe(128);
  expect(memory.storeRawBytes.mock.calls[0][1]).toHaveLength(16);
  expect(queue.onSubmittedWorkDone).not.toHaveBeenCalled();
});

test("buffer deallocation flushes pending copies before destroy", () => {
  const { context, queue, events, source, destination } = createContext();
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const free = context.getDeviceAPI("deviceFreeDataSpace");

  copyWithinGPU(source, 0, destination, 0, 16);
  free(source);

  expect(queue.submit).toHaveBeenCalledTimes(1);
  expect(events).toEqual(["copy", "finish", "submit", "destroy"]);
});

test("drawing flushes pending copies first", () => {
  const { context, queue, events, source, destination } = createContext();
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const canvasRenderManager = {
    draw: jest.fn(() => events.push("draw")),
  };
  context.canvasRenderManager = canvasRenderManager;

  copyWithinGPU(source, 0, destination, 0, 16);
  context.drawImageFromBuffer(destination, 2, 2);

  expect(queue.submit).toHaveBeenCalledTimes(1);
  expect(canvasRenderManager.draw).toHaveBeenCalledTimes(1);
  expect(events).toEqual(["copy", "finish", "submit", "draw"]);
});

test("sync awaits a readback and a later batched GPU copy", async () => {
  const readback = createDeferred();
  const queueDone = createDeferred();
  const {
    context,
    queue,
    memory,
    source,
    destination,
  } = createContext({
    mapAsync: () => readback.promise,
    onSubmittedWorkDone: () => queueDone.promise,
  });
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");

  copyFromGPU(source, 0, 128, 16);
  copyWithinGPU(source, 0, destination, 0, 16);

  let syncResolved = false;
  const syncPromise = context.sync().then(() => {
    syncResolved = true;
  });

  expect(queue.submit).toHaveBeenCalledTimes(2);
  expect(queue.onSubmittedWorkDone).toHaveBeenCalledTimes(1);

  queueDone.resolve();
  await Promise.resolve();
  expect(syncResolved).toBe(false);
  expect(memory.storeRawBytes).not.toHaveBeenCalled();

  readback.resolve();
  await syncPromise;

  expect(memory.storeRawBytes).toHaveBeenCalledTimes(1);
  expect(memory.storeRawBytes.mock.calls[0][0]).toBe(128);
  expect(memory.storeRawBytes.mock.calls[0][1]).toHaveLength(16);
});

test("a host write after a readback makes sync wait for the queue", async () => {
  const queueDone = createDeferred();
  const {
    context,
    queue,
    memory,
    source,
    destination,
  } = createContext({
    onSubmittedWorkDone: () => queueDone.promise,
  });
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");

  copyFromGPU(source, 0, 128, 16);
  context.copyRawBytesToBuffer(
    new Uint8Array([1, 2, 3, 4]),
    destination,
    0,
    4
  );

  let syncResolved = false;
  const syncPromise = context.sync().then(() => {
    syncResolved = true;
  });
  expect(queue.onSubmittedWorkDone).toHaveBeenCalledTimes(1);

  await Promise.resolve();
  expect(syncResolved).toBe(false);

  queueDone.resolve();
  await syncPromise;

  expect(memory.storeRawBytes).toHaveBeenCalledTimes(1);
});

test("sync propagates a pending readback failure", async () => {
  const readError = new Error("mapAsync failed");
  const {
    context,
    queue,
    source,
    destination,
  } = createContext({
    mapAsync: () => Promise.reject(readError),
  });
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");

  copyFromGPU(source, 0, 128, 16);
  copyWithinGPU(source, 0, destination, 0, 16);

  await expect(context.sync()).rejects.toBe(readError);
  expect(queue.onSubmittedWorkDone).toHaveBeenCalledTimes(1);
});
