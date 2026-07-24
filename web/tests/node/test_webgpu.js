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

function createMockDevice() {
  const events = [];
  const encoders = [];

  const queue = {
    submit: jest.fn(() => events.push("submit")),
    writeBuffer: jest.fn(() => events.push("writeBuffer")),
    onSubmittedWorkDone: jest.fn(() => Promise.resolve()),
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
        copyBufferToBuffer: jest.fn(() => {
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
      return {
        size: descriptor.size,
        destroy: jest.fn(() => events.push("destroy")),
        mapAsync: jest.fn(() => Promise.resolve()),
        getMappedRange: jest.fn(() => mappedData),
        unmap: jest.fn(),
      };
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

  return { device, queue, events, encoders };
}

function createContext() {
  const gpu = createMockDevice();
  const memory = {
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
