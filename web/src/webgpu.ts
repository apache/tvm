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
import "@webgpu/types";
import { assert } from "./support";
import { Pointer } from "./ctypes";
import { Memory } from "./memory";

/** A pointer to points to the raw address space. */
export type GPUPointer = number;

/**
 * DetectGPU device in the environment.
 */
export async function detectGPUDevice(): Promise<GPUDevice | undefined | null> {
  if (typeof navigator !== "undefined" && navigator.gpu !== undefined) {
    const adapter = await navigator.gpu.requestAdapter();
    return await adapter?.requestDevice();
  } else {
    return undefined;
  }
}

interface FunctionInfo {
  name: string;
  arg_types: Array<string>;
  launch_param_tags: Array<string>;
}

/**
 * WebGPU context
 * Manages all the webgpu resources here.
 */
export class WebGPUContext {
  device: GPUDevice;
  memory: Memory;

  //private readBuffer:;
  private bufferTable: Array<GPUBuffer | undefined> = [undefined];
  private bufferTableFreeId: Array<number> = [];
  private pendingRead: Promise<void> = Promise.resolve();
  private numPendingReads = 0;

  constructor(memory: Memory, device: GPUDevice) {
    this.memory = memory;
    this.device = device;
  }

  /**
   * Wait for all pending GPU tasks to complete
   */
  async sync(): Promise<void> {
    const fence = this.device.defaultQueue.createFence();
    this.device.defaultQueue.signal(fence, 1);
    if (this.numPendingReads != 0) {
      // eslint-disable-next-line @typescript-eslint/no-empty-function
      await Promise.all([fence.onCompletion(1), this.pendingRead]);
    } else {
      await fence.onCompletion(1);
    }
  }

  /**
   * Create a PackedFunc that runs the given shader
   *
   * @param info The function information in json.
   * @param data The shader data(in SPIRV)
   */
  createShader(info: string, data: Uint8Array): Function {
    const finfo = JSON.parse(info);
    const layoutEntries: Array<GPUBindGroupLayoutEntry> = [];
    for (let i = 0; i < finfo.arg_types.length; ++i) {
      const dtype = finfo.arg_types[i];
      if (dtype == "handle") {
        layoutEntries.push({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          type: "storage-buffer"
        });
      } else {
        throw new Error("Cannot handle argument type " + dtype + " in WebGPU shader");
      }
    }
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: layoutEntries
    });

    const pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [ bindGroupLayout ]
      }),
      computeStage: {
        module: this.device.createShaderModule({
          code: new Uint32Array(data.buffer)
        }),
        entryPoint: "main"
      }
    });

    const dispatchToDim: Array<number> = [];

    for (let i = 0; i < finfo.launch_param_tags.length; ++i) {
      const tag: string = finfo.launch_param_tags[i];
      if (tag.startsWith("blockIdx.")) {
        const target: number = tag.charCodeAt(tag.length - 1) - ("x".charCodeAt(0));
        assert(target >= 0 && target < 3);
        dispatchToDim.push(target);
      } else if (tag.startsWith("threadIdx.")) {
        const target: number = tag.charCodeAt(tag.length - 1) - ("x".charCodeAt(0));
        assert(target >= 0 && target < 3);
        dispatchToDim.push(target + 3);
      } else {
        throw new Error("Cannot handle thread_axis " + tag);
      }
    }

    const submitShader = (...args: Array<GPUPointer | number>): void => {
      const commandEncoder = this.device.createCommandEncoder();
      const compute = commandEncoder.beginComputePass();
      compute.setPipeline(pipeline);
      const bindGroupEntries: Array<GPUBindGroupEntry> = [];
      assert(args.length == layoutEntries.length + dispatchToDim.length);

      for (let i = 0; i < layoutEntries.length; ++i) {
        bindGroupEntries.push({
          binding: i,
          resource: {
            buffer: this.gpuBufferFromPtr(args[i])
          }
        });
      }

      compute.setBindGroup(0, this.device.createBindGroup({
        layout: bindGroupLayout,
        entries: bindGroupEntries
      }));
      const wl: Array<number> = [1, 1, 1, 1, 1, 1];
      for (let i = 0; i < dispatchToDim.length; ++i) {
        wl[dispatchToDim[i]] = args[layoutEntries.length + i];
      }
      compute.dispatch(wl[0], wl[1], wl[2]);
      compute.endPass();
      const command = commandEncoder.finish();
      this.device.defaultQueue.submit([command]);
    };

    return submitShader;
  }

  /**
   * Get the device API according to its name
   * @param The name of the API.
   * @returns The corresponding device api.
   */
  getDeviceAPI(name: string): Function {
    if (name == "deviceAllocDataSpace") {
      return (nbytes: number): GPUPointer => {
        return this.deviceAllocDataSpace(nbytes);
      };
    } else if (name == "deviceFreeDataSpace") {
      return (ptr: GPUPointer): void => {
        return this.deviceFreeDataSpace(ptr);
      };
    } else if (name == "deviceCopyToGPU") {
      return (
        from: Pointer,
        to: GPUPointer,
        toOffset: number,
        nbytes: number
      ): void => {
        this.deviceCopyToGPU(from, to, toOffset, nbytes);
      };
    } else if (name == "deviceCopyFromGPU") {
      return (
        from: GPUPointer,
        fromOffset: number,
        to: Pointer,
        nbytes: number
      ): void => {
        this.deviceCopyFromGPU(from, fromOffset, to, nbytes);
      };
    } else if (name == "deviceCopyWithinGPU") {
      return (
        from: GPUPointer,
        fromOffset: number,
        to: Pointer,
        toOffset: number,
        nbytes: number
      ): void => {
        this.deviceCopyWithinGPU(from, fromOffset, to, toOffset, nbytes);
      };
    } else {
      throw new Error("Unknown DeviceAPI function " + name);
    }

  }

  // DeviceAPI
  private deviceAllocDataSpace(nbytes: number): GPUPointer {
    const buffer = this.device.createBuffer({
      size: nbytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    return this.attachToBufferTable(buffer);
  }

  private deviceFreeDataSpace(ptr: GPUPointer): void {
    const idx = ptr;
    const buffer = this.bufferTable[idx];
    this.bufferTable[idx] = undefined;
    assert(buffer !== undefined);
    this.bufferTableFreeId.push(idx);
    buffer.destroy();
  }

  private deviceCopyToGPU(
    from: Pointer,
    to: GPUPointer,
    toOffset: number,
    nbytes: number
  ): void {
    // Perhaps it would be more useful to use a staging buffer?
    const gpuTemp = this.device.createBuffer({
      mappedAtCreation: true,
      size: nbytes,
      usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC
    });

    const cpuTemp = gpuTemp.getMappedRange();

    const viewU8 = new Uint8Array(cpuTemp);
    viewU8.set(this.memory.loadRawBytes(from, nbytes));
    gpuTemp.unmap();

    const copyEncoder = this.device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(
      gpuTemp,
      0,
      this.gpuBufferFromPtr(to),
      toOffset,
      nbytes
    );
    const copyCommands = copyEncoder.finish();
    this.device.defaultQueue.submit([copyCommands]);
    gpuTemp.destroy();
  }

  private deviceCopyFromGPU(
    from: GPUPointer,
    fromOffset: number,
    to: Pointer,
    nbytes: number
  ): void {
    // Perhaps it would be more useful to resuse a staging buffer?
    const gpuTemp = this.device.createBuffer({
      size: nbytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const copyEncoder = this.device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(
      this.gpuBufferFromPtr(from),
      fromOffset,
      gpuTemp,
      0,
      nbytes
    );
    const copyCommands = copyEncoder.finish();
    this.device.defaultQueue.submit([copyCommands]);

    this.numPendingReads += 1;

    const readEvent = gpuTemp.mapAsync(GPUMapMode.READ).then((data: unknown) => {
      this.memory.storeRawBytes(to, new Uint8Array(data as ArrayBuffer));
      this.numPendingReads -= 1;
      gpuTemp.destroy();
    });

    if (this.numPendingReads == 1) {
      this.pendingRead = readEvent;
    } else {
      this.pendingRead = Promise.all([
        this.pendingRead,
        readEvent,
        // eslint-disable-next-line @typescript-eslint/no-empty-function
      ]).then(() => {});
    }
  }

  private deviceCopyWithinGPU(
    from: GPUPointer,
    fromOffset: number,
    to: Pointer,
    toOffset: number,
    nbytes: number
  ): void {
    const copyEncoder = this.device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(
      this.gpuBufferFromPtr(from),
      fromOffset,
      this.gpuBufferFromPtr(to),
      toOffset,
      nbytes
    );
    const copyCommands = copyEncoder.finish();
    this.device.defaultQueue.submit([copyCommands]);
  }

  private gpuBufferFromPtr(ptr: GPUPointer): GPUBuffer {
    const buffer = this.bufferTable[ptr];
    assert(buffer !== undefined);
    return buffer;
  }

  private attachToBufferTable(buffer: GPUBuffer): GPUPointer {
    if (this.bufferTableFreeId.length != 0) {
      const idx = this.bufferTableFreeId.pop() as number;
      this.bufferTable[idx] = buffer;
      return idx;
    } else {
      const idx = this.bufferTable.length;
      this.bufferTable.push(buffer);
      return idx;
    }
  }
}
