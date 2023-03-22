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
import { Disposable } from "./types";

/** A pointer to points to the raw address space. */
export type GPUPointer = number;

export interface GPUDeviceDetectOutput {
  adapter: GPUAdapter;
  adapterInfo: GPUAdapterInfo;
  device: GPUDevice;
}

/**
 * DetectGPU device in the environment.
 */
export async function detectGPUDevice(): Promise<GPUDeviceDetectOutput | undefined> {
  if (typeof navigator !== "undefined" && navigator.gpu !== undefined) {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter == null) {
      throw Error("Cannot find adapter that matches the request");
    }
    const adapterInfo = await adapter.requestAdapterInfo();
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: 1 << 30,
        maxStorageBufferBindingSize: 1 << 30,
        maxComputeWorkgroupStorageSize: 32 << 10,
      }
    });
    return {
      adapter: adapter,
      adapterInfo: adapterInfo,
      device: device
    };
  } else {
    return undefined;
  }
}

const canvasRenderWGSL =`
@group(0) @binding(0) var my_sampler : sampler;
@group(0) @binding(1) var my_texture : texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

@vertex
fn vertex_main(@builtin(vertex_index) vidx : u32) -> VertexOutput {
  const pos = array(
    vec2( 1.0,  1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0, -1.0),
    vec2(-1.0,  1.0),
  );

  const uv = array(
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0),
    vec2(0.0, 0.0),
  );

  var output : VertexOutput;
  output.position = vec4(pos[vidx], 0.0, 1.0);
  output.uv = uv[vidx];
  return output;
}

@fragment
fn fragment_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  return textureSample(my_texture, my_sampler, uv);
}

@fragment
fn fragment_clear(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  return vec4(1.0, 1.0, 1.0, 1.0);
}
`
class CanvaRenderManager implements Disposable {
  private device: GPUDevice;
  private canvasContext: GPUCanvasContext;
  private stagingTexture: GPUTexture;
  private renderSampler: GPUSampler;
  private renderPipeline: GPURenderPipeline;
  private clearPipeline: GPURenderPipeline;
  private canvasTextureFormat: GPUTextureFormat;

  constructor(device: GPUDevice, canvas: HTMLCanvasElement) {
    this.device = device;
    const ctx = canvas.getContext("webgpu");
    if (ctx == null) {
      throw Error("Cannot bind WebGPU context");
    }
    this.canvasContext = ctx;
    this.canvasTextureFormat = navigator.gpu.getPreferredCanvasFormat();
    this.canvasContext.configure({
      device: this.device,
      format: this.canvasTextureFormat,
      alphaMode: "opaque",
    });

    this.renderPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: device.createShaderModule({
          code: canvasRenderWGSL,
        }),
        entryPoint: "vertex_main",
      },
      fragment: {
        module: device.createShaderModule({
          code: canvasRenderWGSL,
        }),
        entryPoint: "fragment_main",
        targets: [{
            format: this.canvasTextureFormat,
        }],
      },
      primitive: {
        topology: "triangle-list",
      },
    });

    this.clearPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: device.createShaderModule({
          code: canvasRenderWGSL,
        }),
        entryPoint: "vertex_main",
      },
      fragment: {
        module: device.createShaderModule({
          code: canvasRenderWGSL,
        }),
        entryPoint: "fragment_clear",
        targets: [{
            format: this.canvasTextureFormat,
        }],
      },
      primitive: {
        topology: "triangle-list",
      },
    });

    this.renderSampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });
    // staging texture always be in RGBA
    this.stagingTexture = device.createTexture({
      size: [canvas.height, canvas.width, 1],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  clear() {
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.canvasContext.getCurrentTexture().createView(),
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    passEncoder.setPipeline(this.clearPipeline);
    const renderBindingGroup = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.renderSampler },
        { binding: 1, resource: this.stagingTexture.createView() },
      ],
    });
    passEncoder.setBindGroup(0, renderBindingGroup);
    passEncoder.draw(6, 1, 0, 0);
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  draw(buffer: GPUBuffer, height: number, width: number) {
    // resize the staging texture
    if (height != this.stagingTexture.height || width != this.stagingTexture.width) {
      this.stagingTexture.destroy();
      this.stagingTexture = this.device.createTexture({
        size: [height, width, 1],
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToTexture({
      buffer: buffer,
      offset: 0,
      bytesPerRow: this.stagingTexture.width * 4
    }, {
      texture: this.stagingTexture
    },{
      width: this.stagingTexture.width,
      height: this.stagingTexture.height
    });

    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.canvasContext.getCurrentTexture().createView(),
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    passEncoder.setPipeline(this.renderPipeline);
    const renderBindingGroup = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.renderSampler },
        { binding: 1, resource: this.stagingTexture.createView() },
      ],
    });
    passEncoder.setBindGroup(0, renderBindingGroup);
    passEncoder.draw(6, 1, 0, 0);
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  dispose() : void {
    this.stagingTexture.destroy();
  }
}

/**
 * Function info from the API
 */
export interface FunctionInfo {
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
  // internal data
  private bufferTable: Array<GPUBuffer | undefined> = [undefined];
  private bufferTableFreeId: Array<number> = [];
  private canvasRenderManager?: CanvaRenderManager = undefined;
  // flags for debugging
  // stats of the runtime.
  // peak allocation
  private peakAllocatedBytes: number = 0;
  // current allocation
  private currAllocatedBytes: number = 0;
  // all allocation(ignoring free)
  private allAllocatedBytes: number = 0;
  // shader submit counter
  private shaderSubmitCounter: number = 0;
  // limite number of shaders to be submitted, useful for debugging, default to -1
  protected debugShaderSubmitLimit: number = -1;
  // log and sync each step
  protected debugLogFinish: boolean = false;

  constructor(memory: Memory, device: GPUDevice) {
    this.memory = memory;
    this.device = device;
  }

  /**
   * Wait for all pending GPU tasks to complete
   */
  async sync(): Promise<void> {
    await this.device.queue.onSubmittedWorkDone();
  }

  /**
   * Dispose the binded canvas.
   */
  disposeCanvas() {
    this.canvasRenderManager?.dispose();
    this.canvasRenderManager = undefined;
  }

  /**
   * Obtain the runtime information in readable format.
   */
  runtimeStatsText(): string {
    let info = "peak-memory=" + Math.ceil(this.peakAllocatedBytes / (1 << 20)) + " MB";
    info += ", all-memory=" + Math.ceil(this.allAllocatedBytes / (1 << 20)) + " MB";
    info += ", shader-submissions=" + this.shaderSubmitCounter;
    return info;
  }

  /**
   * Draw image from data in storage buffer.
   * @param ptr The GPU ptr
   * @param height The height of the image.
   * @param width The width of the image.
   */
  drawImageFromBuffer(ptr: GPUPointer, height: number, width: number) {
    if (this.canvasRenderManager == undefined) {
      throw Error("Do not have a canvas context, call bindCanvas first");
    }
    this.canvasRenderManager.draw(this.gpuBufferFromPtr(ptr), height, width);
  }

  /**
   * Copy raw bytes into buffer ptr.
   *
   * @param rawBytes The raw bytes
   * @param toPtr The target gpu buffer ptr
   * @param toOffset The beginning offset
   * @param nbytes Number of bytes
   */
  copyRawBytesToBuffer(
    rawBytes: Uint8Array,
    toPtr: GPUPointer,
    toOffset: number,
    nbytes: number
  ): void {
    // Perhaps it would be more useful to use a staging buffer?
    this.device.queue.writeBuffer(
      this.gpuBufferFromPtr(toPtr),
      toOffset,
      rawBytes,
      0,
      nbytes
    );
  }
  /**
   * Clear canvas
   */
  clearCanvas() {
    this.canvasRenderManager?.clear();
  }

  /**
   * Bind a canvas element to the runtime.
   * @param canvas The HTML canvas/
   */
  bindCanvas(canvas: HTMLCanvasElement) {
    this.canvasRenderManager = new CanvaRenderManager(this.device, canvas);
  }

  /**
   * Create a PackedFunc that runs the given shader
   * via createComputePipeline
   *
   * @param info The function information already parsed as a record.
   * @param code The shader data(in WGSL)
   * @returns The shader
   */
  createShader(finfo: FunctionInfo, code: string) : Function {
    return this.createShadeInternl(finfo, code, false) as Function;
  }

  /**
   * Create a PackedFunc that runs the given shader asynchrously
   * via createComputePipelineAsync
   *
   * @param info The function information already parsed as a record.
   * @param code The shader data(in WGSL)
   * @returns The shader
   */
  async createShaderAsync(finfo: FunctionInfo, code: string) : Promise<Function> {
    return await (this.createShadeInternl(finfo, code, true) as Promise<Function>);
  }

  /**
   * Internal impl of createShader for both async and sync mode.
   *
   * @param info The function information already parsed as a record.
   * @param code The shader data(in WGSL)
   * @param asyncMode Whether use async mode.
   * @returns The shader function or promise of shader func.
   */
  private createShadeInternl(
    finfo: FunctionInfo,
    code: string,
    asyncMode: boolean
  ): Function | Promise<Function> {
    const dispatchToDim: Array<number> = [];
    let paramWriteAccess: Array<number> = [];

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
      } else if (tag.startsWith("paramWriteAccess:")) {
        paramWriteAccess = JSON.parse(tag.substring(17));
      } else {
        throw new Error("Cannot handle thread_axis " + tag);
      }
    }

    assert(paramWriteAccess.length == finfo.arg_types.length);

    const layoutEntries: Array<GPUBindGroupLayoutEntry> = [];
    for (let i = 0; i < finfo.arg_types.length; ++i) {
      const dtype = finfo.arg_types[i];
      if (dtype == "handle") {
        layoutEntries.push({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer :  {
            type: paramWriteAccess[i] ? "storage" : "read-only-storage"
          }
        });
      } else {
        throw new Error("Cannot handle argument type " + dtype + " in WebGPU shader");
      }
    }
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: layoutEntries
    });
    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [ bindGroupLayout ]
    });

    // Function to create the pipeline.
    const createShaderFunc =  (pipeline: GPUComputePipeline): Function => {
      const submitShader = (...args: Array<GPUPointer | number>): void => {
        if (this.debugShaderSubmitLimit != -1 &&
            this.shaderSubmitCounter >= this.debugShaderSubmitLimit) {
          this.shaderSubmitCounter += 1;
          return;
        }

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

        // get around 65535 restriction of blockIdx.x
        if (wl[2] != 1) {
          throw Error("WebGPU: blockIdx.z is reserved for internal use");
        }
        // spread thinsg out into blockIdx.z
        if (wl[0] >= (1 << 16)) {
          let wl_x = wl[0];
          let wl_z = wl[2];

          while (wl_x >= (1 << 16)) {
            if (wl_x % 2 != 0) {
              throw Error("WebGPU: cannot factorize big gridDim.x=" + wl[0].toString());
            }
            wl_x /= 2;
            wl_z *= 2;
          }
          wl[0] = wl_x;
          wl[2] = wl_z;
        }
        compute.dispatchWorkgroups(wl[0], wl[1], wl[2])
        compute.end()
        const command = commandEncoder.finish();
        this.device.queue.submit([command]);

        if (this.debugLogFinish) {
          const currCounter = this.shaderSubmitCounter;
          this.device.queue.onSubmittedWorkDone().then(()=> {
            console.log("["+ currCounter + "][Debug] finish shader" + finfo.name);
          });
        }
        this.shaderSubmitCounter += 1;
      };
      return submitShader;
    };

    const shaderModule = this.device.createShaderModule({
      code: code,
      hints: {
        main: {
          layout: pipelineLayout
        }
      }
    });

    if (asyncMode) {
      return this.device.createComputePipelineAsync({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: finfo.name
        }
      }).then((pipeline: GPUComputePipeline) => {
        return createShaderFunc(pipeline);
      });
    } else {
      const pipeline = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: finfo.name
        }
      });
      return createShaderFunc(pipeline);
    }
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
    this.currAllocatedBytes += nbytes;
    this.allAllocatedBytes += nbytes;
    if (this.currAllocatedBytes > this.peakAllocatedBytes) {
      this.peakAllocatedBytes = this.currAllocatedBytes;
    }
    const ptr = this.attachToBufferTable(buffer);
    return ptr;
  }

  private deviceFreeDataSpace(ptr: GPUPointer): void {
    const idx = ptr;
    const buffer = this.bufferTable[idx];
    this.bufferTable[idx] = undefined;
    assert(buffer !== undefined);
    this.bufferTableFreeId.push(idx);
    this.currAllocatedBytes -= buffer.size;
    buffer.destroy();
  }

  private deviceCopyToGPU(
    from: Pointer,
    to: GPUPointer,
    toOffset: number,
    nbytes: number
  ): void {
    // Perhaps it would be more useful to use a staging buffer?
    const rawBytes = this.memory.loadRawBytes(from, nbytes);
    this.device.queue.writeBuffer(
      this.gpuBufferFromPtr(to),
      toOffset,
      rawBytes,
      0,
      nbytes
    );
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
    this.device.queue.submit([copyCommands]);

    gpuTemp.mapAsync(GPUMapMode.READ).then(() => {
      const data = gpuTemp.getMappedRange();
      this.memory.storeRawBytes(to, new Uint8Array(data));
      gpuTemp.destroy();
    });
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
    this.device.queue.submit([copyCommands]);
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
