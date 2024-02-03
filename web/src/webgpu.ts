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
    const adapter = await navigator.gpu.requestAdapter({"powerPreference":"high-performance"});
    if (adapter == null) {
      throw Error("Cannot find adapter that matches the request");
    }
    const computeMB = (value: number) => {
      return Math.ceil(value  / (1 << 20)) + "MB";
    }

    // more detailed error message
    const requiredMaxBufferSize = 1 << 30;
    if (requiredMaxBufferSize > adapter.limits.maxBufferSize) {
      throw Error(
        `Cannot initialize runtime because of requested maxBufferSize ` +
        `exceeds limit. requested=${computeMB(requiredMaxBufferSize)}, ` +
        `limit=${computeMB(adapter.limits.maxBufferSize)}. ` +
        `This error may be caused by an older version of the browser (e.g. Chrome 112). ` +
        `You can try to upgrade your browser to Chrome 113 or later.`
      );
    }

    let requiredMaxStorageBufferBindingSize = 1 << 30;  // 1GB
    if (requiredMaxStorageBufferBindingSize > adapter.limits.maxStorageBufferBindingSize) {
      // If 1GB is too large, try 128MB (default size for Android)
      const backupRequiredMaxStorageBufferBindingSize = 1 << 27;  // 128MB
      console.log(
        `Requested maxStorageBufferBindingSize exceeds limit. \n` +
        `requested=${computeMB(requiredMaxStorageBufferBindingSize)}, \n` +
        `limit=${computeMB(adapter.limits.maxStorageBufferBindingSize)}. \n` +
        `WARNING: Falling back to ${computeMB(backupRequiredMaxStorageBufferBindingSize)}...`
      );
      requiredMaxStorageBufferBindingSize = backupRequiredMaxStorageBufferBindingSize;
      if (backupRequiredMaxStorageBufferBindingSize > adapter.limits.maxStorageBufferBindingSize) {
        // Fail if 128MB is still too big
        throw Error(
          `Cannot initialize runtime because of requested maxStorageBufferBindingSize ` +
          `exceeds limit. requested=${computeMB(backupRequiredMaxStorageBufferBindingSize)}, ` +
          `limit=${computeMB(adapter.limits.maxStorageBufferBindingSize)}. `
        );
      }
    }

    const requiredMaxComputeWorkgroupStorageSize = 32 << 10;
    if (requiredMaxComputeWorkgroupStorageSize> adapter.limits.maxComputeWorkgroupStorageSize) {
      throw Error(
        `Cannot initialize runtime because of requested maxComputeWorkgroupStorageSize ` +
        `exceeds limit. requested=${requiredMaxComputeWorkgroupStorageSize}, ` +
        `limit=${adapter.limits.maxComputeWorkgroupStorageSize}. `
      );
    }

    const requiredFeatures : GPUFeatureName[] = [];
    // Always require f16 if available
    if (adapter.features.has("shader-f16")) {
      requiredFeatures.push("shader-f16");
    }

    const adapterInfo = await adapter.requestAdapterInfo();
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: requiredMaxBufferSize,
        maxStorageBufferBindingSize: requiredMaxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: requiredMaxComputeWorkgroupStorageSize,
      },
      requiredFeatures
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
class CanvasRenderManager implements Disposable {
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
    // avoid possible ts complain
    this.canvasContext = ctx as any;
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
  private podArgStagingBuffers: Array<GPUBuffer> = [];
  private canvasRenderManager?: CanvasRenderManager = undefined;
  // number of pod arg staging buffers
  private maxNumPodArgsStagingBuffers = 2;
  // flags for debugging
  // stats of the runtime.
  // peak allocation
  private peakAllocatedBytes = 0;
  // current allocation
  private currAllocatedBytes = 0;
  // all allocation(ignoring free)
  private allAllocatedBytes = 0;
  // shader submit counter
  private shaderSubmitCounter = 0;
  // limite number of shaders to be submitted, useful for debugging, default to -1
  protected debugShaderSubmitLimit = -1;
  // log and sync each step
  protected debugLogFinish = false;

  constructor(memory: Memory, device: GPUDevice) {
    this.memory = memory;
    this.device = device;
  }

  /**
   * Dispose context.
   */
  dispose() {
    this.canvasRenderManager?.dispose();
    this.bufferTableFreeId = [];
    while (this.bufferTable.length != 0) {
      this.bufferTable.pop()?.destroy();
    }
    while (this.podArgStagingBuffers.length != 0) {
      this.podArgStagingBuffers.pop()?.destroy();
    }
    this.device.destroy();
  }

  /**
   * Wait for all pending GPU tasks to complete
   */
  async sync(): Promise<void> {
    await this.device.queue.onSubmittedWorkDone();
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
    this.canvasRenderManager = new CanvasRenderManager(this.device, canvas);
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
    return this.createShadeInternal(finfo, code, false) as Function;
  }

  /**
   * Create a PackedFunc that runs the given shader asynchronously
   * via createComputePipelineAsync
   *
   * @param info The function information already parsed as a record.
   * @param code The shader data(in WGSL)
   * @returns The shader
   */
  async createShaderAsync(finfo: FunctionInfo, code: string) : Promise<Function> {
    return await (this.createShadeInternal(finfo, code, true) as Promise<Function>);
  }

  /**
   * Get the pod arg staging buffer
   * \param nbytes The minimum size.
   * \return The allocated buffer
   */
  private getPodArgsBuffer(nbytes: number) : GPUBuffer {
    let buffer : GPUBuffer | undefined = undefined;
    if (this.podArgStagingBuffers.length >= this.maxNumPodArgsStagingBuffers) {
      buffer = this.podArgStagingBuffers.shift();
    }
    // minimum of 16 bytes
    let allocSize = 16;
    if (buffer !== undefined) {
      allocSize = buffer.size;
      if (buffer.size < nbytes) {
        buffer.destroy();
        buffer = undefined;
      }
    }
    while (allocSize < nbytes) {
      allocSize *= 2;
    }

    if (buffer == undefined) {
      // create uniform buffer
      buffer = this.device.createBuffer({
        size: allocSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    assert(nbytes <= buffer.size);
    return buffer;
  }

  /**
   * Internal impl of createShader for both async and sync mode.
   *
   * @param info The function information already parsed as a record.
   * @param code The shader data(in WGSL)
   * @param asyncMode Whether use async mode.
   * @returns The shader function or promise of shader func.
   */
  private createShadeInternal(
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


    const layoutEntries: Array<GPUBindGroupLayoutEntry> = [];
    const bufferArgIndices : Array<number> = [];
    const podArgIndices : Array<number> = [];

    for (let i = 0; i < finfo.arg_types.length; ++i) {
      const dtype = finfo.arg_types[i];
      if (dtype == "handle") {
        layoutEntries.push({
          binding: bufferArgIndices.length,
          visibility: GPUShaderStage.COMPUTE,
          buffer :  {
            type: paramWriteAccess[bufferArgIndices.length] ? "storage" : "read-only-storage"
          }
        });
        bufferArgIndices.push(i);
      } else if (dtype.startsWith("int") || dtype.startsWith("uint") || dtype.startsWith("float")) {
        podArgIndices.push(i);
      } else {
        throw new Error("Cannot handle argument type " + dtype + " in WebGPU shader");
      }
    }

    assert(paramWriteAccess.length == bufferArgIndices.length);
    // POD arguments are pass in the end
    layoutEntries.push({
      binding: bufferArgIndices.length,
      visibility: GPUShaderStage.COMPUTE,
      buffer :  {
        type: "uniform"
      }
    });

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
        const numBufferOrPodArgs = bufferArgIndices.length + podArgIndices.length;

        assert(args.length == numBufferOrPodArgs + dispatchToDim.length);

        const workDim: Array<number> = [1, 1, 1, 1, 1, 1];
        for (let i = 0; i < dispatchToDim.length; ++i) {
          workDim[dispatchToDim[i]] = args[numBufferOrPodArgs + i];
        }

        // get around 65535 restriction of blockIdx.x
        if (workDim[2] != 1) {
          throw Error("WebGPU: blockIdx.z is reserved for internal use");
        }
        const packDimX = workDim[0];
        // spread thinsg out into blockIdx.z
        if (workDim[0] >= (1 << 16)) {
          let wl_x = workDim[0];
          let wl_z = workDim[2];

          while (wl_x >= (1 << 16)) {
            if (wl_x % 2 == 0) {
              wl_x = wl_x / 2;
            } else {
              // pad up
              wl_x = (wl_x + 1) / 2;
            }
            wl_z *= 2;
          }
          workDim[0] = wl_x;
          workDim[2] = wl_z;
          assert(wl_x * wl_z >= packDimX);
        }

        for (let i = 0; i < bufferArgIndices.length; ++i) {
          bindGroupEntries.push({
            binding: i,
            resource: {
              buffer: this.gpuBufferFromPtr(args[bufferArgIndices[i]])
            }
          });
        }

        // push pod buffer
        const sizeOfI32 = 4;
        const podArgBuffer = this.getPodArgsBuffer((podArgIndices.length + 1) * sizeOfI32);
        const i32View = new Int32Array(podArgIndices.length + 1);
        const u32View = new Uint32Array(i32View.buffer);
        const f32View = new Float32Array(i32View.buffer);

        for (let i = 0; i < podArgIndices.length; ++i) {
          const value = args[podArgIndices[i]];
          const dtype = finfo.arg_types[podArgIndices[i]];
          if (dtype.startsWith("int")) {
            i32View[i] = value;
          } else if (dtype.startsWith("uint")) {
            u32View[i] = value;
          } else if (dtype.startsWith("float")) {
            f32View[i] = value;
          } else {
            throw Error("Unknown pod dtype " + dtype);
          }
        }
        // always pass in dim z launching grid size in
        u32View[podArgIndices.length] = packDimX;
        this.device.queue.writeBuffer(podArgBuffer, 0, i32View.buffer);

        bindGroupEntries.push({
          binding: bufferArgIndices.length,
          resource: {
            buffer: podArgBuffer,
            size: i32View.buffer.byteLength
          }
        });

        compute.setBindGroup(0, this.device.createBindGroup({
          layout: bindGroupLayout,
          entries: bindGroupEntries
        }));

        compute.dispatchWorkgroups(workDim[0], workDim[1], workDim[2])
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
      compilationHints: [
        {
          entryPoint: "main",
          layout: pipelineLayout
        }
      ]
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
    // allocate 0 bytes buffer as 1 bytes buffer.
    if (nbytes == 0) {
      nbytes = 1;
    }
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
