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

package org.apache.tvm.contrib;

import org.apache.tvm.Device;
import org.apache.tvm.Function;
import org.apache.tvm.Module;
import org.apache.tvm.NDArray;

/**
 * Wrapper runtime module.
 * This is a thin wrapper of the underlying TVM module.
 * you can also directly call set_input, run, and get_output
 * of underlying module functions.
 */
public class GraphModule {
  private Module module;
  private Device device;

  private Function fsetInput;
  private Function frun;
  private Function fgetOutput;
  private Function fgetInput;
  private Function fdebugGetOutput;
  private Function floadParams;

  public GraphModule(Module module, Device dev) {
    this.module = module;
    this.device = dev;
    fsetInput = module.getFunction("set_input");
    frun = module.getFunction("run");
    fgetInput = module.getFunction("get_input");
    fgetOutput = module.getFunction("get_output");
    try {
      fdebugGetOutput = module.getFunction("debug_get_output");
    } catch (IllegalArgumentException ignored) {
      // ignore
    }
    floadParams = module.getFunction("load_params");
  }

  /**
   * Release the GraphModule.
   * <p>
   * We highly recommend you to do this manually since the GC strategy is lazy.
   * </p>
   */
  public void release() {
    fsetInput.release();
    frun.release();
    fgetInput.release();
    fgetOutput.release();
    if (fdebugGetOutput != null) {
      fdebugGetOutput.release();
    }
    floadParams.release();
    module.release();
  }

  /**
   * Set inputs to the module.
   * @param key The input key.
   * @param value The input value
   * @return self.
   */
  public GraphModule setInput(String key, NDArray value) {
    NDArray input = value;
    if (!value.device().equals(device)) {
      input = NDArray.empty(value.shape(), device);
      value.copyTo(input);
    }
    fsetInput.pushArg(key).pushArg(input).invoke();
    return this;
  }

  /**
   * Set inputs to the module.
   * @param key The input key.
   * @param value The input value.
   * @return self.
   */
  public GraphModule setInput(int key, NDArray value) {
    NDArray input = value;
    if (!value.device().equals(device)) {
      input = NDArray.empty(value.shape(), device);
      value.copyTo(input);
    }
    fsetInput.pushArg(key).pushArg(input).invoke();
    return this;
  }

  /**
   * Run forward execution of the graph.
   * @return self.
   */
  public GraphModule run() {
    frun.invoke();
    return this;
  }

  /**
   * Get index-th input to out.
   * @param index The input index.
   * @param out The output array container.
   * @return out.
   */
  public NDArray getInput(int index, NDArray out) {
    fgetInput.pushArg(index).pushArg(out).invoke();
    return out;
  }

  /**
   * Get index-th output to out.
   * @param index The output index.
   * @param out The output array container.
   * @return out.
   */
  public NDArray getOutput(int index, NDArray out) {
    fgetOutput.pushArg(index).pushArg(out).invoke();
    return out;
  }

  /**
   * Run graph up to node and get the output to out.
   * @param node The node name.
   * @param out The output array container.
   * @return out.
   */
  public NDArray debugGetOutput(String node, NDArray out) {
    if (fdebugGetOutput != null) {
      fdebugGetOutput.pushArg(node).pushArg(out).invoke();
    } else {
      throw new RuntimeException("Please compile runtime with USE_PROFILER = ON");
    }
    return out;
  }

  /**
   * Run graph up to node and get the output to out.
   * @param node The node index.
   * @param out The output array container.
   * @return out.
   */
  public NDArray debugGetOutput(int node, NDArray out) {
    if (fdebugGetOutput != null) {
      fdebugGetOutput.pushArg(node).pushArg(out).invoke();
    } else {
      throw new RuntimeException("Please compile runtime with USE_PROFILER = ON");
    }
    return out;
  }

  /**
   * Load parameters from serialized byte array of parameter dict.
   * @param params The serialized parameter.
   * @return self.
   */
  public GraphModule loadParams(byte[] params) {
    floadParams.pushArg(params).invoke();
    return this;
  }

  /**
   * Get internal module function.
   * @param key The key to the module.
   * @return The function.
   * @throws IllegalArgumentException if function does not exist.
   */
  public Function getFunction(String key) {
    return module.getFunction(key);
  }
}
