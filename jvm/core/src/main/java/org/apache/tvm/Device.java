/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.tvm;

import java.util.HashMap;
import java.util.Map;
import org.apache.tvm.rpc.RPC;

public class Device {
  /**
   * Provides the same information as the C++ enums DLDeviceType and
   * TVMDeviceExtType.
   */
  static final int kDLCPU = 1, kDLCUDA = 2, kDLCUDAHost = 3, kDLOpenCL = 4, kDLVulkan = 7,
                   kDLMetal = 8, kDLVPI = 9, kDLROCM = 10, kDLROCMHost = 11, kDLExtDev = 12,
                   kDLCUDAManaged = 13, kDLOneAPI = 14, kDLWebGPU = 15, kDLHexagon = 16,
                   kDLAOCL = 32, kDLSDAccel = 33, kOpenGL = 34, kDLMicroDev = 35;

  private static final Map<Integer, String> MASK2STR = new HashMap<Integer, String>();
  private static final Map<String, Integer> STR2MASK = new HashMap<String, Integer>();

  static {
    MASK2STR.put(kDLCPU, "cpu");
    MASK2STR.put(kDLCUDA, "cuda");
    MASK2STR.put(kDLOpenCL, "opencl");
    MASK2STR.put(kDLVulkan, "vulkan");
    MASK2STR.put(kDLMetal, "metal");
    MASK2STR.put(kDLVPI, "vpi");
    MASK2STR.put(kDLHexagon, "hexagon");

    STR2MASK.put("cpu", kDLCPU);
    STR2MASK.put("cuda", kDLCUDA);
    STR2MASK.put("cl", kDLOpenCL);
    STR2MASK.put("opencl", kDLOpenCL);
    STR2MASK.put("vulkan", kDLVulkan);
    STR2MASK.put("metal", kDLMetal);
    STR2MASK.put("vpi", kDLVPI);
    STR2MASK.put("hexagon", kDLHexagon);
  }

  /**
   * Construct a CPU device.
   * @param devId The device id
   * @return The created device
   */
  public static Device cpu(int devId) {
    return new Device(kDLCPU, devId);
  }

  public static Device cpu() {
    return cpu(0);
  }

  /**
   * Construct a CUDA GPU device.
   * @param devId The device id
   * @return The created device
   */
  public static Device cuda(int devId) {
    return new Device(kDLCUDA, devId);
  }

  public static Device cuda() {
    return cuda(0);
  }

  /**
   * Construct a OpenCL device.
   * @param devId The device id
   * @return The created device
   */
  public static Device opencl(int devId) {
    return new Device(kDLOpenCL, devId);
  }

  public static Device opencl() {
    return opencl(0);
  }

  /**
   * Construct a Vulkan device.
   * @param devId The device id
   * @return The created device
   */
  public static Device vulkan(int devId) {
    return new Device(kDLVulkan, devId);
  }

  public static Device vulkan() {
    return vulkan(0);
  }

  /**
   * Construct a metal device.
   * @param devId The device id
   * @return The created device
   */
  public static Device metal(int devId) {
    return new Device(kDLMetal, devId);
  }

  public static Device metal() {
    return metal(0);
  }

  /**
   * Construct a VPI simulated device.
   * @param devId The device id
   * @return The created device
   */
  public static Device vpi(int devId) {
    return new Device(kDLVPI, devId);
  }

  public static Device vpi() {
    return vpi(0);
  }

  /**
   * Construct a Hexagon device.
   * @param devId The device id
   * @return The created device
   */
  public static Device hexagon(int devId) {
    return new Device(kDLHexagon, devId);
  }

  public static Device hexagon() {
    return hexagon(0);
  }

  public final int deviceType;
  public final int deviceId;

  public Device(int deviceType, int deviceId) {
    this.deviceType = deviceType;
    this.deviceId = deviceId;
  }

  public Device(String deviceType, int deviceId) {
    this(STR2MASK.get(deviceType), deviceId);
  }

  /**
   * Whether this device exists.
   * @return true if exists.
   */
  public boolean exist() {
    TVMValue ret =
        APIInternal.get("_GetDeviceAttr").pushArg(deviceType).pushArg(deviceId).pushArg(0).invoke();
    return ((TVMValueLong) ret).value != 0;
  }

  /**
   * Maximum number of threads on each block.
   * @return the maximum thread number.
   */
  public long maxThreadsPerBlock() {
    TVMValue ret =
        APIInternal.get("_GetDeviceAttr").pushArg(deviceType).pushArg(deviceId).pushArg(1).invoke();
    return ((TVMValueLong) ret).value;
  }

  /**
   * Number of threads that executes in concurrent.
   * @return the thread number.
   */
  public long warpSize() {
    TVMValue ret =
        APIInternal.get("_GetDeviceAttr").pushArg(deviceType).pushArg(deviceId).pushArg(2).invoke();
    return ((TVMValueLong) ret).value;
  }

  /**
   * Synchronize until jobs finished at the device.
   */
  public void sync() {
    Base.checkCall(Base._LIB.tvmSynchronize(deviceType, deviceId));
  }

  @Override
  public int hashCode() {
    return (deviceType << 16) | deviceId;
  }

  @Override
  public boolean equals(Object other) {
    if (other != null && other instanceof Device) {
      Device obj = (Device) other;
      return deviceId == obj.deviceId && deviceType == obj.deviceType;
    }
    return false;
  }

  @Override
  public String toString() {
    if (deviceType >= RPC.RPC_SESS_MASK) {
      int tblId = deviceType / RPC.RPC_SESS_MASK - 1;
      int devType = deviceType % RPC.RPC_SESS_MASK;
      return String.format("remote[%d]:%s(%d)", tblId, MASK2STR.get(devType), deviceId);
    }
    return String.format("%s(%d)", MASK2STR.get(deviceType), deviceId);
  }
}
