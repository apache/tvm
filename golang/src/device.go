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

/*!
 * \brief gotvm package source for Device interface
 * \file device.go
 */

package gotvm

//#include "gotvm.h"
import "C"

// KDLCPU is golang enum correspond to TVM device type kDLCPU.
var KDLCPU                  = int32(C.kDLCPU)
// kDLCUDA is golang enum correspond to TVM device type kDLCUDA.
var kDLCUDA                  = int32(C.kDLCUDA)
// kDLCUDAHost is golang enum correspond to TVM device type kDLCUDAHost.
var kDLCUDAHost            = int32(C.kDLCUDAHost)
// KDLOpenCL is golang enum correspond to TVM device type kDLOpenCL.
var KDLOpenCL               = int32(C.kDLOpenCL)
// KDLMetal is golang enum correspond to TVM device type kDLMetal.
var KDLMetal                = int32(C.kDLMetal)
// KDLVPI is golang enum correspond to TVM device type kDLVPI.
var KDLVPI                  = int32(C.kDLVPI)
// KDLROCM is golang enum correspond to TVM device type kDLROCM.
var KDLROCM                 = int32(C.kDLROCM)
// KDLSDAccel is golang enum correspond to TVM device type kDLSDAccel.
var KDLSDAccel              = int32(C.kDLSDAccel)
// KDLVulkan is golang enum correspond to TVM device type kDLVulkan.
var KDLVulkan               = int32(C.kDLVulkan)
// KOpenGL is golang enum correspond to TVM device type kOpenGL.
var KOpenGL                 = int32(C.kOpenGL)
// KExtDev is golang enum correspond to TVM device type kDLExtDev.
var KExtDev                 = int32(C.kDLExtDev)

// Device dtype corresponding to Device aka DLDevice
type Device struct {
    DeviceType int32
    DeviceID    int32
}

// CPU returns the Device object for CPU target on given index
func CPU(index int32) Device {
    return Device{KDLCPU, index}
}

// CUDA returns the Device object for CUDA target on given index
func CUDA(index int32) Device {
    return Device{kDLCUDA, index}
}

// CUDAHost returns the Device object for CUDAHost target on given index
func CUDAHost(index int32) Device {
    return Device{kDLCUDAHost, index}
}

// OpenCL returns the Device object for OpenCL target on given index
func OpenCL(index int32) Device {
    return Device{KDLOpenCL, index}
}

// Metal returns the Device object for Metal target on given index
func Metal(index int32) Device {
    return Device{KDLMetal, index}
}

// VPI returns the Device object for VPI target on given index
func VPI(index int32) Device {
    return Device{KDLVPI, index}
}

// ROCM returns the Device object for ROCM target on given index
func ROCM(index int32) Device {
    return Device{KDLROCM, index}
}

// SDAccel returns the Device object for SDAccel target on given index
func SDAccel(index int32) Device {
    return Device{KDLSDAccel, index}
}

// Vulkan returns the Device object for Vulkan target on given index
func Vulkan(index int32) Device {
    return Device{KDLVulkan, index}
}

// OpenGL returns the Device object for OpenGL target on given index
func OpenGL(index int32) Device {
    return Device{KOpenGL, index}
}
