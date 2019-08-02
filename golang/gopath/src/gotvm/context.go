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
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm package source for TVMContext interface
 * \file context.go
 */

package gotvm

//#include "gotvm.h"
import "C"

// KDLCPU is golang enum correspond to TVM device type kDLCPU.
var KDLCPU                  = int32(C.kDLCPU)
// KDLGPU is golang enum correspond to TVM device type kDLGPU.
var KDLGPU                  = int32(C.kDLGPU)
// KDLCPUPinned is golang enum correspond to TVM device type kDLCPUPinned.
var KDLCPUPinned            = int32(C.kDLCPUPinned)
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

// Context dtype corresponding to TVMContext aka DLContext
type Context struct {
    DeviceType int32
    DeviceID    int32
}

// CPU returns the Context object for CPU target on given index
func CPU(index int32) Context {
    return Context{KDLCPU, index}
}

// GPU returns the Context object for GPU target on given index
func GPU(index int32) Context {
    return Context{KDLGPU, index}
}

// CPUPinned returns the Context object for CPUPinned target on given index
func CPUPinned(index int32) Context {
    return Context{KDLCPUPinned, index}
}

// OpenCL returns the Context object for OpenCL target on given index
func OpenCL(index int32) Context {
    return Context{KDLOpenCL, index}
}

// Metal returns the Context object for Metal target on given index
func Metal(index int32) Context {
    return Context{KDLMetal, index}
}

// VPI returns the Context object for VPI target on given index
func VPI(index int32) Context {
    return Context{KDLVPI, index}
}

// ROCM returns the Context object for ROCM target on given index
func ROCM(index int32) Context {
    return Context{KDLROCM, index}
}

// SDAccel returns the Context object for SDAccel target on given index
func SDAccel(index int32) Context {
    return Context{KDLSDAccel, index}
}

// Vulkan returns the Context object for Vulkan target on given index
func Vulkan(index int32) Context {
    return Context{KDLVulkan, index}
}

// OpenGL returns the Context object for OpenGL target on given index
func OpenGL(index int32) Context {
    return Context{KOpenGL, index}
}
