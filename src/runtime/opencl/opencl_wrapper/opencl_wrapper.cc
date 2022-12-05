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
 * \file opencl_wrapper.cc
 * \brief This wrapper is actual for OpenCL 1.2, but can be easily upgraded
 * when TVM will use newer version of OpenCL
 */

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <CL/cl_gl.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>
#include <tvm/runtime/logging.h>

#include <vector>

namespace {
#if defined(__APPLE__) || defined(__MACOSX)
static const std::vector<const char*> default_so_paths = {
    "libOpenCL.so", "/System/Library/Frameworks/OpenCL.framework/OpenCL"};
#elif defined(__ANDROID__)
static const std::vector<const char*> default_so_paths = {
    "libOpenCL.so",
    "/system/lib64/libOpenCL.so",
    "/system/vendor/lib64/libOpenCL.so",
    "/system/vendor/lib64/egl/libGLES_mali.so",
    "/system/vendor/lib64/libPVROCL.so",
    "/data/data/org.pocl.libs/files/lib64/libpocl.so",
    "/system/lib/libOpenCL.so",
    "/system/vendor/lib/libOpenCL.so",
    "/system/vendor/lib/egl/libGLES_mali.so",
    "/system/vendor/lib/libPVROCL.so",
    "/data/data/org.pocl.libs/files/lib/libpocl.so"};
#elif defined(_WIN32)
static const std::vector<const TCHAR*> default_so_paths = {__TEXT("OpenCL.dll")};
#elif defined(__linux__)
static const std::vector<const char*> default_so_paths = {"libOpenCL.so",
                                                          "/usr/lib/libOpenCL.so",
                                                          "/usr/local/lib/libOpenCL.so",
                                                          "/usr/local/lib/libpocl.so",
                                                          "/usr/lib64/libOpenCL.so",
                                                          "/usr/lib32/libOpenCL.so"};
#endif

class LibOpenCLWrapper {
 public:
  static LibOpenCLWrapper& getInstance() {
    static LibOpenCLWrapper instance;
    return instance;
  }
  LibOpenCLWrapper(const LibOpenCLWrapper&) = delete;
  LibOpenCLWrapper& operator=(const LibOpenCLWrapper&) = delete;
  void* getOpenCLFunction(const char* funcName) {
    if (m_libHandler == nullptr) openLibOpenCL();
#if defined(_WIN32)
    return GetProcAddress(m_libHandler, funcName);
#else
    return dlsym(m_libHandler, funcName);
#endif
  }

 private:
  LibOpenCLWrapper() {}
  ~LibOpenCLWrapper() {
#if defined(_WIN32)
    if (m_libHandler) FreeLibrary(m_libHandler);
#else
    if (m_libHandler) dlclose(m_libHandler);
#endif
  }
  void openLibOpenCL() {
    for (const auto it : default_so_paths) {
#if defined(_WIN32)
      m_libHandler = LoadLibrary(it);
#else
      m_libHandler = dlopen(it, RTLD_LAZY);
#endif
      if (m_libHandler != nullptr) return;
    }
    ICHECK(m_libHandler != nullptr) << "Error! Cannot open libOpenCL!";
  }

 private:
#if defined(_WIN32)
  HMODULE m_libHandler = nullptr;
#else
  void* m_libHandler = nullptr;
#endif
};

// Function pointers declaration
using f_pfn_notify = void (*)(const char*, const void*, size_t, void*);
using f_clGetPlatformIDs = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
using f_clGetPlatformInfo = cl_int (*)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
using f_clGetDeviceIDs = cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*,
                                    cl_uint*);
using f_clGetDeviceInfo = cl_int (*)(cl_device_id, cl_device_info, size_t, void*, size_t*);
using f_clCreateContext = cl_context (*)(const cl_context_properties*, cl_uint, const cl_device_id*,
                                         f_pfn_notify, void*, cl_int*);
using f_clReleaseContext = cl_int (*)(cl_context);
using f_clReleaseCommandQueue = cl_int (*)(cl_command_queue);
using f_clGetCommandQueueInfo = cl_int (*)(cl_command_queue, cl_command_queue_info, size_t, void*,
                                           size_t*);
using f_clCreateBuffer = cl_mem (*)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
using f_clCreateImage = cl_mem (*)(cl_context, cl_mem_flags, const cl_image_format*,
                                   const cl_image_desc*, void*, cl_int*);
using f_clReleaseMemObject = cl_int (*)(cl_mem);
using f_clCreateProgramWithSource = cl_program (*)(cl_context, cl_uint, const char**, const size_t*,
                                                   cl_int*);
using f_clCreateProgramWithBinary = cl_program (*)(cl_context, cl_uint, const cl_device_id*,
                                                   const size_t*, const unsigned char**, cl_int*,
                                                   cl_int*);
using f_clReleaseProgram = cl_int (*)(cl_program);
using f_clBuildProgram = cl_int (*)(cl_program, cl_uint, const cl_device_id*, const char*,
                                    void (*pfn_notify)(cl_program program, void* user_data), void*);
using f_clGetProgramBuildInfo = cl_int (*)(cl_program, cl_device_id, cl_program_build_info, size_t,
                                           void*, size_t*);
using f_clCreateKernel = cl_kernel (*)(cl_program, const char*, cl_int*);
using f_clReleaseKernel = cl_int (*)(cl_kernel);
using f_clSetKernelArg = cl_int (*)(cl_kernel, cl_uint, size_t, const void*);
using f_clWaitForEvents = cl_int (*)(cl_uint, const cl_event*);
using f_clCreateUserEvent = cl_event (*)(cl_context, cl_int*);
using f_clGetEventProfilingInfo = cl_int (*)(cl_event, cl_profiling_info, size_t, void*, size_t*);
using f_clFlush = cl_int (*)(cl_command_queue);
using f_clFinish = cl_int (*)(cl_command_queue);
using f_clEnqueueReadBuffer = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*,
                                         cl_uint, const cl_event*, cl_event*);
using f_clEnqueueWriteBuffer = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                                          const void*, cl_uint, const cl_event*, cl_event*);
using f_clEnqueueCopyBuffer = cl_int (*)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t,
                                         cl_uint, const cl_event*, cl_event*);
using f_clEnqueueReadImage = cl_int (*)(cl_command_queue, cl_mem, cl_bool, const size_t*,
                                        const size_t*, size_t, size_t, void*, cl_uint,
                                        const cl_event*, cl_event*);
using f_clEnqueueWriteImage = cl_int (*)(cl_command_queue, cl_mem, cl_bool, const size_t*,
                                         const size_t*, size_t, size_t, const void*, cl_uint,
                                         const cl_event*, cl_event*);
using f_clEnqueueCopyImage = cl_int (*)(cl_command_queue, cl_mem, cl_mem, const size_t*,
                                        const size_t*, const size_t*, cl_uint, const cl_event*,
                                        cl_event*);
using f_clEnqueueCopyImageToBuffer = cl_int (*)(cl_command_queue, cl_mem, cl_mem, const size_t*,
                                                const size_t*, size_t, cl_uint, const cl_event*,
                                                cl_event*);
using f_clEnqueueCopyBufferToImage = cl_int (*)(cl_command_queue, cl_mem, cl_mem, size_t,
                                                const size_t*, const size_t*, cl_uint,
                                                const cl_event*, cl_event*);
using f_clEnqueueNDRangeKernel = cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t*,
                                            const size_t*, const size_t*, cl_uint, const cl_event*,
                                            cl_event*);
using f_clCreateCommandQueue = cl_command_queue (*)(cl_context, cl_device_id,
                                                    cl_command_queue_properties, cl_int*);
}  // namespace

cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clGetPlatformIDs)lib.getOpenCLFunction("clGetPlatformIDs");
  if (func) {
    return func(num_entries, platforms, num_platforms);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name,
                         size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clGetPlatformInfo)lib.getOpenCLFunction("clGetPlatformInfo");
  if (func) {
    return func(platform, param_name, param_value_size, param_value, param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries,
                      cl_device_id* devices, cl_uint* num_devices) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clGetDeviceIDs)lib.getOpenCLFunction("clGetDeviceIDs");
  if (func) {
    return func(platform, device_type, num_entries, devices, num_devices);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size,
                       void* param_value, size_t* param_value_size_ret) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clGetDeviceInfo)lib.getOpenCLFunction("clGetDeviceInfo");
  if (func) {
    return func(device, param_name, param_value_size, param_value, param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_context clCreateContext(const cl_context_properties* properties, cl_uint num_devices,
                           const cl_device_id* devices,
                           void (*pfn_notify)(const char*, const void*, size_t, void*),
                           void* user_data, cl_int* errcode_ret) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clCreateContext)lib.getOpenCLFunction("clCreateContext");
  if (func) {
    return func(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
  } else {
    return nullptr;
  }
}

cl_int clReleaseContext(cl_context context) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clReleaseContext)lib.getOpenCLFunction("clReleaseContext");

  if (func) {
    return func(context);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clReleaseCommandQueue)lib.getOpenCLFunction("clReleaseCommandQueue");
  if (func) {
    return func(command_queue);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clGetCommandQueueInfo(cl_command_queue command_queue, cl_command_queue_info param_name,
                             size_t param_value_size, void* param_value,
                             size_t* param_value_size_ret) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clGetCommandQueueInfo)lib.getOpenCLFunction("clGetCommandQueueInfo");
  if (func) {
    return func(command_queue, param_name, param_value_size, param_value, param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr,
                      cl_int* errcode_ret) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clCreateBuffer)lib.getOpenCLFunction("clCreateBuffer");
  if (func) {
    return func(context, flags, size, host_ptr, errcode_ret);
  } else {
    return nullptr;
  }
}

cl_mem clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format* image_format,
                     const cl_image_desc* image_desc, void* host_ptr, cl_int* errcode_ret) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clCreateImage)lib.getOpenCLFunction("clCreateImage");
  if (func) {
    return func(context, flags, image_format, image_desc, host_ptr, errcode_ret);
  } else {
    return nullptr;
  }
}

cl_int clReleaseMemObject(cl_mem memobj) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clReleaseMemObject)lib.getOpenCLFunction("clReleaseMemObject");
  if (func) {
    return func(memobj);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char** strings,
                                     const size_t* lengths, cl_int* errcode_ret) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clCreateProgramWithSource)lib.getOpenCLFunction("clCreateProgramWithSource");
  if (func) {
    return func(context, count, strings, lengths, errcode_ret);
  } else {
    return nullptr;
  }
}

cl_program clCreateProgramWithBinary(cl_context context, cl_uint num_devices,
                                     const cl_device_id* device_list, const size_t* lengths,
                                     const unsigned char** binaries, cl_int* binary_status,
                                     cl_int* errcode_ret) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clCreateProgramWithBinary)lib.getOpenCLFunction("clCreateProgramWithBinary");
  if (func) {
    return func(context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
  } else {
    return nullptr;
  }
}

cl_int clReleaseProgram(cl_program program) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clReleaseProgram)lib.getOpenCLFunction("clReleaseProgram");
  if (func) {
    return func(program);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id* device_list,
                      const char* options, void (*pfn_notify)(cl_program program, void* user_data),
                      void* user_data) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clBuildProgram)lib.getOpenCLFunction("clBuildProgram");
  if (func) {
    return func(program, num_devices, device_list, options, pfn_notify, user_data);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device,
                             cl_program_build_info param_name, size_t param_value_size,
                             void* param_value, size_t* param_value_size_ret) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clGetProgramBuildInfo)lib.getOpenCLFunction("clGetProgramBuildInfo");
  if (func) {
    return func(program, device, param_name, param_value_size, param_value, param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_kernel clCreateKernel(cl_program program, const char* kernel_name, cl_int* errcode_ret) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clCreateKernel)lib.getOpenCLFunction("clCreateKernel");
  if (func) {
    return func(program, kernel_name, errcode_ret);
  } else {
    return nullptr;
  }
}

cl_int clReleaseKernel(cl_kernel kernel) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clReleaseKernel)lib.getOpenCLFunction("clReleaseKernel");
  if (func) {
    return func(kernel);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clSetKernelArg)lib.getOpenCLFunction("clSetKernelArg");
  if (func) {
    return func(kernel, arg_index, arg_size, arg_value);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clWaitForEvents(cl_uint num_events, const cl_event* event_list) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clWaitForEvents)lib.getOpenCLFunction("clWaitForEvents");
  if (func) {
    return func(num_events, event_list);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_event clCreateUserEvent(cl_context context, cl_int* errcode_ret) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clCreateUserEvent)lib.getOpenCLFunction("clCreateUserEvent");
  if (func) {
    return func(context, errcode_ret);
  } else {
    return nullptr;
  }
}

cl_int clGetEventProfilingInfo(cl_event event, cl_profiling_info param_name,
                               size_t param_value_size, void* param_value,
                               size_t* param_value_size_ret) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clGetEventProfilingInfo)lib.getOpenCLFunction("clGetEventProfilingInfo");
  if (func) {
    return func(event, param_name, param_value_size, param_value, param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clFlush(cl_command_queue command_queue) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clFlush)lib.getOpenCLFunction("clFlush");
  if (func) {
    return func(command_queue);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clFinish(cl_command_queue command_queue) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clFinish)lib.getOpenCLFunction("clFinish");
  if (func) {
    return func(command_queue);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read,
                           size_t offset, size_t size, void* ptr, cl_uint num_events_in_wait_list,
                           const cl_event* event_wait_list, cl_event* event) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clEnqueueReadBuffer)lib.getOpenCLFunction("clEnqueueReadBuffer");
  if (func) {
    return func(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list,
                event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write,
                            size_t offset, size_t size, const void* ptr,
                            cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                            cl_event* event) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clEnqueueWriteBuffer)lib.getOpenCLFunction("clEnqueueWriteBuffer");
  if (func) {
    return func(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list,
                event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clEnqueueCopyBuffer(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer,
                           size_t src_offset, size_t dst_offset, size_t size,
                           cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                           cl_event* event) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clEnqueueCopyBuffer)lib.getOpenCLFunction("clEnqueueCopyBuffer");
  if (func) {
    return func(command_queue, src_buffer, dst_buffer, src_offset, dst_offset, size,
                num_events_in_wait_list, event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clEnqueueReadImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_read,
                          const size_t* origin, const size_t* region, size_t row_pitch,
                          size_t slice_pitch, void* ptr, cl_uint num_events_in_wait_list,
                          const cl_event* event_wait_list, cl_event* event) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clEnqueueReadImage)lib.getOpenCLFunction("clEnqueueReadImage");
  if (func) {
    return func(command_queue, image, blocking_read, origin, region, row_pitch, slice_pitch, ptr,
                num_events_in_wait_list, event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clEnqueueWriteImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_write,
                           const size_t* origin, const size_t* region, size_t input_row_pitch,
                           size_t input_slice_pitch, const void* ptr,
                           cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                           cl_event* event) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clEnqueueWriteImage)lib.getOpenCLFunction("clEnqueueWriteImage");
  if (func) {
    return func(command_queue, image, blocking_write, origin, region, input_row_pitch,
                input_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clEnqueueCopyImage(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_image,
                          const size_t* src_origin, const size_t* dst_origin, const size_t* region,
                          cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                          cl_event* event) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clEnqueueCopyImage)lib.getOpenCLFunction("clEnqueueCopyImage");
  if (func) {
    return func(command_queue, src_image, dst_image, src_origin, dst_origin, region,
                num_events_in_wait_list, event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clEnqueueCopyImageToBuffer(cl_command_queue command_queue, cl_mem src_image,
                                  cl_mem dst_buffer, const size_t* src_origin, const size_t* region,
                                  size_t dst_offset, cl_uint num_events_in_wait_list,
                                  const cl_event* event_wait_list, cl_event* event) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clEnqueueCopyImageToBuffer)lib.getOpenCLFunction("clEnqueueCopyImageToBuffer");
  if (func) {
    return func(command_queue, src_image, dst_buffer, src_origin, region, dst_offset,
                num_events_in_wait_list, event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clEnqueueCopyBufferToImage(cl_command_queue command_queue, cl_mem src_buffer,
                                  cl_mem dst_image, size_t src_offset, const size_t* dst_origin,
                                  const size_t* region, cl_uint num_events_in_wait_list,
                                  const cl_event* event_wait_list, cl_event* event) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clEnqueueCopyBufferToImage)lib.getOpenCLFunction("clEnqueueCopyBufferToImage");
  if (func) {
    return func(command_queue, src_buffer, dst_image, src_offset, dst_origin, region,
                num_events_in_wait_list, event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
                              const size_t* global_work_offset, const size_t* global_work_size,
                              const size_t* local_work_size, cl_uint num_events_in_wait_list,
                              const cl_event* event_wait_list, cl_event* event) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clEnqueueNDRangeKernel)lib.getOpenCLFunction("clEnqueueNDRangeKernel");
  if (func) {
    return func(command_queue, kernel, work_dim, global_work_offset, global_work_size,
                local_work_size, num_events_in_wait_list, event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device,
                                      cl_command_queue_properties properties, cl_int* errcode_ret) {
  auto& lib = LibOpenCLWrapper::getInstance();
  auto func = (f_clCreateCommandQueue)lib.getOpenCLFunction("clCreateCommandQueue");
  if (func) {
    return func(context, device, properties, errcode_ret);
  } else {
    return nullptr;
  }
}
