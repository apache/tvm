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
 * \file opencl_common.h
 * \brief OpenCL common header
 */
#ifndef TVM_RUNTIME_OPENCL_OPENCL_COMMON_H_
#define TVM_RUNTIME_OPENCL_OPENCL_COMMON_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/profiling.h>

/* There are many OpenCL platforms that do not yet support OpenCL 2.0,
 * hence we use 1.2 APIs, some of which are now deprecated.  In order
 * to turn off the deprecation warnings (elevated to errors by
 * -Werror) we explicitly disable the 1.2 deprecation warnings.
 *
 * At the point TVM supports minimum version 2.0, we can remove this
 * define.
 */
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

/* Newer releases of OpenCL header files (after May 2018) work with
 * any OpenCL version, with an application's target version
 * specified. Setting the target version disables APIs from after that
 * version, and sets appropriate USE_DEPRECATED macros.  The above
 * macro for CL_USE_DEPRECATED_OPENCL_1_2_APIS is still needed in case
 * we are compiling against the earlier version-specific OpenCL header
 * files.  This also allows us to expose the OpenCL version through
 * tvm.runtime.Device.
 */
#define CL_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../file_utils.h"
#include "../meta_data.h"
#include "../pack_args.h"
#include "../texture.h"
#include "../thread_storage_scope.h"
#include "../workspace_pool.h"

namespace tvm {
namespace runtime {
namespace cl {

static_assert(sizeof(cl_mem) == sizeof(void*), "Required to store cl_mem inside void*");

inline const char* CLGetErrorString(cl_int error) {
  switch (error) {
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
      return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
      return "CL_MAP_FAILURE";
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
      return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
      return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
      return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
      return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
      return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
      return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
      return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
      return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
      return "CL_INVALID_MIP_LEVEL";
    default:
      return "Unknown OpenCL error code";
  }
}

inline cl_channel_type DTypeToOpenCLChannelType(DLDataType data_type) {
  DataType dtype(data_type);
  if (dtype == DataType::Float(32)) {
    return CL_FLOAT;
  } else if (dtype == DataType::Float(16)) {
    return CL_HALF_FLOAT;
  } else if (dtype == DataType::Int(8)) {
    return CL_SIGNED_INT8;
  } else if (dtype == DataType::Int(16)) {
    return CL_SIGNED_INT16;
  } else if (dtype == DataType::Int(32)) {
    return CL_SIGNED_INT32;
  } else if (dtype == DataType::UInt(8)) {
    return CL_UNSIGNED_INT8;
  } else if (dtype == DataType::UInt(16)) {
    return CL_UNSIGNED_INT16;
  } else if (dtype == DataType::UInt(32)) {
    return CL_UNSIGNED_INT32;
  }
  LOG(FATAL) << "data type is not supported in OpenCL runtime yet: " << dtype;
}

/*!
 * \brief Protected OpenCL call
 * \param func Expression to call.
 */
#define OPENCL_CHECK_ERROR(e) \
  { ICHECK(e == CL_SUCCESS) << "OpenCL Error, code=" << e << ": " << cl::CLGetErrorString(e); }

#define OPENCL_CALL(func)  \
  {                        \
    cl_int e = (func);     \
    OPENCL_CHECK_ERROR(e); \
  }

class OpenCLThreadEntry;
struct BufferDescriptor;

/*!
 * \brief Process global OpenCL workspace.
 */
class OpenCLWorkspace : public DeviceAPI {
 public:
  // type key
  std::string type_key{"opencl"};
  // available platforms
  std::vector<cl_platform_id> platform_ids;
  // map platform to its context
  std::unordered_map<cl_platform_id, cl_context> contexts;
  // whether the workspace it initialized.
  bool initialized_{false};
  // map device to platform
  std::unordered_map<cl_device_id, cl_platform_id> device_to_platform;
  // the devices
  std::vector<cl_device_id> devices;
  // the queues
  std::vector<cl_command_queue> queues;
  // the events
  std::vector<std::vector<cl_event>> events;
  // Number of registered kernels
  // Used to register kernel into the workspace.
  size_t num_registered_kernels{0};
  // The version counter, used
  size_t timestamp{0};
  // Ids that are freed by kernels.
  std::vector<size_t> free_kernel_ids;
  // the mutex for initialization
  std::mutex mu;
  // destructor
  ~OpenCLWorkspace() {
    for (auto& it : contexts) {
      OPENCL_CALL(clReleaseContext(it.second));
    }
  }
  // Initialize the device.
  void Init(const std::string& type_key, const std::string& device_type,
            const std::string& platform_name = "");
  virtual void Init() { Init(this->type_key, "gpu"); }
  // Check whether the context is OpenCL or not.
  virtual bool IsOpenCLDevice(Device dev) { return dev.device_type == kDLOpenCL; }
  // get the queue of the device
  cl_command_queue GetQueue(Device dev) {
    ICHECK(IsOpenCLDevice(dev));
    this->Init();
    ICHECK(dev.device_id >= 0 && static_cast<size_t>(dev.device_id) < queues.size())
        << "Invalid OpenCL device_id=" << dev.device_id << ". " << GetError();
    return queues[dev.device_id];
  }
  // get the event queue of the context
  std::vector<cl_event>& GetEventQueue(Device dev) {
    ICHECK(IsOpenCLDevice(dev));
    this->Init();
    ICHECK(dev.device_id >= 0 && static_cast<size_t>(dev.device_id) < queues.size())
        << "Invalid OpenCL device_id=" << dev.device_id << ". " << GetError();
    return events[dev.device_id];
  }
  // is current clCommandQueue in profiling mode
  bool IsProfiling(Device dev) {
    cl_command_queue queue = GetQueue(dev);
    cl_command_queue_properties prop;

    OPENCL_CALL(clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES,
                                      sizeof(cl_command_queue_properties), &prop, nullptr));

    return prop & CL_QUEUE_PROFILING_ENABLE;
  }
  // Check if the device is present or not
  bool IsDeviceExists(unsigned int device_id) { return device_id < devices.size(); }
  // Enable queue profiling, recreate if required
  void EnableQueueProfiling(Device dev, bool enable) {
    bool is_enabled = cl::OpenCLWorkspace::Global()->IsProfiling(dev);
    if (is_enabled == enable) {
      return;
    }
    cl_command_queue_properties prop = (enable) ? CL_QUEUE_PROFILING_ENABLE : 0;
    auto queue = cl::OpenCLWorkspace::Global()->GetQueue(dev);
    OPENCL_CALL(clFlush(queue));
    OPENCL_CALL(clFinish(queue));
    OPENCL_CALL(clReleaseCommandQueue(queue));
    cl_int err_code;
    cl_device_id did = cl::OpenCLWorkspace::Global()->GetCLDeviceID(dev.device_id);
    cl_platform_id platform = cl::OpenCLWorkspace::Global()->device_to_platform[did];
    auto profiling_queue = clCreateCommandQueue(cl::OpenCLWorkspace::Global()->contexts[platform],
                                                did, prop, &err_code);
    OPENCL_CHECK_ERROR(err_code);
    cl::OpenCLWorkspace::Global()->queues[dev.device_id] = profiling_queue;
  }

  cl_device_id GetCLDeviceID(int device_id);
  // override device API
  void SetDevice(Device dev) final;
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final;
  void* AllocDataSpace(Device dev, size_t size, size_t alignment, DLDataType type_hint) final;
  void* AllocDataSpace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                       Optional<String> mem_scope = NullOpt) final;
  void* GetNativePtr(const tvm::runtime::NDArray& narr);
  void FreeDataSpace(Device dev, void* ptr) final;
  void StreamSync(Device dev, TVMStreamHandle stream) final;
  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final;
  void FreeWorkspace(Device dev, void* data) final;

  // Texture (image2d_t) alloca APIs
  cl_mem AllocTexture(Device dev, size_t width, size_t height, DLDataType type_hint);
  void* AllocTextureWorkspace(Device dev, size_t width, size_t height, DLDataType type_hint);
  void FreeTextureWorkspace(Device dev, void* data);

  /*!
   * \brief Get the thread local ThreadEntry
   */
  virtual OpenCLThreadEntry* GetThreadEntry();

  // get the global workspace
  static OpenCLWorkspace* Global();

  void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) final;

  void* CreateHostPtrIfEnabled(BufferDescriptor* desc, Device dev, size_t size);

 private:
  std::string GetError() {
    if (this->devices.size() == 0) return noDevicesErrorMsg;
    return "";
  }
  std::string noDevicesErrorMsg = "";
};

/*! \brief Thread local workspace */
class OpenCLThreadEntry {
 public:
  // The kernel entry and version.
  struct KTEntry {
    // The kernel handle.
    cl_kernel kernel{nullptr};
    // timestamp used to recognize stale kernel
    size_t version{0};
  };
  /*! \brief The current device */
  Device device;
  /*! \brief The thread-local kernel table */
  std::vector<KTEntry> kernel_table;
  /*! \brief workspace pool */
  WorkspacePool pool;
  /*! \brief texture pool */
  TexturePool texture_pool;
  // constructor
  OpenCLThreadEntry(DLDeviceType device_type, DeviceAPI* device_api)
      : pool(device_type, device_api), texture_pool(device_type, device_api) {
    device.device_id = 0;
    device.device_type = device_type;
  }
  OpenCLThreadEntry() : OpenCLThreadEntry(kDLOpenCL, OpenCLWorkspace::Global()) {}

  // get the global workspace
  static OpenCLThreadEntry* ThreadLocal();
};

/*! \brief OpenCL runtime buffer structure with tracked memory layout
    TODO(tvm-team): Uncouple use of storage scope and data layout by using the transform_layout
    schedule primitive to express the desired texture layout. This will require supporting Nd
    indices in BufferLoad and BufferStore in CodegenOpenCL, and ensuring Nd allocations for
    texture are correctly routed to the AllocateTexture packed function in the OpenCL DeviceAPI.
*/
struct BufferDescriptor {
  enum class MemoryLayout {
    /*! \brief One dimensional buffer in row-major layout*/
    kBuffer1D,
    /*! \brief Two dimensional texture w/ width = axis[-1]
     *          e.g. image2d[height=NCH, width=W]
     */
    kImage2DActivation,
    /*! \brief Two dimensional texture w/ height = axis[0]
     *         e.g. image2d[height=O, width=IHW]
     */
    kImage2DWeight,
    /*! \brief Two dimensional texture w/ height = axis[1]
     *         e.g. image2d[height=NH, width=WC]
     */
    kImage2DNHWC,
  };
  BufferDescriptor() = default;
  explicit BufferDescriptor(Optional<String> scope) : layout(MemoryLayoutFromScope(scope)) {}
  static MemoryLayout MemoryLayoutFromScope(Optional<String> mem_scope);
  static String ScopeFromMemoryLayout(MemoryLayout mem_scope);

  cl_mem buffer{nullptr};
  cl_uchar* host_ptr{nullptr};
  MemoryLayout layout{MemoryLayout::kBuffer1D};
};
}  // namespace cl

// Module to support thread-safe multi-device execution.
// OpenCL runtime is a bit tricky because clSetKernelArg is not thread-safe
// To make the call thread-safe, we create a thread-local kernel table
// and lazily install new kernels into the kernel table when the kernel is called.
// The kernels are recycled when the module get destructed.
class OpenCLModuleNodeBase : public ModuleNode {
 public:
  // Kernel table reference entry.
  struct KTRefEntry {
    size_t kernel_id;
    size_t version;
  };
  explicit OpenCLModuleNodeBase(std::unordered_map<std::string, FunctionInfo> fmap) : fmap_(fmap) {}
  // destructor
  ~OpenCLModuleNodeBase();

  /*!
   * \brief Get the global workspace
   */
  virtual cl::OpenCLWorkspace* GetGlobalWorkspace();

  const char* type_key() const final { return workspace_->type_key.c_str(); }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final {
    return ModulePropertyMask::kBinarySerializable | ModulePropertyMask::kRunnable;
  }

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) override;

  // Initialize the programs
  virtual void Init() = 0;
  // install a new kernel to thread local entry
  virtual cl_kernel InstallKernel(cl::OpenCLWorkspace* w, cl::OpenCLThreadEntry* t,
                                  const std::string& func_name, const KTRefEntry& e) = 0;

 protected:
  // The workspace, need to keep reference to use it in destructor.
  // In case of static destruction order problem.
  cl::OpenCLWorkspace* workspace_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // Module local mutex
  std::mutex build_lock_;
  // Mapping from primitive name to cl program for each device.
  std::unordered_map<std::string, std::vector<cl_program>> programs_;
  // kernel id cache
  std::unordered_map<std::string, KTRefEntry> kid_map_;
  // kernels built so far.
  std::vector<cl_kernel> kernels_;
};

class OpenCLModuleNode : public OpenCLModuleNodeBase {
 public:
  explicit OpenCLModuleNode(std::string data, std::string fmt,
                            std::unordered_map<std::string, FunctionInfo> fmap, std::string source)
      : OpenCLModuleNodeBase(fmap), data_(data), fmt_(fmt), source_(source) {}

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final;
  // Return true if OpenCL program for the requested function and device was created
  bool IsProgramCreated(const std::string& func_name, int device_id);
  void SaveToFile(const String& file_name, const String& format) final;
  void SaveToBinary(dmlc::Stream* stream) final;
  void SetPreCompiledPrograms(const std::string& bytes);
  std::string GetPreCompiledPrograms();
  String GetSource(const String& format) final;

  // Initialize the programs
  void Init() override;
  // install a new kernel to thread local entry
  cl_kernel InstallKernel(cl::OpenCLWorkspace* w, cl::OpenCLThreadEntry* t,
                          const std::string& func_name, const KTRefEntry& e) override;

 private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // The OpenCL source.
  std::string source_;
  // parsed kernel data
  std::unordered_map<std::string, std::string> parsed_kernels_;
};

/*! \brief OpenCL timer node */
class OpenCLTimerNode : public TimerNode {
 public:
  // Timer start
  virtual void Start() {
    this->duration = 0;
    if (count_timer_execs == 0) {
      cl::OpenCLWorkspace::Global()->GetEventQueue(dev_).clear();
      // Very first call of Start() leads to the recreation of
      // OpenCL command queue in profiling mode. This allows to run profile after inference.
      recreateCommandQueue();
    }
    ++count_timer_execs;
    // set new first idx in event queue
    if (event_start_idxs.size() < count_timer_execs) {
      event_start_idxs.push_back(0);
    }
  }
  // Timer stop
  virtual void Stop() {
    std::vector<cl_event> evt_queue = cl::OpenCLWorkspace::Global()->GetEventQueue(dev_);
    cl_ulong start, end;
    size_t start_idx = event_start_idxs[count_timer_execs - 1];

    if (cl::OpenCLWorkspace::Global()->GetEventQueue(dev_).size() > 0) {
      OPENCL_CALL(clWaitForEvents(1, &(cl::OpenCLWorkspace::Global()->GetEventQueue(dev_).back())));
      for (size_t i = start_idx; i < evt_queue.size(); ++i) {
        auto& kevt = evt_queue[i];
        OPENCL_CALL(clGetEventProfilingInfo(kevt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
                                            &start, nullptr));
        OPENCL_CALL(clGetEventProfilingInfo(kevt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end,
                                            nullptr));
        this->duration += (end - start);
      }
    }
    // update event index for current call nesting
    event_start_idxs[count_timer_execs - 1] = evt_queue.size();
    --count_timer_execs;
  }
  virtual int64_t SyncAndGetElapsedNanos() { return this->duration; }
  // destructor
  virtual ~OpenCLTimerNode() {
    // Profiling session ends, recreate clCommandQueue in non-profiling mode
    // This will disable collection of cl_events in case of executing inference after profile
    if (count_timer_execs == 0) {
      recreateCommandQueue();
      event_start_idxs.clear();
    }
  }
  // constructor
  OpenCLTimerNode() {}
  explicit OpenCLTimerNode(Device dev) : dev_(dev) {}

  static constexpr const char* _type_key = "OpenCLTimerNode";
  static size_t count_timer_execs;
  static std::vector<size_t> event_start_idxs;
  TVM_DECLARE_FINAL_OBJECT_INFO(OpenCLTimerNode, TimerNode);

 private:
  int64_t duration;
  Device dev_;

  void recreateCommandQueue() {
    cl::OpenCLWorkspace::Global()->EnableQueueProfiling(
        dev_, !cl::OpenCLWorkspace::Global()->IsProfiling(dev_));
  }
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_OPENCL_OPENCL_COMMON_H_
