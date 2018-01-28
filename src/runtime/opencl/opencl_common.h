/*!
 *  Copyright (c) 2017 by Contributors
 * \file opencl_common.h
 * \brief OpenCL common header
 */
#ifndef TVM_RUNTIME_OPENCL_OPENCL_COMMON_H_
#define TVM_RUNTIME_OPENCL_OPENCL_COMMON_H_

#include <tvm/runtime/config.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/device_api.h>
#include <dmlc/logging.h>

#if TVM_OPENCL_RUNTIME
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <mutex>
#include <string>
#include <vector>
#include "../workspace_pool.h"

namespace tvm {
namespace runtime {
namespace cl {

static_assert(sizeof(cl_mem) ==sizeof(void*),
              "Required to store cl_mem inside void*");

inline const char* CLGetErrorString(cl_int error) {
  switch (error) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
    default: return "Unknown OpenCL error code";
  }
}

/*!
 * \brief Protected OpenCL call
 * \param func Expression to call.
 */
#define OPENCL_CHECK_ERROR(e)                                           \
  {                                                                     \
    CHECK(e == CL_SUCCESS)                                              \
        << "OpenCL Error, code=" << e << ": " << cl::CLGetErrorString(e); \
  }

#define OPENCL_CALL(func)                                             \
  {                                                                   \
    cl_int e = (func);                                                \
    OPENCL_CHECK_ERROR(e);                                            \
  }

/*!
 * \brief Process global OpenCL workspace.
 */
class OpenCLWorkspace final : public DeviceAPI {
 public:
  // global platform id
  cl_platform_id platform_id;
  // global context of this process
  cl_context context{nullptr};
  // whether the workspace it initialized.
  bool initialized_{false};
  // the devices
  std::vector<cl_device_id> devices;
  // the queues
  std::vector<cl_command_queue> queues;
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
    if (context != nullptr) {
      OPENCL_CALL(clReleaseContext(context));
    }
  }
  // Initialzie the device.
  void Init();
  // get the queue of the context
  cl_command_queue GetQueue(TVMContext ctx) {
    CHECK_EQ(ctx.device_type, kDLOpenCL);
    this->Init();
    CHECK(ctx.device_id >= 0  && static_cast<size_t>(ctx.device_id) < queues.size())
        << "Invalid OpenCL device_id=" << ctx.device_id;
    return queues[ctx.device_id];
  }
  // override device API
  void SetDevice(TVMContext ctx) final;
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final;
  void* AllocDataSpace(TVMContext ctx,
                       size_t size,
                       size_t alignment,
                       TVMType type_hint) final;
  void FreeDataSpace(TVMContext ctx, void* ptr) final;
  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMStreamHandle stream) final;
  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final;
  void* AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint) final;
  void FreeWorkspace(TVMContext ctx, void* data) final;
  // get the global workspace
  static const std::shared_ptr<OpenCLWorkspace>& Global();
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
  /*! \brief The current context */
  TVMContext context;
  /*! \brief The thread-local kernel table */
  std::vector<KTEntry> kernel_table;
  /*! \brief workspace pool */
  WorkspacePool pool;
  // constructor
  OpenCLThreadEntry()
      : pool(kDLOpenCL, OpenCLWorkspace::Global()) {
    context.device_id = 0;
    context.device_type = kDLOpenCL;
  }
  // get the global workspace
  static OpenCLThreadEntry* ThreadLocal();
};
}  // namespace cl
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_OPENCL_RUNTIME
#endif  // TVM_RUNTIME_OPENCL_OPENCL_COMMON_H_
