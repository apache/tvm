# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Optional module to support faster DLPack conversion.

This is an optional module to support faster DLPack conversion for torch.
Some of the changes are merged but not yet released, so it is used
as a stop gap to support faster DLPack conversion.

This file contains source code from PyTorch:
License: licenses/LICENSE.pytorch.txt

This module only serves as temp measure and will
likely be phased away and deleted after changes landed and released in pytorch.

This module will load slowly at first time due to JITing,
subsequent calls will be much faster.
"""
import warnings
from . import libinfo


def load_torch_c_dlpack_extension():
    """Load the torch c dlpack extension."""
    cpp_source = """
#include <dlpack/dlpack.h>
#include <ATen/DLConvertor.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>

using namespace std;
namespace at {
namespace {

DLDataType getDLDataTypeForDLPackv1(const Tensor& t) {
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = t.element_size() * 8;
  switch (t.scalar_type()) {
    case ScalarType::UInt1:
    case ScalarType::UInt2:
    case ScalarType::UInt3:
    case ScalarType::UInt4:
    case ScalarType::UInt5:
    case ScalarType::UInt6:
    case ScalarType::UInt7:
    case ScalarType::Byte:
    case ScalarType::UInt16:
    case ScalarType::UInt32:
    case ScalarType::UInt64:
      dtype.code = DLDataTypeCode::kDLUInt;
      break;
    case ScalarType::Int1:
    case ScalarType::Int2:
    case ScalarType::Int3:
    case ScalarType::Int4:
    case ScalarType::Int5:
    case ScalarType::Int6:
    case ScalarType::Int7:
    case ScalarType::Char:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Double:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Float:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Int:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Long:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Short:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Half:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Bool:
      dtype.code = DLDataTypeCode::kDLBool;
      break;
    case ScalarType::ComplexHalf:
    case ScalarType::ComplexFloat:
    case ScalarType::ComplexDouble:
      dtype.code = DLDataTypeCode::kDLComplex;
      break;
    case ScalarType::BFloat16:
      dtype.code = DLDataTypeCode::kDLBfloat;
      break;
     case ScalarType::Float8_e5m2:
      dtype.code = DLDataTypeCode::kDLFloat8_e5m2;
      break;
    case ScalarType::Float8_e5m2fnuz:
      dtype.code = DLDataTypeCode::kDLFloat8_e5m2fnuz;
      break;
    case ScalarType::Float8_e4m3fn:
      dtype.code = DLDataTypeCode::kDLFloat8_e4m3fn;
      break;
    case ScalarType::Float8_e4m3fnuz:
      dtype.code = DLDataTypeCode::kDLFloat8_e4m3fnuz;
      break;
    case ScalarType::Float8_e8m0fnu:
      dtype.code = DLDataTypeCode::kDLFloat8_e8m0fnu;
      break;
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 8
    case ScalarType::Float4_e2m1fn_x2:
      dtype.code = DLDataTypeCode::kDLFloat4_e2m1fn;
      break;
#endif
   default:
      TORCH_CHECK(false, "Unsupported scalar type: ");
  }
  return dtype;
}

DLDevice torchDeviceToDLDeviceForDLPackv1(at::Device device) {
  DLDevice ctx;

  ctx.device_id = (device.is_cuda() || device.is_privateuseone())
      ? static_cast<int32_t>(static_cast<unsigned char>(device.index()))
      : 0;

  switch (device.type()) {
    case DeviceType::CPU:
      ctx.device_type = DLDeviceType::kDLCPU;
      break;
    case DeviceType::CUDA:
#ifdef USE_ROCM
      ctx.device_type = DLDeviceType::kDLROCM;
#else
      ctx.device_type = DLDeviceType::kDLCUDA;
#endif
      break;
    case DeviceType::OPENCL:
      ctx.device_type = DLDeviceType::kDLOpenCL;
      break;
    case DeviceType::HIP:
      ctx.device_type = DLDeviceType::kDLROCM;
      break;
    case DeviceType::XPU:
      ctx.device_type = DLDeviceType::kDLOneAPI;
      ctx.device_id = at::detail::getXPUHooks().getGlobalIdxFromDevice(device);
      break;
    case DeviceType::MAIA:
      ctx.device_type = DLDeviceType::kDLMAIA;
      break;
    case DeviceType::PrivateUse1:
      ctx.device_type = DLDeviceType::kDLExtDev;
      break;
    case DeviceType::MPS:
      ctx.device_type = DLDeviceType::kDLMetal;
      break;
    default:
      TORCH_CHECK(false, "Cannot pack tensors on " + device.str());
  }

  return ctx;
}

template <class T>
struct ATenDLMTensor {
  Tensor handle;
  T tensor{};
};

template <class T>
void deleter(T* arg) {
  delete static_cast<ATenDLMTensor<T>*>(arg->manager_ctx);
}

// Adds version information for DLManagedTensorVersioned.
// This is a no-op for the other types.
template <class T>
void fillVersion(T* tensor) {}

template <>
void fillVersion<DLManagedTensorVersioned>(
    DLManagedTensorVersioned* tensor) {
  tensor->flags = 0;
  tensor->version.major = DLPACK_MAJOR_VERSION;
  tensor->version.minor = DLPACK_MINOR_VERSION;
}

// This function returns a shared_ptr to memory managed DLpack tensor
// constructed out of ATen tensor
template <class T>
T* toDLPackImpl(const Tensor& src) {
  auto view = src;

  bool need_normalize_strides = false;
  int64_t expected_stride = 1;
  for (int i = src.dim() - 1; i >= 0; i--) {
    // detect if we do not meet continuous pattern
    // and the size is 1, so there is opportunity to normalize
    if (src.stride(i) != expected_stride && src.size(i) == 1) {
      need_normalize_strides = true;
      break;
    }
    expected_stride *= src.size(i);
  }

  // less common case, try normalizing the strides
  if (need_normalize_strides) {
    // create a new tensor with possibly normalized strides
    // gh-83069
    auto shape = src.sizes();
    auto strides = src.strides().vec();
    for (int i = 0; i < src.dim(); i++) {
      if (shape[i] < 2) {
        strides[i] = 1;
      }
    }
    view = src.as_strided(shape, strides, src.storage_offset());
  }

  ATenDLMTensor<T>* atDLMTensor(new ATenDLMTensor<T>);
  atDLMTensor->handle = view;
  atDLMTensor->tensor.manager_ctx = atDLMTensor;
  atDLMTensor->tensor.deleter = &deleter<T>;
  atDLMTensor->tensor.dl_tensor.data = view.data_ptr();
  atDLMTensor->tensor.dl_tensor.device = torchDeviceToDLDeviceForDLPackv1(src.device());
  atDLMTensor->tensor.dl_tensor.ndim = static_cast<int32_t>(src.dim());
  atDLMTensor->tensor.dl_tensor.dtype = getDLDataTypeForDLPackv1(src);
  atDLMTensor->tensor.dl_tensor.shape = const_cast<int64_t*>(view.sizes().data());
  atDLMTensor->tensor.dl_tensor.strides = const_cast<int64_t*>(view.strides().data());
  atDLMTensor->tensor.dl_tensor.byte_offset = 0;
  fillVersion(&atDLMTensor->tensor);
  return &(atDLMTensor->tensor);
}

static Device getATenDeviceForDLPackv1(DLDeviceType type, c10::DeviceIndex index, void* data = nullptr) {
  switch (type) {
    case DLDeviceType::kDLCPU:
      return at::Device(DeviceType::CPU);
#ifndef USE_ROCM
    // if we are compiled under HIP, we cannot do cuda
    case DLDeviceType::kDLCUDA:
      return at::Device(DeviceType::CUDA, index);
#endif
    case DLDeviceType::kDLOpenCL:
      return at::Device(DeviceType::OPENCL, index);
    case DLDeviceType::kDLROCM:
#ifdef USE_ROCM
      // this looks funny, we need to return CUDA here to masquerade
      return at::Device(DeviceType::CUDA, index);
#else
      return at::Device(DeviceType::HIP, index);
#endif
    case DLDeviceType::kDLOneAPI:
      TORCH_CHECK(data != nullptr, "Can't get ATen device for XPU without XPU data.");
      return at::detail::getXPUHooks().getDeviceFromPtr(data);
    case DLDeviceType::kDLMAIA:
      return at::Device(DeviceType::MAIA, index);
    case DLDeviceType::kDLExtDev:
      return at::Device(DeviceType::PrivateUse1, index);
    case DLDeviceType::kDLMetal:
      return at::Device(DeviceType::MPS, index);
    default:
      TORCH_CHECK(
          false, "Unsupported device_type: ", std::to_string(type));
  }
}


// This function constructs a Tensor from a memory managed DLPack which
// may be represented as either: DLManagedTensor and DLManagedTensorVersioned.
template <class T>
at::Tensor fromDLPackImpl(T* src, std::function<void(void*)> deleter) {
  if (!deleter) {
    deleter = [src](void* self [[maybe_unused]]) {
      if (src->deleter) {
        src->deleter(src);
      }
    };
  }

  DLTensor& dl_tensor = src->dl_tensor;
  Device device = getATenDeviceForDLPackv1(dl_tensor.device.device_type, dl_tensor.device.device_id, dl_tensor.data);
  ScalarType stype = toScalarType(dl_tensor.dtype);

  if (!dl_tensor.strides) {
    return at::from_blob(
        dl_tensor.data,
        IntArrayRef(dl_tensor.shape, dl_tensor.ndim),
        std::move(deleter),
        at::device(device).dtype(stype),
        {device});
  }
  return at::from_blob(
      dl_tensor.data,
      IntArrayRef(dl_tensor.shape, dl_tensor.ndim),
      IntArrayRef(dl_tensor.strides, dl_tensor.ndim),
      deleter,
      at::device(device).dtype(stype),
      {device});
}

} // namespace
} // namespace at

int TorchDLPackFromPyObject(void* py_obj, DLManagedTensorVersioned** out, void** env_stream) {
  try {
    py::handle handle(static_cast<PyObject*>(py_obj));
    at::Tensor tensor = handle.cast<at::Tensor>();
    if (env_stream != nullptr && tensor.is_cuda()) {
      *env_stream = at::cuda::getCurrentCUDAStream(tensor.device().index()).stream();
    }
    *out = at::toDLPackImpl<DLManagedTensorVersioned>(tensor);
    return 0;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  }
}

int TorchDLPackToPyObject(DLManagedTensorVersioned* src, void** py_obj_out) {
  try {
    at::Tensor tensor = at::fromDLPackImpl<DLManagedTensorVersioned>(src, nullptr);
    *py_obj_out = THPVariable_Wrap(tensor);
    return 0;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  }
}

int TorchDLPackTensorAllocator(
    DLTensor* prototype, DLManagedTensorVersioned** out, void* error_ctx,
    void (*SetError)(void* error_ctx, const char* kind, const char* message)
) {
  try {
    at::IntArrayRef shape(prototype->shape, prototype->shape + prototype->ndim);
    at::TensorOptions options = at::TensorOptions()
      .dtype(at::toScalarType(prototype->dtype))
      .device(at::getATenDeviceForDLPackv1(prototype->device.device_type, prototype->device.device_id));
    at::Tensor tensor = at::empty(shape, options);
    *out = at::toDLPackImpl<DLManagedTensorVersioned>(tensor);
    return 0;
  } catch (const std::exception& e) {
    SetError(error_ctx, "TorchDLPackTensorAllocator", e.what());
    return -1;
  }
}

int64_t TorchDLPackFromPyObjectPtr() {
  return reinterpret_cast<int64_t>(TorchDLPackFromPyObject);
}

int64_t TorchDLPackToPyObjectPtr() {
  return reinterpret_cast<int64_t>(TorchDLPackToPyObject);
}

int64_t TorchDLPackTensorAllocatorPtr() {
  return reinterpret_cast<int64_t>(TorchDLPackTensorAllocator);
}
    """
    try:
        # optionally import torch
        import torch
        from torch.utils import cpp_extension

        mod = cpp_extension.load_inline(
            name="to_dlpack",
            cpp_sources=cpp_source,
            functions=[
                "TorchDLPackFromPyObjectPtr",
                "TorchDLPackToPyObjectPtr",
                "TorchDLPackTensorAllocatorPtr",
            ],
            extra_cflags=["-O3"],
            extra_include_paths=libinfo.include_paths() + cpp_extension.include_paths("cuda"),
            verbose=True,
        )
        # set the dlpack related flags
        torch.Tensor.__c_dlpack_from_pyobject__ = mod.TorchDLPackFromPyObjectPtr()
        torch.Tensor.__c_dlpack_to_pyobject__ = mod.TorchDLPackToPyObjectPtr()
        torch.Tensor.__c_dlpack_tensor_allocator__ = mod.TorchDLPackTensorAllocatorPtr()
        return mod
    except ImportError:
        pass
    except Exception as e:
        warnings.warn(
            f"Failed to load torch c dlpack extension: {e},"
            "EnvTensorAllocator will not be enabled."
        )
        return None


# keep alive
_mod = load_torch_c_dlpack_extension()
