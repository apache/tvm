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
 * \file src/runtime/relax_vm/builtin.cc
 */
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/memory.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/builtin.h>
#include <tvm/runtime/relax_vm/bytecode.h>
#include <tvm/runtime/relax_vm/vm.h>

#include "../runtime_base.h"

namespace tvm {
namespace runtime {
namespace relax_vm {

using tvm::runtime::NDArray;

//-------------------------------------------------
//  Shape/StructInfo handling.
//-------------------------------------------------
/*!
 * \brief Builtin function to allocate shape heap.
 * \param ctx_ptr The context module pointer.
 * \param size the size of the heap.
 * \return An allocate NDArray as shape heap.
 */
NDArray AllocShapeHeap(void* ctx_ptr, int64_t size) {
  VirtualMachine* vm = static_cast<VirtualMachine*>(ctx_ptr);
  // use host allocator, which is always last element.
  size_t host_device_index = vm->devices.size() - 1;
  // specially handle hexagon on-device RT.
  // TODO(relax-team): visit and consider other possible choices.
  if (vm->devices[0].device_type == kDLHexagon) {
    host_device_index = 0;
  } else {
    ICHECK_EQ(vm->devices[host_device_index].device_type, kDLCPU);
  }
  auto* alloc = vm->allocators[host_device_index];
  return alloc->Empty({size}, DLDataType{kDLInt, 64, 1}, vm->devices[host_device_index]);
}

TVM_REGISTER_GLOBAL("vm.builtin.alloc_shape_heap").set_body_typed(AllocShapeHeap);

/*!
 * \brief Builtin match R.Prim function.
 *
 * \param input_value The runtime value provided by the user
 *
 * \param heap The VM storage for symbolic shapes
 *
 * \param code_value The op code, defined in MatchShapeCode,
 *     indicating how this value should be interpreted.
 *
 * \param reg The register, if using kStoreToHeap or
 *     kAssertEqualToLoad, or a literal value if using kAssertEqualToImm
 *
 * \param err_ctx An optional string used in error messages, providing
 *     additional context
 *
 * \sa MatchShape
 */
void MatchPrimValue(int64_t input_value, DLTensor* heap, int code_value, int64_t reg,
                    Optional<String> err_ctx) {
  int64_t* heap_data = heap == nullptr ? nullptr : static_cast<int64_t*>(heap->data);
  MatchShapeCode code = static_cast<MatchShapeCode>(code_value);

  if (code == MatchShapeCode::kAssertEqualToImm) {
    CHECK_EQ(input_value, reg) << "RuntimeError: " << err_ctx.value_or("") << " match_cast error, "
                               << " PrimValue mismatch to specified constant.";
  } else if (code == MatchShapeCode::kStoreToHeap) {
    heap_data[reg] = input_value;
  } else if (code == MatchShapeCode::kNoOp) {
  } else if (code == MatchShapeCode::kAssertEqualToLoad) {
    CHECK_EQ(input_value, heap_data[reg])
        << "RuntimeError: " << err_ctx.value_or("") << " match_cast error, "
        << " PrimValue mismatch to a previous populated value.";
  } else {
    LOG(FATAL) << "Unknown match shape code: " << static_cast<int>(code);
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.match_prim_value").set_body_typed(MatchPrimValue);

/*!
 * \brief Builtin match shape function.
 * \param args The packed function arguments.
 * \param rv The return value.
 *
 * \sa MatchShapeCode
 */
void MatchShape(ffi::PackedArgs args, Any* rv) {
  // input shape the first argument can take in tensor or shape.
  ffi::Shape input_shape;
  if (auto opt_nd = args[0].as<NDArray>()) {
    input_shape = opt_nd.value().Shape();
  } else {
    input_shape = args[0].cast<ffi::Shape>();
  }
  auto heap = args[1].as<DLTensor*>();
  int64_t* heap_data = heap.has_value() ? static_cast<int64_t*>((*heap)->data) : nullptr;
  int64_t size = args[2].cast<int64_t>();
  const int64_t kBeginCode = 3;
  ICHECK_LE(kBeginCode + size * 2, args.size());
  // a function that lazily get context for error reporting
  const int64_t kErrorContextOffset = kBeginCode + size * 2;
  Optional<String> err_ctx = args[kErrorContextOffset].cast<String>();

  CHECK_EQ(input_shape.size(), size)
      << "RuntimeError: " << err_ctx.value_or("") << " match_cast shape size mismatch.";

  for (int64_t i = 0; i < size; ++i) {
    MatchShapeCode code = static_cast<MatchShapeCode>(args[kBeginCode + i * 2].cast<int>());
    int64_t reg = args[kBeginCode + i * 2 + 1].cast<int64_t>();

    if (code == MatchShapeCode::kAssertEqualToImm) {
      CHECK_EQ(input_shape[i], reg)
          << "RuntimeError: " << err_ctx.value_or("") << " match_cast error, "
          << " shape[" << i << "]"
          << " mismatch to specified constant.";
    } else if (code == MatchShapeCode::kStoreToHeap) {
      heap_data[reg] = input_shape[i];
    } else if (code == MatchShapeCode::kNoOp) {
    } else {
      ICHECK(code == MatchShapeCode::kAssertEqualToLoad);
      CHECK_EQ(input_shape[i], heap_data[reg])
          << "RuntimeError: " << err_ctx.value_or("") << " match_cast error, "
          << " shape[" << i << "]"
          << " mismatch to a previous populated value.";
    }
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.match_shape").set_body_packed(MatchShape);

/*!
 * \brief Builtin make prim value function.
 * \param heap The shape heap to use
 * \param shape_code The shape code of the value
 * \param rv The return value.
 *
 * \sa MakeShape
 */
int64_t MakePrimValue(DLTensor* heap, int shape_code, int64_t reg) {
  // NOTE: heap can be nullptr
  int64_t* heap_data = heap == nullptr ? nullptr : static_cast<int64_t*>(heap->data);

  MakeShapeCode code = static_cast<MakeShapeCode>(shape_code);
  if (code == MakeShapeCode::kUseImm) {
    return reg;
  } else if (code == MakeShapeCode::kLoadShape) {
    return heap_data[reg];
  } else {
    LOG(FATAL) << "Invalid shape code: " << shape_code;
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.make_prim_value").set_body_typed(MakePrimValue);

/*!
 * \brief Builtin make shape function.
 * \param args The packed function arguments.
 * \param rv The return value.
 *
 * \sa MakeShapeCode
 */
void MakeShape(ffi::PackedArgs args, Any* rv) {
  // NOTE: heap can be nullptr
  auto heap = args[0].as<DLTensor*>();
  int64_t* heap_data = heap.has_value() ? static_cast<int64_t*>((*heap)->data) : nullptr;
  int64_t size = args[1].cast<int64_t>();
  const int64_t kBeginCode = 2;

  std::vector<int64_t> shape(size);

  for (int64_t i = 0; i < size; ++i) {
    MakeShapeCode code = static_cast<MakeShapeCode>(args[kBeginCode + i * 2].cast<int>());
    int64_t reg = args[kBeginCode + i * 2 + 1].cast<int64_t>();
    if (code == MakeShapeCode::kUseImm) {
      shape[i] = reg;
    } else {
      ICHECK(code == MakeShapeCode::kLoadShape);
      shape[i] = heap_data[reg];
    }
  }
  *rv = ffi::Shape(std::move(shape));
}

TVM_REGISTER_GLOBAL("vm.builtin.make_shape").set_body_packed(MakeShape);

/*!
 * \brief Builtin function to check if arg is Tensor(dtype, ndim)
 * \param arg The input argument.
 * \param ndim Expected ndim of the Tensor, can be -1 (indicate unknown).
 * \param dtype The expected content data type.
 * \param err_ctx Additional context if error occurs.
 */
void CheckTensorInfo(ffi::PackedArgs args, Any* rv) {
  AnyView arg = args[0];
  int ndim = args[1].cast<int>();
  DataType dtype;
  Optional<String> err_ctx;

  if (args.size() == 3) {
    dtype = DataType::Void();
    err_ctx = args[2].cast<Optional<String>>();
  } else {
    dtype = args[2].cast<DataType>();
    err_ctx = args[3].cast<Optional<String>>();
  }

  auto opt_ptr = arg.as<DLTensor*>();
  CHECK(opt_ptr.has_value()) << "TypeError: " << err_ctx.value_or("") << " expect a Tensor but get "
                             << arg.GetTypeKey();

  DLTensor* ptr = opt_ptr.value();
  if (ndim != -1) {
    CHECK(ptr->ndim == ndim) << "ValueError: " << err_ctx.value_or("")
                             << " expect Tensor with ndim " << ndim << " but get " << ptr->ndim;
  }

  if (dtype != DataType::Void()) {
    CHECK(DataType(ptr->dtype) == dtype)
        << "ValueError: " << err_ctx.value_or("") << " expect Tensor with dtype " << dtype
        << " but get " << DataType(ptr->dtype);
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.check_tensor_info").set_body_packed(CheckTensorInfo);

/*!
 * \brief Builtin function to check if arg is Shape(ndim)
 * \param arg The input argument.
 * \param ndim Expected size of the shape, can be -1 (indicate unknown).
 * \param err_ctx Additional context if error occurs.
 */
void CheckShapeInfo(ObjectRef arg, int ndim, Optional<String> err_ctx) {
  // a function that lazily get context for error reporting
  auto* ptr = arg.as<ffi::Shape::ContainerType>();
  CHECK(ptr != nullptr) << "TypeError: " << err_ctx.value_or("") << " expect a Shape but get "
                        << arg->GetTypeKey();
  if (ndim != -1) {
    CHECK(ptr->size == static_cast<uint64_t>(ndim))
        << "ValueError: " << err_ctx.value_or("") << " expect Shape with ndim " << ndim
        << " but get " << ptr->size;
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.check_shape_info").set_body_typed(CheckShapeInfo);

/*!
 * \brief Builtin function to check if arg is PrimValue(dtype)
 * \param arg The input argument.
 * \param dtype Expected dtype of the PrimValue.  Can be DataType::Void() for unknown dtype.
 * \param err_ctx Additional context if error occurs.
 */
void CheckPrimValueInfo(AnyView arg, DataType dtype, Optional<String> err_ctx) {
  if (auto opt_obj = arg.as<ObjectRef>()) {
    LOG(FATAL) << "TypeError: " << err_ctx.value_or("") << ", expected dtype " << dtype
               << ", but received ObjectRef of type " << opt_obj.value()->GetTypeKey();
  } else if (dtype.is_bool()) {
    arg.cast<bool>();
  } else if (dtype.is_int()) {
    arg.cast<int64_t>();
  } else if (dtype.is_uint()) {
    arg.cast<uint64_t>();
  } else if (dtype.is_float()) {
    arg.cast<double>();
  } else if (dtype.is_handle()) {
    arg.cast<void*>();
  } else {
    LOG(FATAL) << "TypeError: " << err_ctx.value_or("") << ", unsupported dtype " << dtype;
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.check_prim_value_info").set_body_typed(CheckPrimValueInfo);

/*!
 * \brief Builtin function to check if arg is Tuple with size elements.
 * \param arg The input argument.
 * \param size The expected size of the tuple.
 * \param err_ctx Additional context if error occurs.
 */
void CheckTupleInfo(ObjectRef arg, int64_t size, Optional<String> err_ctx) {
  // a function that lazily get context for error reporting
  auto* ptr = arg.as<ffi::ArrayObj>();
  CHECK(ptr != nullptr) << "TypeError: " << err_ctx.value_or("") << " expect a Tuple but get "
                        << arg->GetTypeKey();
  CHECK(static_cast<int64_t>(ptr->size()) == size)
      << "ValueError: " << err_ctx.value_or("") << " expect a Tuple with " << size << " elements, "
      << " but get a Tuple with " << ptr->size() << " elements.";
}

TVM_REGISTER_GLOBAL("vm.builtin.check_tuple_info").set_body_typed(CheckTupleInfo);

/*!
 * \brief Builtin function to check if arg is a callable function.
 * \param arg The input argument.
 * \param err_ctx Additional context if error occurs.
 */
void CheckFuncInfo(ObjectRef arg, Optional<String> err_ctx) {
  // a function that lazily get context for error reporting
  bool is_func = arg.as<ffi::Function::ContainerType>() || arg.as<VMClosure::ContainerType>();
  CHECK(is_func) << "TypeError: " << err_ctx.value_or("") << " expect a Function but get "
                 << arg->GetTypeKey();
}

TVM_REGISTER_GLOBAL("vm.builtin.check_func_info").set_body_typed(CheckFuncInfo);

//-------------------------------------------------
//  Storage management.
//-------------------------------------------------
Storage VMAllocStorage(void* ctx_ptr, ffi::Shape buffer_shape, Index device_index,
                       DLDataType dtype_hint, String mem_scope) {
  VirtualMachine* vm = static_cast<VirtualMachine*>(ctx_ptr);

  ICHECK_LT(device_index, vm->devices.size())
      << "The device index is out of VM physical devices list";

  if (device_index == -1) {
    // Allocate on host. Host is always the last element of vm->devices.
    device_index = vm->devices.size() - 1;
  }

  auto* alloc = vm->allocators[device_index];
  ICHECK(alloc) << "Did you forget to init the VirtualMachine with devices?";

  auto buffer = alloc->Alloc(vm->devices[device_index], buffer_shape, dtype_hint, mem_scope);

  return Storage(buffer, alloc);
}

TVM_REGISTER_GLOBAL("vm.builtin.alloc_storage").set_body_typed(VMAllocStorage);

TVM_REGISTER_GLOBAL("vm.builtin.alloc_tensor").set_body_method(&StorageObj::AllocNDArray);

//-------------------------------------------------
//  Closure function handling, calling convention
//-------------------------------------------------
TVM_REGISTER_GLOBAL("vm.builtin.make_closure").set_body_packed([](ffi::PackedArgs args, Any* rv) {
  VMClosure clo = args[0].cast<VMClosure>();
  std::vector<Any> saved_args;
  saved_args.resize(args.size() - 1);
  for (size_t i = 0; i < saved_args.size(); ++i) {
    saved_args[i] = args[i + 1];
  }
  auto impl = VMClosure::BindLastArgs(clo->impl, saved_args);
  *rv = VMClosure(clo->func_name, impl);
});

TVM_REGISTER_GLOBAL("vm.builtin.invoke_closure").set_body_packed([](ffi::PackedArgs args, Any* rv) {
  // args[0]: vm; args[1]: closure; args[2, 3, ...]: function arguments
  VirtualMachine* vm = VirtualMachine::GetContextPtr(args[0]);
  ObjectRef vm_closure = args[1].cast<ObjectRef>();
  vm->InvokeClosurePacked(vm_closure, args.Slice(2), rv);
});

TVM_REGISTER_GLOBAL("vm.builtin.call_tir_dyn").set_body_packed([](ffi::PackedArgs args, Any* rv) {
  ffi::Function func = args[0].cast<ffi::Function>();
  ffi::Shape to_unpack = args[args.size() - 1].cast<ffi::Shape>();
  size_t num_tensor_args = args.size() - 2;

  std::vector<AnyView> packed_args(num_tensor_args + to_unpack.size());
  std::copy(args.data() + 1, args.data() + args.size() - 1, packed_args.data());

  for (size_t i = 0; i < to_unpack.size(); ++i) {
    packed_args[i + num_tensor_args] = to_unpack[i];
  }
  func.CallPacked(ffi::PackedArgs(packed_args.data(), packed_args.size()), rv);
});

//-------------------------------------
//  Builtin runtime operators.
//-------------------------------------
TVM_REGISTER_GLOBAL("vm.builtin.shape_of").set_body_method(&NDArray::Shape);

TVM_REGISTER_GLOBAL("vm.builtin.copy").set_body_typed([](Any a) -> Any { return a; });

TVM_REGISTER_GLOBAL("vm.builtin.reshape").set_body_typed([](NDArray data, ffi::Shape new_shape) {
  return data.CreateView(new_shape, data->dtype);
});

TVM_REGISTER_GLOBAL("vm.builtin.null_value").set_body_typed([]() -> std::nullptr_t {
  return nullptr;
});

TVM_REGISTER_GLOBAL("vm.builtin.to_device")
    .set_body_typed([](NDArray data, int dev_type, int dev_id) {
      Device dst_device = {(DLDeviceType)dev_type, dev_id};
      return data.CopyTo(dst_device);
    });

/*!
 * \brief Load the scalar value in cond and return the result value.
 * \param cond The condition
 * \return Bool
 */
bool ReadIfCond(AnyView cond) {
  if (auto opt_int = cond.as<bool>()) {
    return opt_int.value();
  }
  NDArray arr = cond.cast<tvm::runtime::NDArray>();
  if (arr->device.device_type != kDLCPU) {
    arr = arr.CopyTo(DLDevice{kDLCPU, 0});
  }
  ICHECK(arr->dtype.code == kDLInt || arr->dtype.code == kDLUInt);
  int64_t result;
  switch (arr->dtype.bits) {
    case 1: {
      result = reinterpret_cast<int8_t*>(arr->data)[0];
      break;
    }
    case 8: {
      result = reinterpret_cast<int8_t*>(arr->data)[0];
      break;
    }
    case 16: {
      result = reinterpret_cast<int16_t*>(arr->data)[0];
      break;
    }
    case 32: {
      result = reinterpret_cast<int32_t*>(arr->data)[0];
      break;
    }
    case 64: {
      result = reinterpret_cast<int64_t*>(arr->data)[0];
      break;
    }
    default:
      LOG(FATAL) << "Unknown scalar int type: " << DLDataTypeToString(arr->dtype);
      throw;
  }
  return result != 0;
}

TVM_REGISTER_GLOBAL("vm.builtin.read_if_cond").set_body_typed(ReadIfCond);

//-------------------------------------
//  Debugging API
//-------------------------------------

TVM_REGISTER_GLOBAL("vm.builtin.invoke_debug_func")
    .set_body_packed([](ffi::PackedArgs args, Any* rv) -> void {
      ICHECK_GE(args.size(), 3);
      int num_args = args.size() - 3;
      ObjectRef io_effect = args[0].cast<ObjectRef>();
      ICHECK(!io_effect.defined()) << "ValueError: IOEffect is expected to be lowered to None.";
      String debug_func_name = args[1].cast<String>();
      const auto debug_func = tvm::ffi::Function::GetGlobal(debug_func_name);
      CHECK(debug_func.has_value()) << "ValueError: " << debug_func_name << " is not found. "
                                    << "Use the decorator `@tvm.register_func(\"" << debug_func_name
                                    << "\")` to register it.";
      String line_info = args[2].cast<String>();
      std::vector<AnyView> call_args(num_args + 1);
      {
        call_args[0] = line_info;
        for (int i = 0; i < num_args; ++i) {
          call_args[i + 1] = args[i + 3];
        }
      }
      debug_func->CallPacked(ffi::PackedArgs(call_args.data(), call_args.size()), rv);
      *rv = io_effect;
    });

//-------------------------------------
//  Data structure API
//-------------------------------------
TVM_REGISTER_GLOBAL("vm.builtin.tuple_getitem").set_body_typed([](Array<Any> arr, int64_t index) {
  return arr[index];
});

TVM_REGISTER_GLOBAL("vm.builtin.tuple_reset_item")
    .set_body_typed([](const ffi::ArrayObj* arr, int64_t index) {
      const_cast<ffi::ArrayObj*>(arr)->SetItem(index, nullptr);
    });

TVM_REGISTER_GLOBAL("vm.builtin.make_tuple").set_body_packed([](ffi::PackedArgs args, Any* rv) {
  Array<Any> arr;
  for (int i = 0; i < args.size(); ++i) {
    arr.push_back(args[i]);
  }
  *rv = arr;
});

TVM_REGISTER_GLOBAL("vm.builtin.tensor_to_shape").set_body_typed([](NDArray data) {
  NDArray arr = data;
  if (data->device.device_type != kDLCPU) {
    arr = data.CopyTo(DLDevice{kDLCPU, 0});
  }

  ICHECK_EQ(arr->ndim, 1);
  ICHECK_EQ(arr->dtype.code, kDLInt);

  std::vector<int64_t> out_shape;
  for (int i = 0; i < arr.Shape()[0]; ++i) {
    int64_t result;
    switch (arr->dtype.bits) {
      case 16: {
        result = reinterpret_cast<int16_t*>(arr->data)[i];
        break;
      }
      case 32: {
        result = reinterpret_cast<int32_t*>(arr->data)[i];
        break;
      }
      case 64: {
        result = reinterpret_cast<int64_t*>(arr->data)[i];
        break;
      }
      default:
        LOG(FATAL) << "Unknown scalar int type: " << DLDataTypeToString(arr->dtype);
        throw;
    }
    out_shape.push_back(result);
  }
  return ffi::Shape(out_shape);
});

TVM_REGISTER_GLOBAL("vm.builtin.ensure_zero_offset").set_body_typed([](NDArray data) {
  if (data->byte_offset == 0) {
    return data;
  }
  auto* device_api = DeviceAPI::Get(data->device);
  if (device_api->SupportsDevicePointerArithmeticsOnHost() &&
      data->byte_offset % tvm::runtime::kAllocAlignment == 0) {
    DLManagedTensor* dl_tensor = data.ToDLPack();
    dl_tensor->dl_tensor.data =
        reinterpret_cast<char*>(dl_tensor->dl_tensor.data) + dl_tensor->dl_tensor.byte_offset;
    dl_tensor->dl_tensor.byte_offset = 0;
    return NDArray::FromDLPack(dl_tensor);
  } else {
    auto new_array = NDArray::Empty(data.Shape(), data->dtype, data->device);
    new_array.CopyFrom(data);
    return new_array;
  }
});

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm

//-------------------------------------------------
// AnyList C runtime API: keep in relax for now.
//--------------------------------------------------
extern "C" {
/*!
 * \brief Backend function to get anylist item and set into Packed Func call arg stack.
 *
 * \param anylist The handle to the anylist, backed by ffi::Any*
 * \param int The index.
 * \param args The args stack.
 * \param arg_offset The offset of argument.
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_DLL int TVMBackendAnyListSetPackedArg(void* anylist, int index, TVMFFIAny* args,
                                          int arg_offset);
/*!
 * \brief Backend function to get anylist item and set into Packed Func call arg stack.
 *
 * \param anylist The handle to the anylist, backed by ffi::Any*
 * \param int The index.
 */
TVM_DLL int TVMBackendAnyListResetItem(void* anylist, int index);

/*!
 * \brief Backend function to set anylist item by moving from packed func return.
 *
 * \param anylist The handle to the anylist, backed by ffi::Any*
 * \param int The index.
 * \param args The args stack.
 * \param type_codes The type codes stack.
 * \param arg_offset The offset of argument.
 * \return 0 when no error is thrown, -1 when failure happens.
 */
TVM_DLL int TVMBackendAnyListMoveFromPackedReturn(void* anylist, int index, TVMFFIAny* args,
                                                  int ret_offset);

int TVMBackendAnyListSetPackedArg(void* anylist, int index, TVMFFIAny* args, int arg_offset) {
  using namespace tvm::runtime;
  API_BEGIN();
  auto* list = static_cast<TVMFFIAny*>(anylist);
  args[arg_offset] = list[index];
  API_END();
}

int TVMBackendAnyListResetItem(void* anylist, int index) {
  using namespace tvm::runtime;
  API_BEGIN();
  auto* list = static_cast<Any*>(anylist);
  list[index] = nullptr;
  API_END();
}

int TVMBackendAnyListMoveFromPackedReturn(void* anylist, int index, TVMFFIAny* args,
                                          int ret_offset) {
  using namespace tvm::runtime;
  API_BEGIN();
  auto* list = static_cast<Any*>(anylist);
  list[index] = tvm::ffi::details::AnyUnsafe::MoveTVMFFIAnyToAny(std::move(args[ret_offset]));
  API_END();
}
}  // extern "C"
