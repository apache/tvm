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
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
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
void MatchShape(TVMArgs args, TVMRetValue* rv) {
  // input shape the first argument can take in tensor or shape.
  ShapeTuple input_shape;
  if (args[0].IsObjectRef<NDArray>()) {
    input_shape = args[0].operator NDArray().Shape();
  } else {
    input_shape = args[0];
  }
  DLTensor* heap = args[1];
  int64_t* heap_data = heap == nullptr ? nullptr : static_cast<int64_t*>(heap->data);
  int64_t size = args[2];
  const int64_t kBeginCode = 3;
  ICHECK_LE(kBeginCode + size * 2, args.size());
  // a function that lazily get context for error reporting
  const int64_t kErrorContextOffset = kBeginCode + size * 2;
  Optional<String> err_ctx = args[kErrorContextOffset];

  CHECK_EQ(input_shape.size(), size)
      << "RuntimeError: " << err_ctx.value_or("") << " match_cast shape size mismatch.";

  for (int64_t i = 0; i < size; ++i) {
    MatchShapeCode code = static_cast<MatchShapeCode>(args[kBeginCode + i * 2].operator int());
    int64_t reg = args[kBeginCode + i * 2 + 1];

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

TVM_REGISTER_GLOBAL("vm.builtin.match_shape").set_body(MatchShape);

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
void MakeShape(TVMArgs args, TVMRetValue* rv) {
  // NOTE: heap can be nullptr
  DLTensor* heap = args[0];
  int64_t* heap_data = heap == nullptr ? nullptr : static_cast<int64_t*>(heap->data);
  int64_t size = args[1];
  const int64_t kBeginCode = 2;

  std::vector<int64_t> shape(size);

  for (int64_t i = 0; i < size; ++i) {
    MakeShapeCode code = static_cast<MakeShapeCode>(args[kBeginCode + i * 2].operator int());
    int64_t reg = args[kBeginCode + i * 2 + 1];
    if (code == MakeShapeCode::kUseImm) {
      shape[i] = reg;
    } else {
      ICHECK(code == MakeShapeCode::kLoadShape);
      shape[i] = heap_data[reg];
    }
  }
  *rv = ShapeTuple(std::move(shape));
}

TVM_REGISTER_GLOBAL("vm.builtin.make_shape").set_body(MakeShape);

/*!
 * \brief Builtin function to check if arg is Tensor(dtype, ndim)
 * \param arg The input argument.
 * \param ndim Expected ndim of the Tensor, can be -1 (indicate unknown).
 * \param dtype The expected content data type.
 * \param err_ctx Additional context if error occurs.
 */
void CheckTensorInfo(TVMArgs args, TVMRetValue* rv) {
  ObjectRef arg = args[0];
  int ndim = args[1];
  DataType dtype;
  Optional<String> err_ctx;

  if (args.size() == 3) {
    dtype = DataType::Void();
    err_ctx = args[2].operator Optional<String>();
  } else {
    dtype = args[2];
    err_ctx = args[3].operator Optional<String>();
  }

  auto* ptr = arg.as<NDArray::ContainerType>();
  CHECK(ptr != nullptr) << "TypeError: " << err_ctx.value_or("") << " expect a Tensor but get "
                        << arg->GetTypeKey();

  if (ndim != -1) {
    CHECK(ptr->dl_tensor.ndim == ndim)
        << "ValueError: " << err_ctx.value_or("") << " expect Tensor with ndim " << ndim
        << " but get " << ptr->dl_tensor.ndim;
  }

  if (dtype != DataType::Void()) {
    CHECK(DataType(ptr->dl_tensor.dtype) == dtype)
        << "ValueError: " << err_ctx.value_or("") << " expect Tensor with dtype " << dtype
        << " but get " << DataType(ptr->dl_tensor.dtype);
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.check_tensor_info").set_body(CheckTensorInfo);

/*!
 * \brief Builtin function to check if arg is Shape(ndim)
 * \param arg The input argument.
 * \param ndim Expected size of the shape, can be -1 (indicate unknown).
 * \param err_ctx Additional context if error occurs.
 */
void CheckShapeInfo(ObjectRef arg, int ndim, Optional<String> err_ctx) {
  // a function that lazily get context for error reporting
  auto* ptr = arg.as<ShapeTuple::ContainerType>();
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
void CheckPrimValueInfo(TVMArgValue arg, DataType dtype, Optional<String> err_ctx) {
  if (dtype.is_bool()) {
    arg.operator bool();
  } else if (dtype.is_int()) {
    arg.operator int64_t();
  } else if (dtype.is_uint()) {
    arg.operator uint64_t();
  } else if (dtype.is_float()) {
    arg.operator double();
  } else if (dtype.is_handle()) {
    arg.operator void*();
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
  auto* ptr = arg.as<runtime::ArrayNode>();
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
  bool is_func = arg.as<PackedFunc::ContainerType>() || arg.as<VMClosure::ContainerType>();
  CHECK(is_func) << "TypeError: " << err_ctx.value_or("") << " expect a Function but get "
                 << arg->GetTypeKey();
}

TVM_REGISTER_GLOBAL("vm.builtin.check_func_info").set_body_typed(CheckFuncInfo);

//-------------------------------------------------
//  Storage management.
//-------------------------------------------------
Storage VMAllocStorage(void* ctx_ptr, ShapeTuple buffer_shape, Index device_index,
                       DLDataType dtype_hint, String mem_scope) {
  VirtualMachine* vm = static_cast<VirtualMachine*>(ctx_ptr);

  ICHECK_LT(device_index, vm->devices.size())
      << "The device index is out of VM physical devices list";

  if (device_index == -1) {
    // Allocate on host. Host is always the last element of vm->devices.
    device_index = vm->devices.size() - 1;
  }

  auto storage_obj = runtime::SimpleObjAllocator().make_object<StorageObj>();
  auto* alloc = vm->allocators[device_index];
  ICHECK(alloc) << "Did you forget to init the VirtualMachine with devices?";

  storage_obj->buffer = alloc->Alloc(buffer_shape, dtype_hint, mem_scope);
  Storage storage(storage_obj);
  return storage;
}

TVM_REGISTER_GLOBAL("vm.builtin.alloc_storage").set_body_typed(VMAllocStorage);

TVM_REGISTER_GLOBAL("vm.builtin.alloc_tensor").set_body_method<Storage>(&StorageObj::AllocNDArray);

//-------------------------------------------------
//  Closure function handling, calling convention
//-------------------------------------------------
TVM_REGISTER_GLOBAL("vm.builtin.make_closure").set_body([](TVMArgs args, TVMRetValue* rv) {
  VMClosure clo = args[0];
  std::vector<TVMRetValue> saved_args;
  saved_args.resize(args.size() - 1);
  for (size_t i = 0; i < saved_args.size(); ++i) {
    saved_args[i] = args[i + 1];
  }
  auto impl = VMClosure::BindLastArgs(clo->impl, saved_args);
  *rv = VMClosure(clo->func_name, impl);
});

TVM_REGISTER_GLOBAL("vm.builtin.invoke_closure").set_body([](TVMArgs args, TVMRetValue* rv) {
  // args[0]: vm; args[1]: closure; args[2, 3, ...]: function arguments
  VirtualMachine* vm = VirtualMachine::GetContextPtr(args[0]);
  ObjectRef vm_closure = args[1];
  vm->InvokeClosurePacked(vm_closure,
                          TVMArgs(args.values + 2, args.type_codes + 2, args.size() - 2), rv);
});

TVM_REGISTER_GLOBAL("vm.builtin.call_tir_dyn").set_body([](TVMArgs args, TVMRetValue* rv) {
  PackedFunc func = args[0];
  ShapeTuple to_unpack = args[args.size() - 1];
  size_t num_tensor_args = args.size() - 2;

  std::vector<TVMValue> values(num_tensor_args + to_unpack.size());
  std::vector<int> tcodes(num_tensor_args + to_unpack.size());
  runtime::TVMArgsSetter setter(values.data(), tcodes.data());

  std::copy(args.values + 1, args.values + args.size() - 1, values.data());
  std::copy(args.type_codes + 1, args.type_codes + args.size() - 1, tcodes.data());

  for (size_t i = 0; i < to_unpack.size(); ++i) {
    setter(i + num_tensor_args, to_unpack[i]);
  }
  TVMArgs func_args(values.data(), tcodes.data(), values.size());
  func.CallPacked(func_args, rv);
});

//-------------------------------------
//  Builtin runtime operators.
//-------------------------------------
TVM_REGISTER_GLOBAL("vm.builtin.shape_of").set_body_method(&NDArray::Shape);

TVM_REGISTER_GLOBAL("vm.builtin.copy").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = args[0];
});

TVM_REGISTER_GLOBAL("vm.builtin.reshape").set_body_typed([](NDArray data, ShapeTuple new_shape) {
  return data.CreateView(new_shape, data->dtype);
});

TVM_REGISTER_GLOBAL("vm.builtin.null_value").set_body([](TVMArgs args, TVMRetValue* rv) {
  CHECK_EQ(args.size(), 0);
  *rv = nullptr;
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
bool ReadIfCond(TVMArgValue cond) {
  if (cond.type_code() == kDLInt) return cond.operator bool();
  NDArray arr = cond.operator tvm::runtime::NDArray();
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
      LOG(FATAL) << "Unknown scalar int type: " << DLDataType2String(arr->dtype);
      throw;
  }
  return result != 0;
}

TVM_REGISTER_GLOBAL("vm.builtin.read_if_cond").set_body_typed(ReadIfCond);

//-------------------------------------
//  Debugging API
//-------------------------------------

TVM_REGISTER_GLOBAL("vm.builtin.invoke_debug_func")
    .set_body([](TVMArgs args, TVMRetValue* rv) -> void {
      ICHECK_GE(args.size(), 3);
      int num_args = args.size() - 3;
      ObjectRef io_effect = args[0];
      ICHECK(!io_effect.defined()) << "ValueError: IOEffect is expected to be lowered to None.";
      String debug_func_name = args[1];
      const PackedFunc* debug_func = runtime::Registry::Get(debug_func_name);
      CHECK(debug_func) << "ValueError: " << debug_func_name << " is not found. "
                        << "Use the decorator `@tvm.register_func(\"" << debug_func_name
                        << "\")` to register it.";
      String line_info = args[2];
      std::vector<TVMValue> call_args(num_args + 1);
      std::vector<int> call_type_codes(num_args + 1);
      {
        TVMArgsSetter setter(call_args.data(), call_type_codes.data());
        setter(0, line_info);
        for (int i = 0; i < num_args; ++i) {
          setter(i + 1, args[i + 3]);
        }
      }
      debug_func->CallPacked(TVMArgs(call_args.data(), call_type_codes.data(), num_args + 1), rv);
      *rv = io_effect;
    });

//-------------------------------------
//  Data structure API
//-------------------------------------
TVM_REGISTER_GLOBAL("vm.builtin.tuple_getitem")
    .set_body_typed([](runtime::Array<ObjectRef> arr, int64_t index) { return arr[index]; });

TVM_REGISTER_GLOBAL("vm.builtin.make_tuple").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Array<ObjectRef> arr;
  for (int i = 0; i < args.num_args; ++i) {
    arr.push_back(args[i].operator ObjectRef());
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
        LOG(FATAL) << "Unknown scalar int type: " << DLDataType2String(arr->dtype);
        throw;
    }
    out_shape.push_back(result);
  }
  return ShapeTuple(out_shape);
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
 * \param anylist The handle to the anylist, backed by TVMRetValue*
 * \param int The index.
 * \param args The args stack.
 * \param type_codes The type codes stack.
 * \param arg_offset The offset of argument.
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_DLL int TVMBackendAnyListSetPackedArg(void* anylist, int index, TVMValue* args, int* type_codes,
                                          int arg_offset);
/*!
 * \brief Backend function to get anylist item and set into Packed Func call arg stack.
 *
 * \param anylist The handle to the anylist, backed by TVMRetValue*
 * \param int The index.
 */
TVM_DLL int TVMBackendAnyListResetItem(void* anylist, int index);

/*!
 * \brief Backend function to set anylist item by moving from packed func return.
 *
 * \param anylist The handle to the anylist, backed by TVMRetValue*
 * \param int The index.
 * \param args The args stack.
 * \param type_codes The type codes stack.
 * \param arg_offset The offset of argument.
 * \return 0 when no error is thrown, -1 when failure happens.
 */
TVM_DLL int TVMBackendAnyListMoveFromPackedReturn(void* anylist, int index, TVMValue* args,
                                                  int* type_codes, int ret_offset);

int TVMBackendAnyListSetPackedArg(void* anylist, int index, TVMValue* args, int* type_codes,
                                  int arg_offset) {
  using namespace tvm::runtime;
  API_BEGIN();
  auto* list = static_cast<TVMRetValue*>(anylist);
  TVMArgsSetter setter(args, type_codes);
  setter(arg_offset, list[index]);
  API_END();
}

int TVMBackendAnyListResetItem(void* anylist, int index) {
  using namespace tvm::runtime;
  API_BEGIN();
  auto* list = static_cast<TVMRetValue*>(anylist);
  list[index] = nullptr;
  API_END();
}

int TVMBackendAnyListMoveFromPackedReturn(void* anylist, int index, TVMValue* args, int* type_codes,
                                          int ret_offset) {
  using namespace tvm::runtime;
  API_BEGIN();
  auto* list = static_cast<TVMRetValue*>(anylist);
  if (type_codes[ret_offset] == kTVMStr || type_codes[ret_offset] == kTVMBytes) {
    list[index] = TVMArgValue(args[ret_offset], type_codes[ret_offset]);
  } else {
    list[index] = TVMRetValue::MoveFromCHost(args[ret_offset], type_codes[ret_offset]);
  }
  API_END();
}
}  // extern "C"
