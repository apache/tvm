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

#include "hexagon_module.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif
#include <dmlc/logging.h>
#include <tvm/runtime/registry.h>

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "../file_util.h"
#include "../meta_data.h"

namespace tvm {
namespace runtime {

hexagon::Device::~Device() {}

namespace hexagon {

/*!
 * \brief Function argument locations according to the Hexagon ABI.
 *
 * In order to invoke a function whose arguments are in TVMArgs list, at
 * some point before branching to the function's address, these arguments
 * need to be loaded into locations (registers or stack) specified by the
 * corresponding ABI.
 * When a host wants to call a function on Hexagon, the host will identify
 * how each element of the TVMArgs list will be passed to the Hexagon
 * function. This class is a description of which values should go into
 * registers, and which values should be on stack. Right before the call
 * this class will be serialized and transfereed over to the Hexagon side.
 * The code running on Hexagon will then execute the argument placement
 * and invoke the function.
 */
struct ArgLayout {
  std::vector<uint32_t> Scalar; /*!< Values going into registers, maximum  */
                                /*!< 6, including dummy values for skipped */
                                /*!< registers.                            */
  std::vector<uint32_t> Stack;  /*!< Values going on stack, including      */
                                /*!< dummy values for padding.             */
  // There are no vector types at this time.

  /*!
   * \brief Alignment of type T on Hexagon.
   */
  template <typename T>
  static constexpr unsigned align_of();
  /*!
   * \brief Size of type T on Hexagon.
   */
  template <typename T>
  static constexpr unsigned size_of();

  /*!
   * \brief Add a value of type T to the layout.
   */
  template <typename T>
  void Push(const T& v);

 private:
  /*!
   * \brief Add raw data to the layout.
   * \param v         Pointer to the raw data as an array of 32-bit words.
   * \param t_size    Number of bytes to add.
   * \param t_align   Required alignment of the data on Hexagon.
   */
  void Push(uint32_t* v, unsigned t_size, unsigned t_align);
};

template <>
constexpr unsigned ArgLayout::align_of<int32_t>() {
  return 4;
}
template <>
constexpr unsigned ArgLayout::align_of<uint32_t>() {
  return 4;
}
template <>
constexpr unsigned ArgLayout::align_of<float>() {
  return 4;
}
template <>
constexpr unsigned ArgLayout::align_of<void*>() {
  return 4;
}
template <>
constexpr unsigned ArgLayout::align_of<int64_t>() {
  return 8;
}
template <>
constexpr unsigned ArgLayout::align_of<uint64_t>() {
  return 8;
}
template <>
constexpr unsigned ArgLayout::align_of<double>() {
  return 8;
}
template <>
constexpr unsigned ArgLayout::align_of<DLTensor*>() {
  return 4;
}

template <typename T>
constexpr unsigned ArgLayout::align_of() {
  // The static_assertion should depend on T so that it's only checked
  // after instantiation.
  static_assert((sizeof(T), false), "Implement align_of for this type");
  return 0;
}

template <typename T>
constexpr unsigned ArgLayout::size_of() {
  return ArgLayout::align_of<T>();
}

template <typename T>
void ArgLayout::Push(const T& v) {
  static_assert(std::is_scalar<T>::value, "T must be a scalar");
  constexpr unsigned T_size = size_of<T>();
  // The reason for this assertion is to avoid sign-extensions here:
  // an extra bit of information would be required to determine whether
  // a size- or a zero-extension is needed.
  static_assert(T_size >= 4, "Type should be of size that is at least 4");
  union {
    uint32_t v[(T_size + 3) / 4];
    T t;
  } u;

  u.t = v;
  Push(u.v, T_size, align_of<T>());
}

void ArgLayout::Push(uint32_t* v, unsigned t_size, unsigned t_align) {
  // t_size == 4 and t_size == 8 can be passed in scalar registers.
  bool InReg = false;
  if (t_size == 4) {
    if (Scalar.size() < 6) {
      Scalar.push_back(v[0]);
      InReg = true;
    }
  } else if (t_size == 8) {
    // Round the size up to the next
    unsigned cs = Scalar.size();
    if (cs <= 4) {
      // There is room in the scalar registers.
      if (cs & 1) Scalar.push_back(0u);
      Scalar.push_back(v[0]);
      Scalar.push_back(v[1]);
      InReg = true;
    }
  }

  if (!InReg) {
    // Allocate on stack.
    CHECK_EQ((t_align & (t_align - 1)), 0)
        << "Alignment should be a power of 2";
    CHECK_GE(t_align, 4) << "Alignment should be at least 4";
    // Round t_size up to a multiple of 4.
    unsigned s_size = Stack.size();
    unsigned s_align = t_align / 4;  // Alignment of T in words on the stack.
    unsigned pad = ((s_size + s_align - 1) / s_align) * s_align - s_size;
    Stack.insert(Stack.end(), pad / 4, 0u);
    Stack.insert(Stack.end(), v, v + t_size / 4);
  }
}

}  // namespace hexagon

class HexagonModuleNode final : public runtime::ModuleNode {
 public:
  HexagonModuleNode(std::string data, std::string fmt,
                    std::unordered_map<std::string, FunctionInfo> fmap,
                    std::string asm_str, std::string obj_str,
                    std::string ir_str, std::string bc_str,
                    const std::set<std::string>& packed_c_abi)
      : hexagon_device_(hexagon::Device::Global()),
        data_(data),
        fmt_(fmt),
        fmap_(fmap),
        asm_(asm_str),
        obj_(obj_str),
        ir_(ir_str),
        bc_(bc_str),
        packed_c_abi_funcs_(packed_c_abi) {
    dl_handle_ = hexagon_device_->Load(data, fmt);
  }
  ~HexagonModuleNode() {
    if (dl_handle_) {
      hexagon_device_->Unload(dl_handle_);
    }
  }

  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final;

  const char* type_key() const final { return "hexagon"; }

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = runtime::GetFileFormat(file_name, format);
    if (fmt == "so" || fmt == "dll" || fmt == "hexagon") {
      std::string meta_file = GetMetaFilePath(file_name);
      SaveMetaDataToFile(meta_file, fmap_);
      std::string c = "cp " + data_ + " " + file_name;
      CHECK(std::system(c.c_str()) == 0) << "Cannot create " + file_name;
    } else if (fmt == "s" || fmt == "asm") {
      CHECK(!asm_.empty()) << "Assembler source not available";
      SaveBinaryToFile(file_name, asm_);
    } else if (fmt == "o" || fmt == "obj") {
      CHECK(!obj_.empty()) << "Object data not available";
      SaveBinaryToFile(file_name, obj_);
    } else if (fmt == "ll") {
      CHECK(!ir_.empty()) << "LLVM IR source not available";
      SaveBinaryToFile(file_name, ir_);
    } else if (fmt == "bc") {
      CHECK(!bc_.empty()) << "LLVM IR bitcode not available";
      SaveBinaryToFile(file_name, bc_);
    } else {
      LOG(FATAL) << "HexagonModuleNode::SaveToFile: unhandled format `" << fmt
                 << "'";
    }
  }
  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
  }

 private:
  void CallRemotePackedCABI(void* func_ptr, const TVMArgs& args,
                            TVMRetValue* rv) const;
  void CallRemoteDirect(void* func_ptr, const TVMArgs& args,
                        TVMRetValue* rv) const;
  void RemapArgs(const TVMArgs& args,
                 std::vector<TVMValue>& values,              // NOLINT(*)
                 std::vector<int>& type_codes,               // NOLINT(*)
                 std::vector<void*>& remote_tensors) const;  // NOLINT(*)
  void* CreateRemoteTensor(const DLTensor* T) const;
  hexagon::ArgLayout BuildArgLayout(const TVMArgs& Aa) const;

  std::shared_ptr<hexagon::Device> hexagon_device_;
  void* dl_handle_ = nullptr;
  std::string data_;
  std::string fmt_;
  std::unordered_map<std::string, FunctionInfo> fmap_;
  std::string asm_;
  std::string obj_;
  std::string ir_;
  std::string bc_;
  std::set<std::string> packed_c_abi_funcs_;
};

void HexagonModuleNode::CallRemotePackedCABI(void* func_ptr,
                                             const TVMArgs& args,
                                             TVMRetValue* rv) const {
  // Remap all arguments, creating remote DLTensors.
  std::vector<TVMValue> values;
  std::vector<int> codes;
  std::vector<void*> remote_tensors;

  RemapArgs(args, values, codes, remote_tensors);
  // The prototype of packed C function is
  //   int (TVMValue* args, int* type_codes, int num_args,
  //        TVMValue* ret_value, int* ret_code)
  // The pointers must point to allocated space, the return information
  // will be filled in by the callee.
  // Allocate remote buffer to hold:
  // 1. argument TVMValues,
  // 2. return TVMValue,
  // 3. argument type codes,
  // 4. return type code.

  int num_args = args.size();
  int values_size = num_args * sizeof(TVMValue);
  int codes_size = num_args * sizeof(int);
  void* remote = hexagon_device_->Alloc(
      values_size + sizeof(TVMValue) + codes_size + sizeof(int), 8);

  // Copy all argument TVMValues to the remote space.
  void* remote_values = remote;
  void* remote_ret_value = static_cast<char*>(remote_values) + values_size;
  void* remote_codes = static_cast<char*>(remote_ret_value) + sizeof(TVMValue);
  void* remote_ret_code = static_cast<char*>(remote_codes) + codes_size;
  hexagon_device_->CopyHostToDevice(remote_values, values.data(), values_size);
  hexagon_device_->CopyHostToDevice(remote_codes, codes.data(), codes_size);

  // Call the function: construct temporary values/codes and pass them through
  // the arg layout building to preprare for the actual remote call.
  TVMValue temp_values[5];
  temp_values[0].v_handle = remote_values;
  temp_values[1].v_handle = remote_codes;
  temp_values[2].v_int64 = num_args;
  temp_values[3].v_handle = remote_ret_value;
  temp_values[4].v_handle = remote_ret_code;
  int temp_codes[5] = {kTVMOpaqueHandle, kTVMOpaqueHandle, kDLInt,
                       kTVMOpaqueHandle, kTVMOpaqueHandle};
  TVMArgs temp_args(temp_values, temp_codes, 5);
  hexagon::ArgLayout as = BuildArgLayout(temp_args);
  hexagon_device_->Call(func_ptr, as.Scalar.data(), as.Scalar.size(),
                        as.Stack.data(), as.Stack.size());

  // TODO(kparzysz-quic): copy return value back
  std::for_each(remote_tensors.begin(), remote_tensors.end(),
                [this](void* t) { hexagon_device_->Free(t); });
  hexagon_device_->Free(remote);
}

void HexagonModuleNode::CallRemoteDirect(void* func_ptr, const TVMArgs& args,
                                         TVMRetValue* rv) const {
  hexagon::ArgLayout as = BuildArgLayout(args);
  hexagon_device_->Call(func_ptr, as.Scalar.data(), as.Scalar.size(),
                        as.Stack.data(), as.Stack.size());
}

PackedFunc HexagonModuleNode::GetFunction(
    const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  auto f = fmap_.find(name);
  if (f == fmap_.end()) return PackedFunc(nullptr);

  // Get function pointer from device.
  void* pf = hexagon_device_->Resolve(name);
  // The cast result and the original share ownership. Do the cast here
  // so that sptr_to_self can be destroyed (i.e. "func" will only have
  // one shared pointer to HexagonModuleNode).
  auto sref = ObjectRef(sptr_to_self);

  if (packed_c_abi_funcs_.count(name)) {
    // Calling packed C func, follow the TVMBackendPackedCFunc prototype.
    return PackedFunc([pf, sref](TVMArgs args, TVMRetValue* rv) {
      const auto* hm = sref.as<HexagonModuleNode>();
      hm->CallRemotePackedCABI(pf, args, rv);
    });
  } else {
    // Direct call to a non-packed-C function.
    return PackedFunc([pf, sref](TVMArgs args, TVMRetValue* rv) {
      const auto* hm = sref.as<HexagonModuleNode>();
      hm->CallRemoteDirect(pf, args, rv);
    });
  }
}

void HexagonModuleNode::RemapArgs(const TVMArgs& args,
                                  std::vector<TVMValue>& values,
                                  std::vector<int>& type_codes,
                                  std::vector<void*>& remote_tensors) const {
  for (unsigned i = 0, e = args.size(); i != e; ++i) {
    const TVMArgValue& a = args[i];

    switch (unsigned tc = a.type_code()) {
      case kTVMNDArrayHandle:
      case kTVMDLTensorHandle: {
        DLTensor* t = static_cast<DLTensor*>(a);
        assert(TVMDeviceExtType(t->ctx.device_type) == kDLHexagon);
        TVMValue v;
        v.v_handle = CreateRemoteTensor(t);
        remote_tensors.push_back(v.v_handle);
        values.push_back(v);
        type_codes.push_back(tc);
        break;
      }

      default:
        values.push_back(a.value());
        type_codes.push_back(tc);
        break;
    }
  }
}

void* HexagonModuleNode::CreateRemoteTensor(const DLTensor* t) const {
  /*
    Layout of the DLTensor structure on Hexagon.

    DLTensor:                       Size  offset
      data              void*          4       0
      ctx.device_type   enum           1       4
      <pad>                            3       5
      ctx.device_id     int            4       8
      ndim              int            4      12
      dtype.code        uint8_t        1      16
      dtype.bits        uint8_t        1      17
      dtype.lanes       uint16_t       2      18
      shape             int64_t*       4      20
      strides           int64_t*       4      24
      <pad>                            4      28
      byte_offset       uint64_t       8      32
      .. end ................................ 40
  */
  struct __attribute__((packed)) HexagonDLTensor {
    uint32_t data;
    uint8_t ctx_device_type;
    uint8_t pad0[3];  // MUST BE ZERO!
    int32_t ctx_device_id;
    int32_t ndim;
    uint8_t dtype_code;
    uint8_t dtype_bits;
    uint16_t dtype_lanes;
    uint32_t shape;
    uint32_t strides;
    uint8_t pad1[4];
    uint64_t byte_offset;
  };

  constexpr uint32_t size_ht = sizeof(HexagonDLTensor);
  static_assert(size_ht == 40, "HexagonDLTensor should be 40 bytes");

  // Shape and strides will contain ndim elements of size sizeof(uint64_t)
  // each. Allocate them after the main structure.
  int ndim = t->ndim;
  uint32_t size_s = 8 * ndim;  // sizeof(uint64_t)*ndim
  uint32_t size_ss = t->strides ? 2 * size_s : size_s;
  void* remote = hexagon_device_->Alloc(size_ht + size_ss, 8);
  uint32_t remote_as_int = reinterpret_cast<uintptr_t>(remote);
  void* remote_ss = reinterpret_cast<void*>(remote_as_int + size_ht);

  HexagonDLTensor local = {
      .data = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(t->data)),
      .ctx_device_type = uint8_t(t->ctx.device_type),
      .pad0 = {0, 0, 0},
      .ctx_device_id = t->ctx.device_id,
      .ndim = t->ndim,
      .dtype_code = t->dtype.code,
      .dtype_bits = t->dtype.bits,
      .dtype_lanes = t->dtype.lanes,
      .shape = remote_as_int + size_ht,
      .strides = t->strides ? remote_as_int + size_ht + size_s : 0u,
      .byte_offset = t->byte_offset};

  std::vector<uint64_t> local_ss(size_ss / 8);
  for (int i = 0; i != ndim; ++i) local_ss[i] = t->shape[i];
  if (t->strides) {
    for (int i = 0; i != ndim; ++i) local_ss[ndim + i] = t->strides[i];
  }

  hexagon_device_->CopyHostToDevice(remote, &local, sizeof local);
  hexagon_device_->CopyHostToDevice(remote_ss, local_ss.data(), size_ss);
  return remote;
}

hexagon::ArgLayout HexagonModuleNode::BuildArgLayout(const TVMArgs& As) const {
  hexagon::ArgLayout Args;

  for (unsigned i = 0, e = As.size(); i != e; ++i) {
    const TVMArgValue& A = As[i];
    unsigned TC = A.type_code();
    switch (TC) {
      // Treat all integers as 32-bit values.
      case kDLInt:
      case kDLUInt:
        // KLUDGE: There is no distinction between 32- and 64-bit integer
        // types, so there is no way to tell if the value being passed needs
        // one or two registers. Assume that all integers are 32-bit, and
        // simply abort if the actual value does not fit.
        CHECK_EQ(static_cast<int64_t>(A), static_cast<int32_t>(A));
        Args.Push(static_cast<int>(A));
        break;
      // 64-bit values
      case kDLFloat:
        Args.Push(static_cast<double>(A));
        break;

      case kTVMOpaqueHandle:
      case kTVMNullptr:
      case kTVMObjectHandle:
      case kTVMModuleHandle:
      case kTVMPackedFuncHandle:
        Args.Push(static_cast<void*>(A));
        break;

      case kTVMNDArrayHandle:
      case kTVMDLTensorHandle:
        LOG(FATAL) << __func__ << ": cannot handle DLTensor*, code:" << TC;

      default:
        LOG(FATAL) << __func__ << ": unhandled type code" << TC;
        break;
    }
  }

  return Args;
}

Module HexagonModuleCreate(std::string data, std::string fmt,
                           std::unordered_map<std::string, FunctionInfo> fmap,
                           std::string asm_str, std::string obj_str,
                           std::string ir_str, std::string bc_str,
                           const std::set<std::string>& packed_c_abi) {
  auto n = make_object<HexagonModuleNode>(data, fmt, fmap, asm_str, obj_str,
                                          ir_str, bc_str, packed_c_abi);
  return Module(n);
}

// Load module from file.
Module HexagonModuleLoadFile(const std::string& file_name,
                             const std::string& format) {
  std::string data = file_name;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadMetaDataFromFile(meta_file, &fmap);

  std::string empty;
  // This passes {} as the set of packed C functions. Won't work for
  // standalone functions on target.
  return HexagonModuleCreate(data, fmt, fmap, empty, empty, empty, empty, {});
}

namespace hexagon {

std::shared_ptr<Device> Device::Global() {
  // Declare device constructors.
#ifdef __ANDROID__
  std::shared_ptr<Device> CreateHexagonTarget(void);
#else
  std::shared_ptr<Device> CreateHexagonSimulator(void);
#endif

  static std::shared_ptr<Device> dev(
#ifdef __ANDROID__
      CreateHexagonTarget()
#else
      CreateHexagonSimulator()
#endif
  );  // NOLINT

  return dev;
}

}  // namespace hexagon

TVM_REGISTER_GLOBAL("runtime.module.loadfile_hexagon")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = HexagonModuleLoadFile(args[0], args[1]);
    });

}  // namespace runtime
}  // namespace tvm
