/*!
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
 * This Contribution is being provided by Qualcomm Technologies, Inc.,
 * a Delaware corporation, or its subsidiary Qualcomm Innovation Center, Inc.,
 * a California corporation, under certain additional terms and conditions
 * pursuant to Section 5 of the Apache 2.0 license.  In this regard, with
 * respect to this Contribution, the term "Work" in Section 1 of the
 * Apache 2.0 license means only the specific subdirectory within the TVM repo
 * (currently at https://github.com/dmlc/tvm) to which this Contribution is
 * made.
 * In any case, this submission is "Not a Contribution" with respect to its
 * permitted use with any of the "vta" and "verilog" subdirectories in the TVM
 * repo.
 * Qualcomm Technologies, Inc. and Qualcomm Innovation Center, Inc. retain
 * copyright of their respective Contributions.
 */
#include "hexagon_module.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif
#include <dmlc/logging.h>
#include <tvm/runtime/registry.h>

#include <memory>
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
                    std::unordered_map<std::string, FunctionInfo> fmap)
      : data_(data), fmt_(fmt), fmap_(fmap) {
    dl_handle_ = hexagon::Device::Global()->Load(data, fmt);
  }
  ~HexagonModuleNode() {
    if (dl_handle_) {
      hexagon::Device::Global()->Unload(dl_handle_);
    }
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final;
  const char* type_key() const final;
  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string meta_file = GetMetaFilePath(file_name);
    SaveMetaDataToFile(meta_file, fmap_);
    std::string c = "cp " + data_ + " " + file_name;
    CHECK(std::system(c.c_str()) == 0) << "Cannot create " + file_name;
  }

 private:
  hexagon::ArgLayout BuildArgLayout(const TVMArgs& Aa) const;

  void* dl_handle_ = nullptr;
  std::string data_;
  std::string fmt_;
  std::unordered_map<std::string, FunctionInfo> fmap_;
};

const char* HexagonModuleNode::type_key() const { return "hexagon"; }

PackedFunc HexagonModuleNode::GetFunction(
    const std::string& name, const std::shared_ptr<ModuleNode>& sptr_to_self) {
  auto f = fmap_.find(name);
  if (f == fmap_.end()) return PackedFunc(nullptr);

  // Get function pointer from device.
  void* pf = hexagon::Device::Global()->Resolve(name);

  auto func = [pf, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
    auto m = std::static_pointer_cast<const HexagonModuleNode>(sptr_to_self);
    hexagon::ArgLayout As = m->BuildArgLayout(args);
    hexagon::Device* HD = hexagon::Device::Global();
    HD->Call(pf, As.Scalar.data(), As.Scalar.size(), As.Stack.data(),
             As.Stack.size());
  };
  return PackedFunc(func);
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

      case kHandle:
      case kNull:
      case kArrayHandle:
      case kNodeHandle:
      case kModuleHandle:
      case kFuncHandle:
        Args.Push(static_cast<void*>(A));
        break;

      default:
        LOG(FATAL) << "Unhandled type code" << TC;
        break;
    }
  }

  return Args;
}

Module HexagonModuleCreate(
    std::string data, std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap) {
  return Module(std::make_shared<HexagonModuleNode>(data, fmt, fmap));
}

// Load module from file.
Module HexagonModuleLoadFile(const std::string& file_name,
                             const std::string& format) {
  std::string data = file_name;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadMetaDataFromFile(meta_file, &fmap);
  return HexagonModuleCreate(data, fmt, fmap);
}

namespace hexagon {

Device* Device::Global() {
  // Declare device constructors.
#ifdef __ANDROID__
  std::unique_ptr<Device> CreateHexagonTarget(void);
#else
  std::unique_ptr<Device> CreateHexagonSimulator(void);
#endif

  static std::unique_ptr<Device> dev(
#ifdef __ANDROID__
      CreateHexagonTarget()
#else
      CreateHexagonSimulator()
#endif
  );  // NOLINT

  return dev.get();
}

}  // namespace hexagon

TVM_REGISTER_GLOBAL("module.loadfile_hexagon")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = HexagonModuleLoadFile(args[0], args[1]);
    });

}  // namespace runtime
}  // namespace tvm
