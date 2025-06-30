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
 * \file pack_args.h
 * \brief Utility to pack ffi::PackedArgs to other type-erased fution calling convention.
 *
 *  Two type erased function signatures are supported.
 *   - cuda_style(void** args, int num_args);
 *      - Pack everything by address
 *   - metal_style(void** buffers, int num_buffers,
 *                 union_32bit args[N], int num_args);
 *      - Pack buffer by address, pack rest parameter into 32bit union buffer.
 */
#ifndef TVM_RUNTIME_PACK_ARGS_H_
#define TVM_RUNTIME_PACK_ARGS_H_

#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>

#include <cstring>
#include <vector>

namespace tvm {
namespace runtime {
/*!
 * \brief argument union type of 32bit.
 */
union ArgUnion32 {
  int32_t v_int32;
  uint32_t v_uint32;
  float v_float32;
};

/*!
 * \brief argument union type of 64 bit, for use by Vulkan and Metal runtime.
 */
union ArgUnion64 {
  int32_t v_int32[2];
  uint32_t v_uint32[2];
  float v_float32[2];
  int64_t v_int64;
  uint64_t v_uint64;
  double v_float64;
};
/*!
 * \brief Create a packed function from void addr types.
 *
 * \param f with signiture (ffi::PackedArgs args, ffi::Any* rv, void* void_args)
 * \param arg_types The arguments type information.
 * \param arg_extra_tags extra tags for the arguments
 * \tparam F the function type
 *
 * \return The wrapped packed function.
 */
template <typename F>
inline ffi::Function PackFuncVoidAddr(
    F f, const std::vector<DLDataType>& arg_types,
    const std::vector<FunctionInfo::ArgExtraTags>& arg_extra_tags = {});
/*!
 * \brief Create a packed function that from function only packs buffer arguments.
 *
 * \param f with signiture (ffi::PackedArgs args, ffi::Any* rv, ArgUnion* pack_args)
 * \param arg_types The arguments type information.
 * \tparam F the function type
 *
 * \return The wrapped packed function.
 */
template <typename F>
inline ffi::Function PackFuncNonBufferArg(F f, const std::vector<DLDataType>& arg_types);
/*!
 * \brief Create a packed function that from function that takes a packed arguments.
 *
 * This procedure ensures inserts padding to ensure proper alignment of struct fields
 * per C struct convention
 *
 * \param f with signature (ffi::PackedArgs args, ffi::Any* rv, void* pack_args, size_t nbytes)
 * \param arg_types The arguments that wish to get from
 * \tparam F the function type
 *
 * \return The wrapped packed function.
 */
template <typename F>
inline ffi::Function PackFuncPackedArgAligned(F f, const std::vector<DLDataType>& arg_types);
/*!
 * \brief Extract number of buffer argument from the argument types.
 * \param arg_types The argument types.
 * \return number of buffer arguments
 */
inline size_t NumBufferArgs(const std::vector<DLDataType>& arg_types);

// implementations details
namespace detail {
template <typename T, int kSize>
class TempArray {
 public:
  explicit TempArray(int size) {}
  T* data() { return data_; }

 private:
  T data_[kSize];
};
template <typename T>
class TempArray<T, 0> {
 public:
  explicit TempArray(int size) : data_(size) {}
  T* data() { return data_.data(); }

 private:
  std::vector<T> data_;
};

/*! \brief conversion code used in void arg. */
enum ArgConvertCode {
  INT64_TO_INT64,
  INT64_TO_INT32,
  INT64_TO_UINT32,
  FLOAT64_TO_FLOAT32,
  FLOAT64_TO_FLOAT64,
  HANDLE_TO_HANDLE,
  HANDLE_TO_TENSORMAP
};

inline ArgConvertCode GetArgConvertCode(DLDataType t) {
  ICHECK_EQ(t.lanes, 1U) << "Cannot pass vector type argument to device function for now";
  if (t.code == kDLInt) {
    if (t.bits == 64U) return INT64_TO_INT64;
    if (t.bits == 32U) return INT64_TO_INT32;
  } else if (t.code == kDLUInt) {
    if (t.bits == 32U) return INT64_TO_UINT32;
  } else if (t.code == kDLFloat) {
    if (t.bits == 64U) return FLOAT64_TO_FLOAT64;
    if (t.bits == 32U) return FLOAT64_TO_FLOAT32;
  } else if (t.code == kDLOpaqueHandle) {
    return HANDLE_TO_HANDLE;
  }
  LOG(FATAL) << "Cannot handle " << t << " as device function argument";
}

template <int N, typename F>
inline ffi::Function PackFuncVoidAddr_(F f, const std::vector<ArgConvertCode>& codes) {
  int num_args = static_cast<int>(codes.size());
  auto ret = [f, codes, num_args](ffi::PackedArgs args, ffi::Any* ret) {
    TempArray<void*, N> addr_(num_args);
    TempArray<ArgUnion32, N> holder_(num_args);
    void** addr = addr_.data();
    ArgUnion32* holder = holder_.data();
    // NOTE: we need the real address of the args.data for some addr translation
    const TVMFFIAny* raw_args = reinterpret_cast<const TVMFFIAny*>(args.data());

    for (int i = 0; i < num_args; ++i) {
      switch (codes[i]) {
        case INT64_TO_INT64:
        case FLOAT64_TO_FLOAT64:
        case HANDLE_TO_HANDLE: {
          addr[i] = (void*)&(raw_args[i].v_ptr);  // NOLINT(*)
          break;
        }
        case INT64_TO_INT32: {
          holder[i].v_int32 = static_cast<int32_t>(raw_args[i].v_int64);
          addr[i] = &(holder[i]);
          break;
        }
        case INT64_TO_UINT32: {
          holder[i].v_uint32 = static_cast<uint32_t>(raw_args[i].v_int64);
          addr[i] = &(holder[i]);
          break;
        }
        case FLOAT64_TO_FLOAT32: {
          holder[i].v_float32 = static_cast<float>(raw_args[i].v_float64);
          addr[i] = &(holder[i]);
          break;
        }
        case HANDLE_TO_TENSORMAP: {
          addr[i] = raw_args[i].v_ptr;
          break;
        }
      }
    }
    f(args, ret, addr);
  };
  return ffi::Function(ret);
}

template <int N, typename F>
inline ffi::Function PackFuncNonBufferArg_(F f, int base,
                                           const std::vector<ArgConvertCode>& codes) {
  int num_args = static_cast<int>(codes.size());
  auto ret = [f, codes, base, num_args](ffi::PackedArgs args, ffi::Any* ret) {
    TempArray<ArgUnion64, N> holder_(num_args);
    ArgUnion64* holder = holder_.data();
    // NOTE: we need the real address of the args.data for some addr translation
    const TVMFFIAny* raw_args = reinterpret_cast<const TVMFFIAny*>(args.data());

    for (int i = 0; i < num_args; ++i) {
      switch (codes[i]) {
        case INT64_TO_INT64: {
          holder[i].v_int64 = raw_args[base + i].v_int64;
          break;
        }
        case FLOAT64_TO_FLOAT64: {
          holder[i].v_float64 = raw_args[base + i].v_float64;
          break;
        }
        case INT64_TO_INT32: {
          holder[i].v_int32[0] = static_cast<int32_t>(raw_args[base + i].v_int64);
          break;
        }
        case INT64_TO_UINT32: {
          holder[i].v_uint32[0] = static_cast<uint32_t>(raw_args[base + i].v_int64);
          break;
        }
        case FLOAT64_TO_FLOAT32: {
          holder[i].v_float32[0] = static_cast<float>(raw_args[base + i].v_float64);
          break;
        }
        case HANDLE_TO_HANDLE:
        case HANDLE_TO_TENSORMAP: {
          LOG(FATAL) << "not reached";
          break;
        }
      }
    }
    f(args, ret, holder);
  };
  return ffi::Function(ret);
}

template <int N, typename F>
inline ffi::Function PackFuncPackedArgAligned_(F f, const std::vector<ArgConvertCode>& codes) {
  int num_args = static_cast<int>(codes.size());
  auto ret = [f, codes, num_args](ffi::PackedArgs args, ffi::Any* ret) {
    TempArray<uint64_t, N> pack_(num_args);
    int32_t* pack = reinterpret_cast<int32_t*>(pack_.data());
    int32_t* ptr = pack;
    static_assert(sizeof(void*) % sizeof(int32_t) == 0, "invariant");
    const TVMFFIAny* raw_args = reinterpret_cast<const TVMFFIAny*>(args.data());

    // function to ensure alignment so fields are properly aligned
    // factor: how many multiple of i32 we need to align to
    auto ensure_alignment_to_multiple_of_i32 = [&](size_t factor) {
      while ((ptr - pack) % factor != 0) {
        ++ptr;
      }
    };

    for (int i = 0; i < num_args; ++i) {
      switch (codes[i]) {
        case HANDLE_TO_HANDLE: {
          ensure_alignment_to_multiple_of_i32(sizeof(void*) / sizeof(int32_t));
          std::memcpy(ptr, &(raw_args[i].v_ptr), sizeof(void*));
          ptr += sizeof(void*) / sizeof(int32_t);
          break;
        }
        case INT64_TO_INT64:
        case FLOAT64_TO_FLOAT64: {
          ensure_alignment_to_multiple_of_i32(2);
          std::memcpy(ptr, &(raw_args[i].v_int64), sizeof(int64_t));
          ptr += 2;
          break;
        }
        case INT64_TO_INT32: {
          ensure_alignment_to_multiple_of_i32(1);
          *ptr = static_cast<int32_t>(raw_args[i].v_int64);
          ++ptr;
          break;
        }
        case INT64_TO_UINT32: {
          ensure_alignment_to_multiple_of_i32(1);
          *reinterpret_cast<uint32_t*>(ptr) = static_cast<uint32_t>(raw_args[i].v_int64);
          ++ptr;
          break;
        }
        case FLOAT64_TO_FLOAT32: {
          ensure_alignment_to_multiple_of_i32(1);
          *reinterpret_cast<float*>(ptr) = static_cast<float>(raw_args[i].v_float64);
          ++ptr;
          break;
        }
        case HANDLE_TO_TENSORMAP:
        default: {
          LOG(FATAL) << "not reached";
          break;
        }
      }
    }
    f(args, ret, pack, (ptr - pack) * sizeof(int32_t));
  };
  return ffi::Function(ret);
}
}  // namespace detail

template <typename F>
inline ffi::Function PackFuncVoidAddr(
    F f, const std::vector<DLDataType>& arg_types,
    const std::vector<FunctionInfo::ArgExtraTags>& arg_extra_tags) {
  std::vector<detail::ArgConvertCode> codes(arg_types.size());
  for (size_t i = 0; i < arg_types.size(); ++i) {
    if (arg_extra_tags.size() > i && arg_extra_tags[i] == FunctionInfo::ArgExtraTags::kTensorMap) {
      codes[i] = detail::HANDLE_TO_TENSORMAP;
    } else {
      codes[i] = detail::GetArgConvertCode(arg_types[i]);
    }
  }
  size_t num_void_args = arg_types.size();
  // specialization
  if (num_void_args <= 4) {
    return detail::PackFuncVoidAddr_<4>(f, codes);
  } else if (num_void_args <= 8) {
    return detail::PackFuncVoidAddr_<8>(f, codes);
  } else {
    return detail::PackFuncVoidAddr_<0>(f, codes);
  }
}

inline size_t NumBufferArgs(const std::vector<DLDataType>& arg_types) {
  size_t base = arg_types.size();
  for (size_t i = 0; i < arg_types.size(); ++i) {
    if (arg_types[i].code != kDLOpaqueHandle) {
      base = i;
      break;
    }
  }
  for (size_t i = base; i < arg_types.size(); ++i) {
    ICHECK(arg_types[i].code != kDLOpaqueHandle) << "Device function need to be organized";
  }
  return base;
}

template <typename F>
inline ffi::Function PackFuncNonBufferArg(F f, const std::vector<DLDataType>& arg_types) {
  size_t num_buffer = NumBufferArgs(arg_types);
  std::vector<detail::ArgConvertCode> codes;
  for (size_t i = num_buffer; i < arg_types.size(); ++i) {
    codes.push_back(detail::GetArgConvertCode(arg_types[i]));
  }
  int base = static_cast<int>(num_buffer);
  size_t nargs = codes.size();
  // specialization
  if (nargs <= 4) {
    return detail::PackFuncNonBufferArg_<4>(f, base, codes);
  } else {
    return detail::PackFuncNonBufferArg_<0>(f, base, codes);
  }
}

template <typename F>
inline ffi::Function PackFuncPackedArgAligned(F f, const std::vector<DLDataType>& arg_types) {
  std::vector<detail::ArgConvertCode> codes;
  for (size_t i = 0; i < arg_types.size(); ++i) {
    codes.push_back(detail::GetArgConvertCode(arg_types[i]));
  }
  size_t nargs = codes.size();
  // specialization
  if (nargs <= 4) {
    return detail::PackFuncPackedArgAligned_<4>(f, codes);
  } else {
    return detail::PackFuncPackedArgAligned_<0>(f, codes);
  }
}
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_PACK_ARGS_H_
