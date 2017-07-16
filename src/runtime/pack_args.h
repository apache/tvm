/*!
 *  Copyright (c) 2017 by Contributors
 * \file pack_args.h
 * \brief Utility to pack TVMArgs to other type-erased fution calling convention.
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

#include <tvm/runtime/c_runtime_api.h>
#include <vector>

namespace tvm {
namespace runtime {
/*!
 * \brief argument union type of 32bit.
 * Choose 32 bit because most GPU API do not work well with 64 bit.
 */
union ArgUnion {
  int32_t v_int32;
  uint32_t v_uint32;
  float v_float32;
};
/*!
 * \brief Create a packed function from void addr types.
 *
 * \param f with signiture (TVMArgs args, TVMRetValue* rv, void* void_args)
 * \param arg_types The arguments that wish to get from
 * \tparam F the function type
 *
 * \return The wrapped packed function.
 */
template<typename F>
inline PackedFunc PackFuncVoidAddr(F f, const std::vector<TVMType>& arg_types);
/*!
 * \brief Create a packed function that from function only packs buffer arguments.
 *
 * \param f with signiture (TVMArgs args, TVMRetValue* rv, ArgUnion* pack_args)
 * \param arg_types The arguments that wish to get from
 * \tparam F the function type
 *
 * \return The wrapped packed function.
 */
template<typename F>
inline PackedFunc PackFuncNonBufferArg(F f, const std::vector<TVMType>& arg_types);
/*!
 * \brief Extract number of buffer argument from the argument types.
 * \param arg_types The argument types.
 * \return number of buffer arguments
 */
inline size_t NumBufferArgs(const std::vector<TVMType>& arg_types);

// implementations details
namespace detail {
template<typename T, int kSize>
class TempArray {
 public:
  explicit TempArray(int size) {}
  T* data() {
    return data_;
  }
 private:
  T data_[kSize];
};
template<typename T>
class TempArray<T, 0> {
 public:
  explicit TempArray(int size) : data_(size) {}
  T* data() {
    return data_.data();
  }
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
  HANDLE_TO_HANDLE
};

inline ArgConvertCode GetArgConvertCode(TVMType t) {
  CHECK_EQ(t.lanes, 1U)
      << "Cannot pass vector type argument to devic function for now";
  if (t.code == kInt) {
    if (t.bits == 64U) return INT64_TO_INT64;
    if (t.bits == 32U) return INT64_TO_INT32;
  } else if (t.code == kUInt) {
    if (t.bits == 32U) return INT64_TO_UINT32;
  } else if (t.code == kFloat) {
    if (t.bits == 64U) return FLOAT64_TO_FLOAT64;
    if (t.bits == 32U) return FLOAT64_TO_FLOAT32;
  } else if (t.code == kHandle) {
    return HANDLE_TO_HANDLE;
  }
  LOG(FATAL) << "Cannot handle " << t << " as device function argument";
  return HANDLE_TO_HANDLE;
}

template<int N, typename F>
inline PackedFunc PackFuncVoidAddr_(F f, const std::vector<ArgConvertCode>& codes) {
  int num_args = static_cast<int>(codes.size());
  auto ret = [f, codes, num_args](TVMArgs args, TVMRetValue* ret) {
    TempArray<void*, N> addr_(num_args);
    TempArray<ArgUnion, N> holder_(num_args);
    void** addr = addr_.data();
    ArgUnion* holder = holder_.data();
    for (int i = 0; i < num_args; ++i) {
      switch (codes[i]) {
        case INT64_TO_INT64:
        case FLOAT64_TO_FLOAT64:
        case HANDLE_TO_HANDLE: {
          addr[i] = (void*)&(args.values[i]);  // NOLINT(*)
          break;
        }
        case INT64_TO_INT32: {
          holder[i].v_int32 = static_cast<int32_t>(args.values[i].v_int64);
          addr[i] = &(holder[i]);
          break;
        }
        case INT64_TO_UINT32 : {
          holder[i].v_uint32 = static_cast<uint32_t>(args.values[i].v_int64);
          addr[i] = &(holder[i]);
          break;
        }
        case FLOAT64_TO_FLOAT32: {
          holder[i].v_float32 = static_cast<float>(args.values[i].v_float64);
          addr[i] = &(holder[i]);
          break;
        }
      }
    }
    f(args, ret, addr);
  };
  return PackedFunc(ret);
}

template<int N, typename F>
inline PackedFunc PackFuncNonBufferArg_(
    F f, int base, const std::vector<ArgConvertCode>& codes) {
  int num_args = static_cast<int>(codes.size());
  auto ret = [f, codes, base, num_args](TVMArgs args, TVMRetValue* ret) {
    TempArray<ArgUnion, N> holder_(num_args);
    ArgUnion* holder = holder_.data();
    for (int i = 0; i < num_args; ++i) {
      switch (codes[i]) {
        case INT64_TO_INT64:
        case FLOAT64_TO_FLOAT64: {
          LOG(FATAL) << "Donot support 64bit argument to device function"; break;
        }
        case INT64_TO_INT32: {
          holder[i].v_int32 = static_cast<int32_t>(args.values[base + i].v_int64);
          break;
        }
        case INT64_TO_UINT32 : {
          holder[i].v_uint32 = static_cast<uint32_t>(args.values[base + i].v_int64);
          break;
        }
        case FLOAT64_TO_FLOAT32: {
          holder[i].v_float32 = static_cast<float>(args.values[base + i].v_float64);
          break;
        }
        case HANDLE_TO_HANDLE: {
          LOG(FATAL) << "not reached"; break;
        }
      }
    }
    f(args, ret, holder);
  };
  return PackedFunc(ret);
}
}  // namespace detail

template<typename F>
inline PackedFunc PackFuncVoidAddr(F f, const std::vector<TVMType>& arg_types) {
  std::vector<detail::ArgConvertCode> codes(arg_types.size());
  for (size_t i = 0; i < arg_types.size(); ++i) {
    codes[i] = detail::GetArgConvertCode(arg_types[i]);
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

inline size_t NumBufferArgs(const std::vector<TVMType>& arg_types) {
  size_t base = arg_types.size();
  for (size_t i = 0; i < arg_types.size(); ++i) {
    if (arg_types[i].code != kHandle) {
      base = i; break;
    }
  }
  for (size_t i = base; i < arg_types.size(); ++i) {
    CHECK(arg_types[i].code != kHandle)
        << "Device function need to be organized";
  }
  return base;
}

template<typename F>
inline PackedFunc PackFuncNonBufferArg(F f, const std::vector<TVMType>& arg_types) {
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
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_PACK_ARGS_H_
