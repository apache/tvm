/*!
 *  Copyright (c) 2017 by Contributors
 * \file void_addr_args.h
 * \brief Utility to convert TVMArgs to void* array type-erasure function call.
 *
 *  Array of argument address is a typical way of type-erasure for functions.
 *  The function signiture looks like function(void** args, int num_args);
 *  Where args takes the address of each input.
 */
#ifndef TVM_RUNTIME_VOID_ADDR_ARGS_H_
#define TVM_RUNTIME_VOID_ADDR_ARGS_H_

#include <tvm/runtime/c_runtime_api.h>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief Create a packed function from void addr types
 * \param f with signiture (TVMArgs args, TVMRetValue* rv, void* void_args)
 * \param arg_types The arguments that wish to get from
 * \tparam T the function type
 *
 * \return The wrapped packed function.
 */
template<typename F>
inline PackedFunc PackFromVoidAddrArgs(
    F f, const std::vector<TVMType>& arg_types);

// implementations details
namespace detail {
/*!
 * \brief void addr argument data content
 *  holder in case conversion is needed.
 */
union VoidArgHolder {
  int32_t v_int32;
  uint32_t v_uint32;
  float v_float32;
};

template<int MAX_NARG>
class VoidAddrArray {
 public:
  explicit VoidAddrArray(int num_args) {
  }
  void** addr() {
    return addr_;
  }
  VoidArgHolder* holder() {
    return holder_;
  }

 private:
  void* addr_[MAX_NARG];
  VoidArgHolder holder_[MAX_NARG];
};

template<>
class VoidAddrArray<0> {
 public:
  explicit VoidAddrArray(int num_args)
      : addr_(num_args), holder_(num_args) {
  }
  void** addr() {
    return addr_.data();
  }
  VoidArgHolder* holder() {
    return holder_.data();
  }

 private:
  std::vector<void*> addr_;
  std::vector<VoidArgHolder> holder_;
};

/*! \brief conversion code used in void arg. */
enum VoidArgConvertCode {
  INT64_TO_INT64,
  INT64_TO_INT32,
  INT64_TO_UINT32,
  FLOAT64_TO_FLOAT32,
  FLOAT64_TO_FLOAT64,
  HANDLE_TO_HANDLE
};

template<int N, typename F>
inline PackedFunc PackFromVoidAddrArgs_(
    F f, const std::vector<VoidArgConvertCode>& codes) {
  int num_args = static_cast<int>(codes.size());
  auto ret = [f, codes, num_args](TVMArgs args, TVMRetValue* ret) {
    VoidAddrArray<N> temp(num_args);
    void** addr = temp.addr();
    VoidArgHolder* holder = temp.holder();
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

inline VoidArgConvertCode GetVoidArgConvertCode(TVMType t) {
  CHECK_EQ(t.lanes, 1U);
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
  LOG(FATAL) << "Cannot handle " << t;
  return HANDLE_TO_HANDLE;
}

}  // namespace detail

template<typename F>
inline PackedFunc PackFromVoidAddrArgs(
  F f, const std::vector<TVMType>& arg_types) {
  std::vector<detail::VoidArgConvertCode> codes(arg_types.size());
  for (size_t i = 0; i < arg_types.size(); ++i) {
    codes[i] = detail::GetVoidArgConvertCode(arg_types[i]);
  }
  size_t num_void_args = arg_types.size();
  // specialization
  if (num_void_args <= 4) {
    return detail::PackFromVoidAddrArgs_<4>(f, codes);
  } else if (num_void_args <= 8) {
    return detail::PackFromVoidAddrArgs_<8>(f, codes);
  } else {
    return detail::PackFromVoidAddrArgs_<0>(f, codes);
  }
}
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_VOID_ADDR_ARGS_H_
