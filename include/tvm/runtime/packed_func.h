/*!
 *  Copyright (c) 2016 by Contributors
 * \file packed_func.h
 * \brief Runtime related c++ class.
 */
#ifndef TVM_RUNTIME_PACKED_FUNC_H_
#define TVM_RUNTIME_PACKED_FUNC_H_

#include <functional>
#include <tuple>
#include <vector>
#include <string>
#include "./c_runtime_api.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Packed function is a type-erased function.
 *  The arguments are passed by packed format.
 *
 *  This is an useful unified interface to call generated functions,
 *  It is the unified function function type of TVM.
 *  It corresponds to TVMFunctionHandle in C runtime API.
 */
class PackedFunc {
 public:
  /*! \brief The internal std::function */
  using FType = std::function<void(const TVMValue* args, const int* type_codes, int num_args)>;
  /*! \brief default constructor */
  PackedFunc() {}
  /*!
   * \brief constructing a packed function from a std::function.
   * \param body the internal container of packed function.
   */
  explicit PackedFunc(FType body) : body_(body) {}
  /*!
   * \brief Call packed function by directly passing in unpacked format.
   * \param args Arguments to be passed.
   * \tparam Args arguments to be passed.
   */
  template<typename... Args>
  inline void operator()(Args&& ...args) const;
  /*!
   * \brief Call the function in packed format.
   * \param args The arguments
   * \param type_codes The type_codes of the arguments
   * \param num_args Number of arguments.
   */
  inline void CallPacked(const TVMValue* args, const int* type_codes, int num_args) const;
  /*! \return the internal body function */
  inline FType body() const;
  /*!
   * \brief Register f as into global function table
   * \param name The name of the function.
   * \param f The function to be registered.
   * \return Reference to the registered function.
   * \note The returned reference is valid until the end of the program
   */
  static const PackedFunc& RegisterGlobal(const std::string& name, PackedFunc f);
  /*!
   * \brief Get the global function by name.
   * \param name The name of the function.
   * \return reference to the registered function.
   */
  static const PackedFunc& GetGlobal(const std::string& name);
  /*!
   * \brief Get the names of currently registered global function.
   */
  static std::vector<std::string> ListGlobalNames();

 private:
  /*! \brief internal container of packed function */
  FType body_;
};

// implementations
inline void PackedFunc::CallPacked(
    const TVMValue* args, const int* type_codes, int num_args) const {
  body_(args, type_codes, num_args);
}

inline PackedFunc::FType PackedFunc::body() const {
  return body_;
}

template<bool stop, std::size_t I, typename F, typename ...Args>
struct for_each_dispatcher_ {
  static inline void run(const std::tuple<Args...>& args, F f) {
    f(I, std::get<I>(args));
    for_each_dispatcher_<(I + 1) == sizeof...(Args), (I+1), F, Args...>::run(args, f);
  }
};

template<std::size_t I, typename F, typename ...Args>
struct for_each_dispatcher_<true, I, F, Args...>  {
  static inline void run(const std::tuple<Args...>& args, F f) {}
};

template<typename F, typename ...Args>
inline void for_each(const std::tuple<Args...>& args, F f) {
  for_each_dispatcher_<sizeof...(Args) == 0, 0, F, Args...>::run(args, f);
}

namespace arg_setter {
template<typename T>
inline void Set(TVMValue& arg, int& t, T v);  // NOLINT(*)
template<>
inline void Set<double>(TVMValue& arg, int& t, double value) {  // NOLINT(*)
  arg.v_float64 = value;
  t = kFloat;
}
template<>
inline void Set<int>(TVMValue& arg, int& t, int value) {  // NOLINT(*)
  arg.v_int64 = value;
  t = kInt;
}
template<>
inline void Set<long>(TVMValue& arg, int& t, long value) {  // NOLINT(*)
  arg.v_int64 = value;
  t = kInt;
}
template<>
inline void Set<TVMArray*>(TVMValue& arg, int& t, TVMArray* value) {  // NOLINT(*)
  arg.v_handle = value;
  t = kHandle;
}
template<>
inline void Set<void*>(TVMValue& arg, int& t, void* value) {  // NOLINT(*)
  arg.v_handle = value;
  t = kHandle;
}
}  // namespace arg_setter

struct PackedFuncArgSetter {
  TVMValue* args;
  int* type_codes;
  template<typename T>
  inline void operator()(size_t i, T v) const {
    arg_setter::Set(args[i], type_codes[i], v);
  }
};

template<typename... Args>
inline void PackedFunc::operator()(Args&& ...args) const {
  auto targ = std::make_tuple(std::forward<Args>(args)...);
  const int kNumArgs = sizeof...(Args);
  TVMValue tvm_args[kNumArgs];
  int tvm_arg_type_ids[kNumArgs];
  for_each(targ, PackedFuncArgSetter{tvm_args, tvm_arg_type_ids});
  body_(tvm_args, tvm_arg_type_ids, kNumArgs);
}
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_PACKED_FUNC_H_
