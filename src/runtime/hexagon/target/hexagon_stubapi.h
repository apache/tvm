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

#ifndef TVM_RUNTIME_HEXAGON_TARGET_HEXAGON_STUBAPI_H_
#define TVM_RUNTIME_HEXAGON_TARGET_HEXAGON_STUBAPI_H_

#ifdef __ANDROID__
#include <AEEStdErr.h>
#include <dmlc/logging.h>
#include <rpcmem.h>
#include <stdint.h>

#include <tuple>

#include "fastrpc/tvm_hexagon_remote.h"
#include "fastrpc/tvm_hexagon_remote_nd.h"

namespace tvm {
namespace runtime {
namespace hexagon {

/*!
 * Unify the handling of domain and non-domain functions.
 *
 * In most cases, for a function "foo", the domain version will be called
 * "tvm_hexagon_remote_foo", and the non-domain version will have "nd_foo".
 * The interfaces will be the same, except:
 * - the domain version will take "remote_handle64" as the first parameter,
 *   while the non-domain version will not:
 *   int tvm_hexagon_remote_foo   (remote_handle64 h, param1, param2, ...);
 *   int tvm_hexagon_remote_nd_foo                   (param1, param2, ...);
 * - any parameter of type "buffer" in the IDL, will be converted into a
 *   type "tvm_hexagon_remote_buffer" for domain functions, and into
 *   "tvm_hexagon_remote_nd_buffer" for non-domain functions. These two
 *   types are identical, but since they are declared in two different IDLs,
 *   they get different names.
 *
 * For any function, only a pointer to the "buffer" type is passed, but
 * since the pointee types are different, this is enough to create a
 * difference in the function signatures even if the "remote_handle64"
 * parameter is ignored. For this reason, in all function types, the
 * types "tvm_hexagon_remote_buffer *" and "tvm_hexagon_remote_nd_buffer *",
 * both const and non-const, are replaced with "void *", with the
 * corresponding const-qualification. This is done by the templates
 * "replace_pointee_type" and "map_tuple_element" below.
 *
 * The following functions are subject to the uniform handling:
 *
 *   tvm_hexagon_remote_load_library     (remote_handle64 h, p1, p2, ...)
 *   tvm_hexagon_remote_release_library
 *   tvm_hexagon_remote_get_symbol
 *   tvm_hexagon_remote_kernel
 *   tvm_hexagon_remote_close
 *   tvm_hexagon_remote_alloc_vtcm
 *   tvm_hexagon_remote_free_vtcm
 *
 *   tvm_hexagon_remote_nd_load_library  (p1, p2, ...)
 *   tvm_hexagon_remote_nd_release_library
 *   tvm_hexagon_remote_nd_get_symbol
 *   tvm_hexagon_remote_nd_kernel
 *   tvm_hexagon_remote_nd_close
 *
 * The "open" functions differ in their parameters in different ways, and
 * need to be handled individually.
 *
 *   tvm_hexagon_remote_open
 *   tvm_hexagon_remote_nd_open
 */

namespace {
/*!
 * replace_pointee_type<T, M, V>
 *
 * If T is a pointer to a potentially const-qualified M, then replace
 * M in T with V. Otherwise, leave T unchanged.
 */
template <typename T, typename M, typename V>
struct replace_pointee_type {
  using type = T;
};

template <typename M, typename V>
struct replace_pointee_type<M*, M, V> {
  using type = V*;
};

template <typename M, typename V>
struct replace_pointee_type<const M*, M, V> {
  using type = const V*;
};

/*!
 * map_tuple_elements<M, V, std::tuple<As...>>
 *
 * From given tuple <As...>, form another tuple where for each A in As,
 * if A contains a pointer to M, the pointer is replaced with a pointer
 * to V, leaving other types unchanged.
 */
template <typename...>
struct map_tuple_elements;

template <typename M, typename V, typename... As>
struct map_tuple_elements<M, V, std::tuple<As...>> {
  using type = std::tuple<typename replace_pointee_type<As, M, V>::type...>;
};

/*!
 * map_func_type<M, V, F>
 *
 * Given function type F = R(As...), form another function type by replacing
 * each pointer to M with a pointer to V.
 */
template <typename M, typename V, typename F>
struct map_func_type {
  template <typename...>
  struct func_to_tuple;
  template <typename R, typename... As>
  struct func_to_tuple<R(As...)> {
    using args = std::tuple<As...>;
    using ret = R;
  };

  template <typename R, typename... As>
  struct tuple_to_func;
  template <typename R, typename... As>
  struct tuple_to_func<R, std::tuple<As...>> {
    using func = R(As...);
  };

  using arg_tuple = typename func_to_tuple<F>::args;
  using ret_type = typename func_to_tuple<F>::ret;
  using mapped_args = typename map_tuple_elements<M, V, arg_tuple>::type;
  using type = typename tuple_to_func<ret_type, mapped_args>::func;
};
}  // namespace

class StubAPI {
 public:
  StubAPI();
  ~StubAPI();

 private:
  // Create types for each remote function. For functions that take
  // a pointer to tvm_hexagon_remote_buffer or tvm_hexagon_remote_nd_buffer,
  // replace that pointer with pointer to void to make pointers to these
  // two types identical in the function types created below.
  // For example, int foo(tvm_hexagon_remote_buffer*) and
  // int bar(tvm_hexagon_remote_nd_buffer*) should both have the same type.
#define MAPTYPE(fn, ty) \
  using fn##_t = typename map_func_type<ty, void, decltype(::fn)>::type;
  MAPTYPE(tvm_hexagon_remote_load_library, tvm_hexagon_remote_buffer)
  MAPTYPE(tvm_hexagon_remote_release_library, tvm_hexagon_remote_buffer)
  MAPTYPE(tvm_hexagon_remote_get_symbol, tvm_hexagon_remote_buffer)
  MAPTYPE(tvm_hexagon_remote_kernel, tvm_hexagon_remote_buffer)
  MAPTYPE(tvm_hexagon_remote_close, tvm_hexagon_remote_buffer)
  MAPTYPE(tvm_hexagon_remote_alloc_vtcm, tvm_hexagon_remote_buffer)
  MAPTYPE(tvm_hexagon_remote_free_vtcm, tvm_hexagon_remote_buffer)
  MAPTYPE(tvm_hexagon_remote_call_mmap64, tvm_hexagon_remote_buffer)

  MAPTYPE(tvm_hexagon_remote_nd_load_library, tvm_hexagon_remote_nd_buffer)
  MAPTYPE(tvm_hexagon_remote_nd_release_library, tvm_hexagon_remote_nd_buffer)
  MAPTYPE(tvm_hexagon_remote_nd_get_symbol, tvm_hexagon_remote_nd_buffer)
  MAPTYPE(tvm_hexagon_remote_nd_kernel, tvm_hexagon_remote_nd_buffer)
  MAPTYPE(tvm_hexagon_remote_nd_close, tvm_hexagon_remote_buffer)
  MAPTYPE(tvm_hexagon_remote_nd_call_mmap64, tvm_hexagon_remote_buffer)
#undef MAPTYPE

  // For remote functions whose prototypes differ significantly between
  // the domain and non-domain versions, create the types directly.
#define DECLTYPE(fn) using fn##_t = decltype(::fn);
  DECLTYPE(tvm_hexagon_remote_open)
  DECLTYPE(tvm_hexagon_remote_nd_open)

  DECLTYPE(rpcmem_init)
  DECLTYPE(rpcmem_deinit)
  DECLTYPE(rpcmem_alloc)
  DECLTYPE(rpcmem_free)
  DECLTYPE(rpcmem_to_fd)
#undef DECLTYPE

 public:
  template <typename Fd, typename Fnd, typename... Ts>
  int invoke(Fd func_d, Fnd func_nd, remote_handle64 handle,
             Ts... args) const {
    if (enable_domains_) {
      return func_d(handle, args...);
    }
    return func_nd(args...);
  }
  template <typename Fd, typename... Ts>
  int invoke_d(Fd func_d, remote_handle64 handle, Ts... args) const {
    if (enable_domains_) {
      return func_d(handle, args...);
    }
    return 0;
  }

#define CONCAT_STR_FOR_REAL(a, b) a##b
#define CONCAT_STR(a, b) CONCAT_STR_FOR_REAL(a, b)

#define FUNC(name) CONCAT_STR(tvm_hexagon_remote_, name)
#define FUNC_D(name) CONCAT_STR(tvm_hexagon_remote_, name)
#define FUNC_ND(name) CONCAT_STR(tvm_hexagon_remote_nd_, name)
#define PTRNAME(fn) CONCAT_STR(p, CONCAT_STR(fn, _))

#define DECLFUNC(name)                                                   \
  template <typename... Ts>                                              \
  int FUNC(name)(remote_handle64 handle, Ts... args) const {             \
    return invoke(PTRNAME(FUNC_D(name)), PTRNAME(FUNC_ND(name)), handle, \
                  args...);                                              \
  }

#define DECLFUNC_D(name)                                     \
  template <typename... Ts>                                  \
  int FUNC(name)(remote_handle64 handle, Ts... args) const { \
    return invoke_d(PTRNAME(FUNC_D(name)), handle, args...); \
  }

  DECLFUNC(load_library)
  DECLFUNC(release_library)
  DECLFUNC(get_symbol)
  DECLFUNC(kernel)
  DECLFUNC(close)
  DECLFUNC_D(alloc_vtcm)
  DECLFUNC_D(free_vtcm)
  DECLFUNC(call_mmap64)
#undef DECLFUNC

// Implementations provided here in case the target does not have these
// in lib[ac]dsprpc.so.
#define DECLSFUNC(fn) \
  fn##_t* fn##_ptr() const { return p##fn##_; }
  DECLSFUNC(rpcmem_init)
  DECLSFUNC(rpcmem_deinit)
  DECLSFUNC(rpcmem_alloc)
  DECLSFUNC(rpcmem_free)
  DECLSFUNC(rpcmem_to_fd)
#undef DECLSFUNC
#undef DECLFUNC_D

  int tvm_hexagon_remote_open(const char* uri, remote_handle64* handle) const {
    if (enable_domains_) {
      return PTRNAME(tvm_hexagon_remote_open)(uri, handle);
    }
    return PTRNAME(tvm_hexagon_remote_nd_open)();
  }

  static const StubAPI* Global();

 private:
  bool enable_domains_ = true;
  void* lib_handle_ = nullptr;

#define DECLPTR(fn) fn##_t* PTRNAME(fn) = nullptr
  DECLPTR(tvm_hexagon_remote_load_library);
  DECLPTR(tvm_hexagon_remote_release_library);
  DECLPTR(tvm_hexagon_remote_get_symbol);
  DECLPTR(tvm_hexagon_remote_kernel);
  DECLPTR(tvm_hexagon_remote_open);
  DECLPTR(tvm_hexagon_remote_close);
  DECLPTR(tvm_hexagon_remote_alloc_vtcm);
  DECLPTR(tvm_hexagon_remote_free_vtcm);
  DECLPTR(tvm_hexagon_remote_call_mmap64);

  DECLPTR(tvm_hexagon_remote_nd_load_library);
  DECLPTR(tvm_hexagon_remote_nd_release_library);
  DECLPTR(tvm_hexagon_remote_nd_get_symbol);
  DECLPTR(tvm_hexagon_remote_nd_kernel);
  DECLPTR(tvm_hexagon_remote_nd_open);
  DECLPTR(tvm_hexagon_remote_nd_close);
  DECLPTR(tvm_hexagon_remote_nd_call_mmap64);
#undef DECLPTR

// "System" functions.
#define DECLSPTR(fn) fn##_t* p##fn##_ = nullptr;
  // Implementations provided here in case the target does not have these
  // in lib[ac]dsprpc.so.
  DECLSPTR(rpcmem_init);
  DECLSPTR(rpcmem_deinit);
  DECLSPTR(rpcmem_alloc);
  DECLSPTR(rpcmem_free);
  DECLSPTR(rpcmem_to_fd);
#undef DECLSPTR

#undef PTRNAME
#undef FUNC_ND
#undef FUNC_D
#undef FUNC
#undef CONCAT_STR
#undef CONCAT_STR_FOR_REAL

  template <typename T>
  T GetSymbol(const char* sym);
};

}  // namespace hexagon

}  // namespace runtime
}  // namespace tvm

#endif  // __ANDROID__
#endif  // TVM_RUNTIME_HEXAGON_TARGET_HEXAGON_STUBAPI_H_
