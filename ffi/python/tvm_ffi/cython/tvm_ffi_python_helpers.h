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
/*
 * \file tvm_ffi_python_helpers.h
 * \brief C++ based helpers for the Python FFI call to optimize performance.
 */
#ifndef TVM_FFI_PYTHON_HELPERS_H_
#define TVM_FFI_PYTHON_HELPERS_H_

#include <Python.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <cstring>
#include <exception>
#include <iostream>
#include <unordered_map>

//----------------------------------------------------------
// Extra support for DLPack
//----------------------------------------------------------
/*!
 * \brief C-style function pointer to speed convert a PyObject Tensor to a DLManagedTensorVersioned.
 * \param py_obj The Python object to convert, this should be PyObject*
 * \param out The output DLManagedTensorVersioned.
 * \param env_stream Outputs the current context stream of the device provided by the tensor.
 * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
 * \note We use void* to avoid dependency on Python.h so this specific type is
 *       not dependent on Python.h and can be copied to dlpack.h
 */
typedef int (*DLPackFromPyObject)(void* py_obj, DLManagedTensorVersioned** out, void** env_stream);
/*!
 * \brief C-style function pointer to speed convert a DLManagedTensorVersioned to a PyObject Tensor.
 * \param tensor The DLManagedTensorVersioned to convert.
 * \param py_obj_out The output Python object.
 * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
 * \note We use void* to avoid dependency on Python.h so this specific type is
 *       not dependent on Python.h and can be copied to dlpack.h
 */
typedef int (*DLPackToPyObject)(DLManagedTensorVersioned* tensor, void** py_obj_out);

///--------------------------------------------------------------------------------
/// We deliberately designed the data structure and function to be C-style
//  prefixed with TVMFFIPy so they can be easily invoked through Cython.
///--------------------------------------------------------------------------------

/*!
 * \brief Context for each ffi call to track the stream, device and temporary arguments.
 */
struct TVMFFIPyCallContext {
  /*! \brief The workspace for the packed arguments */
  TVMFFIAny* packed_args = nullptr;
  /*! \brief Detected device type, if any */
  int device_type = -1;
  /*! \brief Detected device id, if any */
  int device_id = 0;
  /*! \brief Detected stream, if any */
  void* stream = nullptr;
  /*! \brief the temporary arguments to be recycled */
  void** temp_ffi_objects = nullptr;
  /*! \brief the number of temporary arguments */
  int num_temp_ffi_objects = 0;
  /*! \brief the temporary arguments to be recycled */
  void** temp_py_objects = nullptr;
  /*! \brief the number of temporary arguments */
  int num_temp_py_objects = 0;
  /*! \brief the DLPack exporter, if any */
  DLPackToPyObject c_dlpack_to_pyobject{nullptr};
  /*! \brief the DLPack allocator, if any */
  DLPackTensorAllocator c_dlpack_tensor_allocator{nullptr};
};

/*! \brief Argument setter for a given python argument. */
struct TVMFFIPyArgSetter {
  /*!
   * \brief Function pointer to invoke the setter.
   * \param self Pointer to this, this should be TVMFFIPyArgSetter*
   * \param call_ctx The call context.
   * \param arg The python argument to be set
   * \param out The output argument.
   * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
   */
  int (*func)(TVMFFIPyArgSetter* self, TVMFFIPyCallContext* call_ctx, PyObject* arg,
              TVMFFIAny* out);
  /*!
   * \brief Optional DLPack exporter for for setters that leverages DLPack protocol.
   */
  DLPackFromPyObject c_dlpack_from_pyobject{nullptr};
  /*!
   * \brief Optional DLPack importer for for setters that leverages DLPack protocol.
   */
  DLPackToPyObject c_dlpack_to_pyobject{nullptr};
  /*!
   * \brief Optional DLPack allocator for for setters that leverages DLPack protocol.
   */
  DLPackTensorAllocator c_dlpack_tensor_allocator{nullptr};
  /*!
   * \brief Invoke the setter.
   * \param call_ctx The call context.
   * \param arg The python argument to be set
   * \param out The output argument.
   * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
   */
  int operator()(TVMFFIPyCallContext* call_ctx, PyObject* arg, TVMFFIAny* out) const {
    return (*func)(const_cast<TVMFFIPyArgSetter*>(this), call_ctx, arg, out);
  }
};

//---------------------------------------------------------------------------------------------
// The following section contains predefined setters for common POD types
// They ar not meant to be used directly, but instead being registered to TVMFFIPyCallManager
//---------------------------------------------------------------------------------------------
int TVMFFIPyArgSetterFloat_(TVMFFIPyArgSetter*, TVMFFIPyCallContext*, PyObject* arg,
                            TVMFFIAny* out) noexcept {
  out->type_index = kTVMFFIFloat;
  // this function getsdispatched when type is already float, so no need to worry about error
  out->v_float64 = PyFloat_AsDouble(arg);
  return 0;
}

int TVMFFIPyArgSetterInt_(TVMFFIPyArgSetter*, TVMFFIPyCallContext*, PyObject* arg,
                          TVMFFIAny* out) noexcept {
  int overflow = 0;
  out->type_index = kTVMFFIInt;
  out->v_int64 = PyLong_AsLongLongAndOverflow(arg, &overflow);

  if (overflow != 0) {
    PyErr_SetString(PyExc_OverflowError, "Python int too large to convert to int64_t");
    return -1;
  }
  return 0;
}

int TVMFFIPyArgSetterBool_(TVMFFIPyArgSetter*, TVMFFIPyCallContext*, PyObject* arg,
                           TVMFFIAny* out) noexcept {
  out->type_index = kTVMFFIBool;
  // this function getsdispatched when type is already bool, so no need to worry about error
  out->v_int64 = PyLong_AsLong(arg);
  return 0;
}

int TVMFFIPyArgSetterNone_(TVMFFIPyArgSetter*, TVMFFIPyCallContext*, PyObject* arg,
                           TVMFFIAny* out) noexcept {
  out->type_index = kTVMFFINone;
  out->v_int64 = 0;
  return 0;
}

//---------------------------------------------------------------------------------------------
// The following section contains the dispatcher logic for function calling
//---------------------------------------------------------------------------------------------
/*!
 * \brief Factory function that creates an argument setter for a given Python argument type.
 *
 * This factory function analyzes a Python argument and creates an appropriate setter
 * that can convert Python objects of the same type to C arguments for TVM FFI calls.
 * The setter will be cached for future use for setting argument of the same type.
 *
 * \param arg The Python argument value used as a type example.
 * \param out Output parameter that receives the created argument setter.
 * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
 *
 * \note This is a callback function supplied by the caller. The factory must satisfy
 *       the invariance that the same setter can be used for other arguments with
 *       the same type as the provided example argument.
 */
typedef int (*TVMFFIPyArgSetterFactory)(PyObject* arg, TVMFFIPyArgSetter* out);

/*!
 * \brief A manager class that handles python ffi calls.
 */
class TVMFFIPyCallManager {
 public:
  /*!
   * \brief Get the thread local call manager.
   * \return The thread local call manager.
   */
  static TVMFFIPyCallManager* ThreadLocal() {
    static thread_local TVMFFIPyCallManager inst;
    return &inst;
  }
  /*!
   * \brief auxiliary class that manages the call stack in RAII manner.
   *
   * In most cases, it will try to allocate from temp_stack,
   * then allocate from heap if the request goes beyond the stack size.
   */
  class CallStack : public TVMFFIPyCallContext {
   public:
    CallStack(TVMFFIPyCallManager* manager, int64_t num_args) : manager_ptr_(manager) {
      static_assert(sizeof(TVMFFIAny) >= (sizeof(void*) * 2));
      static_assert(alignof(TVMFFIAny) % alignof(void*) == 0);
      old_stack_top_ = manager->stack_top_;
      int64_t requested_count = num_args * 2;
      TVMFFIAny* stack_head = manager->temp_stack_.data() + manager->stack_top_;
      if (manager->stack_top_ + requested_count >
          static_cast<int64_t>(manager->temp_stack_.size())) {
        // allocate from heap
        heap_ptr_ = new TVMFFIAny[requested_count];
        stack_head = heap_ptr_;
      } else {
        manager->stack_top_ += requested_count;
      }
      this->packed_args = stack_head;
      this->temp_ffi_objects = reinterpret_cast<void**>(stack_head + num_args);
      this->temp_py_objects = this->temp_ffi_objects + num_args;
    }

    ~CallStack() {
      try {
        // recycle the temporary arguments if any
        for (int i = 0; i < this->num_temp_ffi_objects; ++i) {
          TVMFFIObject* obj = static_cast<TVMFFIObject*>(this->temp_ffi_objects[i]);
          if (obj->deleter != nullptr) {
            obj->deleter(obj, kTVMFFIObjectDeleterFlagBitMaskBoth);
          }
        }
        for (int i = 0; i < this->num_temp_py_objects; ++i) {
          Py_DecRef(static_cast<PyObject*>(this->temp_py_objects[i]));
        }
      } catch (const std::exception& ex) {
        // very rare, catch c++ exception and set python error
        PyErr_SetString(PyExc_RuntimeError, ex.what());
      }
      // now recycle the memory of the call stack
      if (heap_ptr_ == nullptr) {
        manager_ptr_->stack_top_ = old_stack_top_;
      } else {
        delete[] heap_ptr_;
      }
    }

   private:
    /*!
     *\brief The manager of the call stack
     * If stored on stack, must set it to point to parent.
     */
    TVMFFIPyCallManager* manager_ptr_ = nullptr;
    /*! \brief The heap of the call stack */
    TVMFFIAny* heap_ptr_ = nullptr;
    /*! \brief The old stack size */
    int64_t old_stack_top_ = 0;
  };

  /*!
   * \brief Call a function with a variable number of arguments
   * \param setter_factory The factory function to create the setter
   * \param func_handle The handle of the function to call
   * \param py_arg_tuple The arguments to the function
   * \param result The result of the function
   * \param c_api_ret_code The return code of the C-call
   * \param release_gil Whether to release the GIL
   * \param optional_out_dlpack_importer The DLPack importer to be used for the result
   * \return 0 on when there is no python error, -1 on python error
   * \note When an error happens on FFI side, we should return 0 and set c_api_ret_code
   */
  int Call(TVMFFIPyArgSetterFactory setter_factory, void* func_handle, PyObject* py_arg_tuple,
           TVMFFIAny* result, int* c_api_ret_code, bool release_gil,
           DLPackToPyObject* optional_out_dlpack_importer) {
    int64_t num_args = PyTuple_Size(py_arg_tuple);
    if (num_args == -1) return -1;
    try {
      // allocate a call stack
      CallStack ctx(this, num_args);
      // Iterate over the arguments and set them
      for (int64_t i = 0; i < num_args; ++i) {
        PyObject* py_arg = PyTuple_GetItem(py_arg_tuple, i);
        TVMFFIAny* c_arg = ctx.packed_args + i;
        if (SetArgument(setter_factory, &ctx, py_arg, c_arg) != 0) return -1;
      }
      TVMFFIStreamHandle prev_stream = nullptr;
      DLPackTensorAllocator prev_tensor_allocator = nullptr;
      // setup stream context if needed
      if (ctx.device_type != -1) {
        c_api_ret_code[0] =
            TVMFFIEnvSetStream(ctx.device_type, ctx.device_id, ctx.stream, &prev_stream);
        // setting failed, directly return
        if (c_api_ret_code[0] != 0) return 0;
      }
      if (ctx.c_dlpack_tensor_allocator != nullptr) {
        c_api_ret_code[0] =
            TVMFFIEnvSetTensorAllocator(ctx.c_dlpack_tensor_allocator, 0, &prev_tensor_allocator);
        if (c_api_ret_code[0] != 0) return 0;
      }
      // call the function
      if (release_gil) {
        // release the GIL
        Py_BEGIN_ALLOW_THREADS;
        c_api_ret_code[0] = TVMFFIFunctionCall(func_handle, ctx.packed_args, num_args, result);
        Py_END_ALLOW_THREADS;
      } else {
        c_api_ret_code[0] = TVMFFIFunctionCall(func_handle, ctx.packed_args, num_args, result);
      }
      // restore the original stream
      if (ctx.device_type != -1 && prev_stream != ctx.stream) {
        // always try recover first, even if error happens
        if (TVMFFIEnvSetStream(ctx.device_type, ctx.device_id, prev_stream, nullptr) != 0) {
          // recover failed, set python error
          PyErr_SetString(PyExc_RuntimeError, "Failed to recover stream");
          return -1;
        }
      }
      if (prev_tensor_allocator != ctx.c_dlpack_tensor_allocator) {
        c_api_ret_code[0] = TVMFFIEnvSetTensorAllocator(prev_tensor_allocator, 0, nullptr);
        if (c_api_ret_code[0] != 0) return 0;
      }
      if (optional_out_dlpack_importer != nullptr && ctx.c_dlpack_to_pyobject != nullptr) {
        *optional_out_dlpack_importer = ctx.c_dlpack_to_pyobject;
      }
      return 0;
    } catch (const std::exception& ex) {
      // very rare, catch c++ exception and set python error
      PyErr_SetString(PyExc_RuntimeError, ex.what());
      return -1;
    }
  }

  int SetField(TVMFFIPyArgSetterFactory setter_factory, TVMFFIFieldSetter field_setter,
               void* field_ptr, PyObject* py_arg, int* c_api_ret_code) {
    try {
      CallStack ctx(this, 1);
      TVMFFIAny* c_arg = ctx.packed_args;
      if (SetArgument(setter_factory, &ctx, py_arg, c_arg) != 0) return -1;
      c_api_ret_code[0] = (*field_setter)(field_ptr, c_arg);
      return 0;
    } catch (const std::exception& ex) {
      // very rare, catch c++ exception and set python error
      PyErr_SetString(PyExc_RuntimeError, ex.what());
      return -1;
    }
  }

  int PyObjectToFFIAny(TVMFFIPyArgSetterFactory setter_factory, PyObject* py_arg, TVMFFIAny* out,
                       int* c_api_ret_code) {
    try {
      CallStack ctx(this, 1);
      TVMFFIAny* c_arg = ctx.packed_args;
      if (SetArgument(setter_factory, &ctx, py_arg, c_arg) != 0) return -1;
      c_api_ret_code[0] = TVMFFIAnyViewToOwnedAny(c_arg, out);
      return 0;
    } catch (const std::exception& ex) {
      // very rare, catch c++ exception and set python error
      PyErr_SetString(PyExc_RuntimeError, ex.what());
      return -1;
    }
  }
  /*!
   * \brief Get the size of the dispatch map
   * \return The size of the dispatch map
   */
  size_t GetDispatchMapSize() const { return dispatch_map_.size(); }

 private:
  TVMFFIPyCallManager() {
    static constexpr size_t kDefaultDispatchCapacity = 32;
    static constexpr size_t kDefaultStackSize = 32;
    dispatch_map_.reserve(kDefaultDispatchCapacity);
    temp_stack_.resize(kDefaultStackSize * 2);
  }
  /*!
   * \brief Set an py_arg to out.
   * \param setter_factory The factory function to create the setter
   * \param ctx The call context
   * \param py_arg The python argument to be set
   * \param out The output argument
   * \return 0 on success, -1 on failure
   */
  int SetArgument(TVMFFIPyArgSetterFactory setter_factory, TVMFFIPyCallContext* ctx,
                  PyObject* py_arg, TVMFFIAny* out) {
    PyTypeObject* py_type = Py_TYPE(py_arg);
    // pre-zero the output argument, modulo the type index
    out->type_index = kTVMFFINone;
    out->zero_padding = 0;
    out->v_int64 = 0;
    // find the pre-cached setter
    // This class is thread-local, so we don't need to worry about race condition
    auto it = dispatch_map_.find(py_type);
    if (it != dispatch_map_.end()) {
      TVMFFIPyArgSetter setter = it->second;
      // if error happens, propagate it back
      if (setter(ctx, py_arg, out) != 0) return -1;
    } else {
      // no dispatch found, query and create a new one.
      TVMFFIPyArgSetter setter;
      // propagate python error back
      if (setter_factory(py_arg, &setter) != 0) {
        return -1;
      }
      // update dispatch table
      dispatch_map_.emplace(py_type, setter);
      if (setter(ctx, py_arg, out) != 0) return -1;
    }
    return 0;
  }
  // internal dispacher
  std::unordered_map<PyTypeObject*, TVMFFIPyArgSetter> dispatch_map_;
  // temp call stack
  std::vector<TVMFFIAny> temp_stack_;
  int64_t stack_top_ = 0;
};

/*!
 * \brief Call a function with a variable number of arguments
 * \param setter_factory The factory function to create the setter
 * \param func_handle The handle of the function to call
 * \param py_arg_tuple The arguments to the function
 * \param result The result of the function
 * \param c_api_ret_code The return code of the function
 * \param release_gil Whether to release the GIL
 * \param out_dlpack_exporter The DLPack exporter to be used for the result
 * \return 0 on success, nonzero on failure
 */
inline int TVMFFIPyFuncCall(TVMFFIPyArgSetterFactory setter_factory, void* func_handle,
                            PyObject* py_arg_tuple, TVMFFIAny* result, int* c_api_ret_code,
                            bool release_gil = true,
                            DLPackToPyObject* out_dlpack_importer = nullptr) {
  return TVMFFIPyCallManager::ThreadLocal()->Call(setter_factory, func_handle, py_arg_tuple, result,
                                                  c_api_ret_code, release_gil, out_dlpack_importer);
}

/*!
 * \brief Set a field of a FFI object
 * \param setter_factory The factory function to create the setter
 * \param field_setter The field setter function
 * \param field_ptr The pointer to the field
 * \param py_arg The python argument to be set
 * \param c_api_ret_code The return code of the function
 * \return 0 on success, nonzero on failure
 */
inline int TVMFFIPyCallFieldSetter(TVMFFIPyArgSetterFactory setter_factory,
                                   TVMFFIFieldSetter field_setter, void* field_ptr,
                                   PyObject* py_arg, int* c_api_ret_code) {
  return TVMFFIPyCallManager::ThreadLocal()->SetField(setter_factory, field_setter, field_ptr,
                                                      py_arg, c_api_ret_code);
}

/*!
 * \brief Convert a Python object to a FFI Any
 * \param setter_factory The factory function to create the setter
 * \param py_arg The python argument to be set
 * \param out The output argument
 * \param c_api_ret_code The return code of the function
 * \return 0 on success, nonzero on failure
 */
inline int TVMFFIPyPyObjectToFFIAny(TVMFFIPyArgSetterFactory setter_factory, PyObject* py_arg,
                                    TVMFFIAny* out, int* c_api_ret_code) {
  return TVMFFIPyCallManager::ThreadLocal()->PyObjectToFFIAny(setter_factory, py_arg, out,
                                                              c_api_ret_code);
}

/*!
 * \brief Get the size of the dispatch map
 * \return The size of the dispatch map
 */
inline size_t TVMFFIPyGetDispatchMapSize() {
  return TVMFFIPyCallManager::ThreadLocal()->GetDispatchMapSize();
}

/*!
 * \brief Push a temporary FFI object to the call context that will be recycled after the call
 * \param ctx The call context
 * \param arg The FFI object to push
 */
inline void TVMFFIPyPushTempFFIObject(TVMFFIPyCallContext* ctx, TVMFFIObjectHandle arg) noexcept {
  // invariance: each ArgSetter can have at most one temporary Python object
  // so it ensures that we won't overflow the temporary Python object stack
  ctx->temp_ffi_objects[ctx->num_temp_ffi_objects++] = arg;
}

/*!
 * \brief Push a temporary Python object to the call context that will be recycled after the call
 * \param ctx The call context
 * \param arg The Python object to push
 */
inline void TVMFFIPyPushTempPyObject(TVMFFIPyCallContext* ctx, PyObject* arg) noexcept {
  // invariance: each ArgSetter can have at most one temporary Python object
  // so it ensures that we won't overflow the temporary Python object stack
  Py_IncRef(arg);
  ctx->temp_py_objects[ctx->num_temp_py_objects++] = arg;
}
#endif  // TVM_FFI_PYTHON_HELPERS_H_
