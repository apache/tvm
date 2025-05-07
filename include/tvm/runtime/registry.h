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
 * \file tvm/runtime/registry.h
 * \brief This file defines the TVM global function registry.
 */
#ifndef TVM_RUNTIME_REGISTRY_H_
#define TVM_RUNTIME_REGISTRY_H_

#include <tvm/runtime/container/string.h>
#include <tvm/runtime/packed_func.h>

#include <type_traits>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {

/*! \brief A class that wraps a Python object and preserves its ownership.

 * This class is used to wrap a PyObject* from the Python API and preserve its ownership.
 * Allows for the creation of strong references to Python objects, which prevent them from being
 * garbage-collected as long as the wrapper object exists.
 */
class WrappedPythonObject {
 public:
  /*! \brief Construct a wrapper that doesn't own anything */
  WrappedPythonObject() : python_obj_(nullptr) {}

  /*! \brief Conversion constructor from nullptr */
  explicit WrappedPythonObject(std::nullptr_t) : python_obj_(nullptr) {}

  /*! \brief Take ownership of a python object
   *
   * A new strong reference is created for the underlying python
   * object.
   *
   * \param python_obj A PyObject* from the Python.h API.  A new
   * strong reference is created using Py_IncRef.
   */
  explicit WrappedPythonObject(void* python_obj);

  /*! \brief Drop ownership of a python object
   *
   * Removes the strong reference held by the wrapper.
   */
  ~WrappedPythonObject();

  WrappedPythonObject(WrappedPythonObject&&);
  WrappedPythonObject& operator=(WrappedPythonObject&&);

  WrappedPythonObject(const WrappedPythonObject&);
  WrappedPythonObject& operator=(const WrappedPythonObject&);
  WrappedPythonObject& operator=(std::nullptr_t);

  operator bool() { return python_obj_; }

  void* raw_pointer() { return python_obj_; }

 private:
  void* python_obj_ = nullptr;
};

/*!
 * \brief Register a function globally.
 * \code
 *   TVM_REGISTER_GLOBAL("MyPrint")
 *   .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
 *   });
 * \endcode
 */
#define TVM_REGISTER_GLOBAL TVM_FFI_REGISTER_GLOBAL

#define TVM_STRINGIZE_DETAIL(x) #x
#define TVM_STRINGIZE(x) TVM_STRINGIZE_DETAIL(x)
#define TVM_DESCRIBE(...) describe(__VA_ARGS__ "\n\nFrom:" __FILE__ ":" TVM_STRINGIZE(__LINE__))
/*!
 * \brief Macro to include current line as string
 */
#define TVM_ADD_FILELINE "\n\nDefined in " __FILE__ ":L" TVM_STRINGIZE(__LINE__)

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_REGISTRY_H_
