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
 * \file tvm/base.h
 * \brief Base utilities
 */
#ifndef TVM_BASE_H_
#define TVM_BASE_H_

#include <dmlc/logging.h>
#include <utility>

namespace tvm {

/*!
 * \brief RAII wrapper function to enter and exit a context object
 *        similar to python's with syntax.
 *
 * \code
 * // context class
 * class MyContext {
 *  private:
 *    friend class With<MyContext>;
      MyContext(arguments);
 *    void EnterWithScope();
 *    void ExitWithScope();
 * };
 *
 * {
 *   With<MyContext> scope(arguments);
 *   // effect take place.
 * }
 * \endcode
 *
 * \tparam ContextType Type of the context object.
 */
template<typename ContextType>
class With {
 public:
  /*!
   * \brief constructor.
   *  Enter the scope of the context.
   */
  template<typename ...Args>
  explicit With(Args&& ...args)
      : ctx_(std::forward<Args>(args)...) {
    ctx_.EnterWithScope();
  }
  /*! \brief destructor, leaves the scope of the context. */
  ~With() DMLC_THROW_EXCEPTION {
    ctx_.ExitWithScope();
  }

 private:
  /*! \brief internal context type. */
  ContextType ctx_;
};

#define TVM_STRINGIZE_DETAIL(x) #x
#define TVM_STRINGIZE(x) TVM_STRINGIZE_DETAIL(x)
#define TVM_DESCRIBE(...) describe(__VA_ARGS__ "\n\nFrom:" __FILE__ ":" TVM_STRINGIZE(__LINE__))
/*!
 * \brief Macro to include current line as string
 */
#define TVM_ADD_FILELINE "\n\nDefined in " __FILE__ ":L" TVM_STRINGIZE(__LINE__)


}  // namespace tvm
#endif  // TVM_BASE_H_
