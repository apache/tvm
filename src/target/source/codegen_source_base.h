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
 * \file codegen_source_base.h
 * \brief Common utilities to source code in text form.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_SOURCE_BASE_H_
#define TVM_TARGET_SOURCE_CODEGEN_SOURCE_BASE_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../runtime/meta_data.h"

namespace tvm {
namespace codegen {

/*!
 * \brief A base class to generate source code.
 * Contains helper utilities to generate nest and ssa form.
 */
class CodeGenSourceBase {
 public:
  virtual ~CodeGenSourceBase() = default;
  /*!
   * \brief Register constant value appeared in expresion tree
   *  This avoid generated a ssa id for each appearance of the value
   * \param value The constant value.
   */
  void MarkConst(std::string value);

 protected:
  /*! \brief entry in ssa assign map */
  struct SSAEntry {
    /*! \brief The value id */
    std::string vid;
    /*! \brief The scope id, used to check if this entry is invalid. */
    int scope_id;
  };
  /*! \brief Clear the states that might relates to function generation */
  void ClearFuncState();
  /*! \brief print the current indented value */
  void PrintIndent();
  /*!
   * \brief Allocate a variable name for a newly defined var.
   * \param v The variable.
   * \return the variable name.
   */
  std::string AllocVarID(const tir::VarNode* v);
  /*!
   * \brief Get a variable name.
   * \param v The variable.
   * \return the variable name.
   */
  std::string GetVarID(const tir::VarNode* v) const;
  /*!
   * \brief Get the SSA ID corresponds to src
   *  If necessary, generate new assignment
   * \param src The source expression
   * \param t The type of the expression.
   */
  std::string SSAGetID(std::string src, DataType t);
  /*!
   * \brief get a unique name with the corresponding prefix
   * \param prefix The prefix of the name
   * \return The returned name.
   */
  std::string GetUniqueName(std::string prefix);
  /*!
   * \brief mark the beginning of a new scope
   * \return The scope id.
   */
  int BeginScope();
  /*!
   * \brief mark the end of an old scope.
   * \param scope_id The scope id to be ended.
   */
  void EndScope(int scope_id);
  /*!
   * \brief Print assignment of src to the id in ssa entry.
   * \param target id of target variable.
   * \param src The source expression.
   * \param t The type of target.
   */
  virtual void PrintSSAAssign(const std::string& target, const std::string& src, DataType t) = 0;

  /*! \brief the declaration stream */
  std::ostringstream decl_stream;
  /*! \brief the stream to be printed */
  std::ostringstream stream;
  /*! \brief name of each variable */
  std::unordered_map<const tir::VarNode*, std::string> var_idmap_;

 private:
  /*! \brief assignment map of ssa */
  std::unordered_map<std::string, SSAEntry> ssa_assign_map_;
  /*! \brief name allocation map */
  std::unordered_map<std::string, int> name_alloc_map_;
  /*! \brief array to check whether we are inside certain scope */
  std::vector<bool> scope_mark_;
  /*! \brief The current indentation value */
  int indent_{0};
};

/*!
 * \brief Create a source module for viewing.
 * \param code The code to be viewed.
 * \param fmt The code. format.
 */
runtime::Module SourceModuleCreate(std::string code, std::string fmt);

/*!
 * \brief Create a C source module for viewing and compiling GCC code.
 * \param code The code to be viewed.
 * \param fmt The code format.
 * \param symbol The symbol that the c source module represents.
 * \param const_vars. The constant variables that the c source module needs.
 * \return The created module.
 */
runtime::Module CSourceModuleCreate(const String& code, const String& fmt,
                                    const String& symbol = "",
                                    const Array<String>& const_vars = {});

/*!
 * \brief Wrap the submodules in a metadata module.
 * \param params The variable to constant mapping that is collected by the host
 *        module.
 * \param dso_module The host module to be wrapped.
 * \param modules The modules to be wrapped.
 * \return The wrapped module.
 */
runtime::Module CreateMetadataModule(
    const std::unordered_map<std::string, runtime::NDArray>& params,
    const runtime::Module& dso_module, const Array<runtime::Module>& modules);

/*!
 * \brief Create a source module for viewing and limited saving for device.
 * \param data The code data to be viewed.
 * \param fmt The code. format.
 * \param fmap The map function information map of each function.
 * \param type_key The type_key of the runtime module of this source code
 * \param fget_source a closure to replace default get source behavior.
 */
runtime::Module DeviceSourceModuleCreate(
    std::string data, std::string fmt, std::unordered_map<std::string, runtime::FunctionInfo> fmap,
    std::string type_key, std::function<std::string(const std::string&)> fget_source = nullptr);
}  // namespace codegen
}  // namespace tvm
#endif  // TVM_TARGET_SOURCE_CODEGEN_SOURCE_BASE_H_
