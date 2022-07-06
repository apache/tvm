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
#ifndef TVM_META_SCHEDULE_ARG_INFO_H_
#define TVM_META_SCHEDULE_ARG_INFO_H_

#include <tvm/ir/module.h>
#include <tvm/node/node.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace meta_schedule {

/*! \brief The argument information. */
class ArgInfoNode : public runtime::Object {
 public:
  static constexpr const char* _type_key = "meta_schedule.ArgInfo";
  TVM_DECLARE_BASE_OBJECT_INFO(ArgInfoNode, runtime::Object);

 public:
  /*! \brief Default destructor. */
  virtual ~ArgInfoNode() = default;
  /*! \brief Converts the ArgInfo to its corresponding JSON representation. */
  virtual ObjectRef AsJSON() const = 0;
};

/*!
 * \brief Managed reference to ArgInfoNode
 * \sa ArgInfoNode
 */
class ArgInfo : public runtime::ObjectRef {
 public:
  /*!
   * \brief Parse the argument information from a JSON object.
   * \param json_obj The json object to parse.
   * \return The argument information parsed.
   */
  TVM_DLL static ArgInfo FromJSON(const ObjectRef& json_obj);
  /*!
   * \brief Extract a list of the argument information from PrimFunc.
   * \param func The PrimFunc to get argument information from.
   * \return An array of the argument information derived.
   */
  TVM_DLL static Array<ArgInfo, void> FromPrimFunc(const tir::PrimFunc& func);
  /*!
   * \brief Extract a list of the argument information from the entry func of an IRModule
   * \param mod The IRModule to extract argument information from.
   * \param remove_preproc Whether to remove the preprocessing blocks.
   * \return An array of the argument information derived.
   */
  TVM_DLL static Array<ArgInfo, void> FromEntryFunc(const IRModule& mod, bool remove_preproc);

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ArgInfo, runtime::ObjectRef, ArgInfoNode);

 protected:
  ArgInfo() = default;
};

/*! \brief The tensor argument information. */
class TensorInfoNode : public ArgInfoNode {
 public:
  /*! \brief The data type of the tensor. */
  runtime::DataType dtype;
  /*! \brief The shape of the tensor. */
  runtime::ShapeTuple shape;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("shape", &shape);
  }

  static constexpr const char* _type_key = "meta_schedule.TensorInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorInfoNode, ArgInfoNode);

 public:
  ObjectRef AsJSON() const;
};

/*!
 * \brief Managed reference to TensorInfoNode
 * \sa TensorInfoNode
 */
class TensorInfo : public ArgInfo {
 public:
  /*!
   * \brief Constructor of TensorInfo.
   * \param dtype The data type of the tensor argument.
   * \param shape The shape tuple of the tensor argument.
   */
  TVM_DLL explicit TensorInfo(runtime::DataType dtype, runtime::ShapeTuple shape);
  /*!
   * \brief Parse the argument information from a JSON object.
   * \param json_obj The json object to parse.
   * \return The argument information parsed.
   */
  TVM_DLL static TensorInfo FromJSON(const ObjectRef& json_obj);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TensorInfo, ArgInfo, TensorInfoNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_ARG_INFO_H_
