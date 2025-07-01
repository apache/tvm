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
#ifndef TVM_META_SCHEDULE_EXTRACTED_TASK_H_
#define TVM_META_SCHEDULE_EXTRACTED_TASK_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/reflection.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/module.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/object.h>
#include <tvm/target/target.h>

namespace tvm {
namespace tir {
class PrimFunc;
}  // namespace tir
namespace te {
class Tensor;
}  // namespace te
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

/*! \brief A tuning task extracted from the high-level IR */
class ExtractedTaskNode : public runtime::Object {
 public:
  /*! \brief The name of the task extracted */
  String task_name;
  /*! \brief The high-level IR */
  IRModule mod;
  /*! \brief Target */
  Target target;
  /*! \brief A list of low-level IRs that the high-level IR could potentially dispatch to */
  Array<IRModule> dispatched;
  /*! \brief Weight of the task */
  int weight;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExtractedTaskNode>()
        .def_ro("task_name", &ExtractedTaskNode::task_name)
        .def_ro("mod", &ExtractedTaskNode::mod)
        .def_ro("target", &ExtractedTaskNode::target)
        .def_ro("dispatched", &ExtractedTaskNode::dispatched)
        .def_ro("weight", &ExtractedTaskNode::weight);
  }

  static constexpr const char* _type_key = "meta_schedule.ExtractedTask";

  TVM_DECLARE_FINAL_OBJECT_INFO(ExtractedTaskNode, runtime::Object);
};

/*!
 * \brief Managed reference to ExtractedTaskNode
 * \sa ExtractedTaskNode
 */
class ExtractedTask : public runtime::ObjectRef {
 public:
  explicit ExtractedTask(String task_name, IRModule mod, Target target, Array<IRModule> dispatched,
                         int weight);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ExtractedTask, runtime::ObjectRef,
                                                    ExtractedTaskNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_EXTRACTED_TASK_H_
