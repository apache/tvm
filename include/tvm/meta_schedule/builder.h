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
#ifndef TVM_META_SCHEDULE_BUILDER_H_
#define TVM_META_SCHEDULE_BUILDER_H_

#include <tvm/ir/module.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/target/target.h>

namespace tvm {
namespace meta_schedule {

/*! \brief The builder's input, containing an IRModule and the target. */
class BuilderInputNode : public runtime::Object {
 public:
  /*! \brief The IRModule to be built. */
  IRModule mod;
  /*! \brief The target to be built for. */
  Target target;
  /*! \brief Parameters for Relay build module. */
  Optional<Map<String, runtime::NDArray>> params;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("mod", &mod);
    v->Visit("target", &target);
    v->Visit("params", &params);
  }

  static constexpr const char* _type_key = "meta_schedule.BuilderInput";
  TVM_DECLARE_FINAL_OBJECT_INFO(BuilderInputNode, runtime::Object);
};

/*!
 * \brief Managed reference to BuilderInputNode
 * \sa BuilderInputNode
 */
class BuilderInput : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor of BuilderInput.
   * \param mod The IRModule to be built.
   * \param target The target to be built for.
   * \param params Parameters for Relay build module.
   */
  TVM_DLL explicit BuilderInput(IRModule mod, Target target,
                                Optional<Map<String, runtime::NDArray>> params = NullOpt);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BuilderInput, runtime::ObjectRef, BuilderInputNode);
};

/*! \brief The builder's output, containing the artifact path or error message if any. */
class BuilderResultNode : public runtime::Object {
 public:
  /*! \brief The path to the built artifact. */
  Optional<String> artifact_path;
  /*! \brief The error message if any. */
  Optional<String> error_msg;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("artifact_path", &artifact_path);
    v->Visit("error_msg", &error_msg);
  }

  static constexpr const char* _type_key = "meta_schedule.BuilderResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(BuilderResultNode, runtime::Object);
};

/*!
 * \brief Managed reference to BuilderResultNode
 * \sa BuilderResultNode
 */
class BuilderResult : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor of BuilderResult.
   * \param artifact_path The path to the built artifact.
   * \param error_msg The error message if any.
   */
  TVM_DLL explicit BuilderResult(Optional<String> artifact_path, Optional<String> error_msg);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BuilderResult, runtime::ObjectRef, BuilderResultNode);
};

/*! \brief The abstract builder interface. */
class BuilderNode : public runtime::Object {
 public:
  /*! \brief Default destructor */
  virtual ~BuilderNode() = default;
  /*!
   * \brief Generate the build results from build inputs.
   * \param build_inputs The inputs to be built.
   * \return The build results.
   */
  virtual Array<BuilderResult> Build(const Array<BuilderInput>& build_inputs) = 0;
  /*!
   * \brief The function type of `Build` method.
   * \param build_inputs The inputs to be built.
   * \return The build results.
   */
  using FBuild = runtime::TypedPackedFunc<Array<BuilderResult>(const Array<BuilderInput>&)>;

  static constexpr const char* _type_key = "meta_schedule.Builder";
  TVM_DECLARE_BASE_OBJECT_INFO(BuilderNode, runtime::Object);
};

/*!
 * \brief Managed reference to BuilderNode
 * \sa BuilderNode
 */
class Builder : public runtime::ObjectRef {
 public:
  /*!
   * \brief Create a builder with customized build method on the python-side.
   * \param f_build The packed function to the `Build` function..
   * \return The Builder created.
   */
  static Builder PyBuilder(BuilderNode::FBuild f_build);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Builder, runtime::ObjectRef, BuilderNode);
};

/*! \brief An abstract builder with customized build method on the python-side. */
class PyBuilderNode : public BuilderNode {
 public:
  /*! \brief The packed function to the `Build` function. */
  FBuild f_build;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_build` is not visited
  }

  Array<BuilderResult> Build(const Array<BuilderInput>& build_inputs) final {
    ICHECK(f_build != nullptr) << "PyBuilder's Build method not implemented!";
    return f_build(build_inputs);
  }

  static constexpr const char* _type_key = "meta_schedule.PyBuilder";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyBuilderNode, BuilderNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_BUILDER_H_
