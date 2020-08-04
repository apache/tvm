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
 * \file tvm/target/target.h
 * \brief Compilation target object.
 */
#ifndef TVM_TARGET_TARGET_H_
#define TVM_TARGET_TARGET_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/transform.h>
#include <tvm/node/container.h>
#include <tvm/support/with.h>
#include <tvm/target/target_id.h>

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
/*!
 * \brief Compilation target.
 * \note Use target::llvm, target::cuda etc functions.
 * \sa Target
 */
class TargetNode : public Object {
 public:
  /*! \brief The id of the target device */
  TargetId id;
  /*! \brief Tag of the the target, can be empty */
  String tag;
  /*! \brief Keys for this target */
  Array<String> keys;
  /*! \brief Collection of attributes */
  Map<String, ObjectRef> attrs;

  /*! \return the full device string to pass to codegen::Build */
  TVM_DLL const std::string& str() const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("id", &id);
    v->Visit("tag", &tag);
    v->Visit("keys", &keys);
    v->Visit("attrs", &attrs);
  }

  template <typename TObjectRef>
  Optional<TObjectRef> GetAttr(
      const std::string& attr_key,
      Optional<TObjectRef> default_value = Optional<TObjectRef>(nullptr)) const {
    static_assert(std::is_base_of<ObjectRef, TObjectRef>::value,
                  "Can only call GetAttr with ObjectRef types.");
    auto it = attrs.find(attr_key);
    if (it != attrs.end()) {
      return Downcast<Optional<TObjectRef>>((*it).second);
    } else {
      return default_value;
    }
  }

  template <typename TObjectRef>
  Optional<TObjectRef> GetAttr(const std::string& attr_key, TObjectRef default_value) const {
    return GetAttr<TObjectRef>(attr_key, Optional<TObjectRef>(default_value));
  }

  /*! \brief Get the keys for this target as a vector of string */
  TVM_DLL std::vector<std::string> GetKeys() const;

  /*! \brief Get the keys for this target as an unordered_set of string */
  TVM_DLL std::unordered_set<std::string> GetLibs() const;

  static constexpr const char* _type_key = "Target";
  TVM_DECLARE_FINAL_OBJECT_INFO(TargetNode, Object);

 private:
  /*! \brief Internal string repr. */
  mutable std::string str_repr_;
  friend class Target;
};

/*!
 * \brief Managed reference class to TargetNode.
 * \sa TargetNode
 */
class Target : public ObjectRef {
 public:
  Target() {}
  explicit Target(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief Create a Target given a string
   * \param target_str the string to parse
   */
  TVM_DLL static Target Create(const String& target_str);
  /*!
   * \brief Construct a Target node from the given name and options.
   * \param name The major target name. Should be one of
   * {"aocl", "aocl_sw_emu", "c", "cuda", "ext_dev", "hexagon", "hybrid", "llvm",
   *  "metal", "nvptx", "opencl", "rocm", "sdaccel", "stackvm", "vulkan"}
   * \param options Additional options appended to the target
   * \return The constructed Target
   */
  TVM_DLL static Target CreateTarget(const std::string& name,
                                     const std::vector<std::string>& options);
  /*!
   * \brief Get the current target context from thread local storage.
   * \param allow_not_defined If the context stack is empty and this is set to true, an
   *   undefined Target will be returned. Otherwise, an empty context stack will cause a
   *   runtime error.
   * \return The target that is the current context. The target may not be defined if
   * allow_not_defined is true.
   */
  TVM_DLL static tvm::Target Current(bool allow_not_defined = true);

  const TargetNode* operator->() const { return static_cast<const TargetNode*>(get()); }

  using ContainerType = TargetNode;
  class Internal;

 private:
  // enable with syntax.
  friend class Internal;
  friend class With<Target>;
  /*!
   * \brief Push a new target context onto the thread local stack.
   *  The Target on top of the stack is used to determine which
   *  specialization to use when invoking a GenericFunc.
   */
  TVM_DLL void EnterWithScope();
  /*!
   * \brief Pop a target off the thread local context stack,
   *  restoring the previous target as the current context.
   */
  TVM_DLL void ExitWithScope();
};

/*! \brief This namespace provides functions to construct Target instances */
namespace target {

/*! \return A target for LLVM */
TVM_DLL Target llvm(const std::vector<std::string>& options = std::vector<std::string>());

/*! \return A target for CUDA */
TVM_DLL Target cuda(const std::vector<std::string>& options = std::vector<std::string>());

/*! \return A target for ROCm */
TVM_DLL Target rocm(const std::vector<std::string>& options = std::vector<std::string>());

/*! \return A target for OpenCL */
TVM_DLL Target opencl(const std::vector<std::string>& options = std::vector<std::string>());

/*! \return A target for Metal */
TVM_DLL Target metal(const std::vector<std::string>& options = std::vector<std::string>());

/*! \return A target for rasp */
TVM_DLL Target rasp(const std::vector<std::string>& options = std::vector<std::string>());

/*! \return A target for Mali */
TVM_DLL Target mali(const std::vector<std::string>& options = std::vector<std::string>());

/*! \return A target for Intel Graphics */
TVM_DLL Target intel_graphics(const std::vector<std::string>& options = std::vector<std::string>());

/*! \return A target for stackvm */
TVM_DLL Target stackvm(const std::vector<std::string>& options = std::vector<std::string>());

/*! \return A target for external device */
TVM_DLL Target ext_dev(const std::vector<std::string>& options = std::vector<std::string>());

/*! \return A target for hexagon */
TVM_DLL Target hexagon(const std::vector<std::string>& options = std::vector<std::string>());
}  // namespace target

}  // namespace tvm
#endif  // TVM_TARGET_TARGET_H_
