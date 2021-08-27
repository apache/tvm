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
#include <tvm/ir/module.h>
#include <tvm/node/node.h>
#include <tvm/support/with.h>
#include <tvm/target/target_kind.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {

class TargetInternal;
class Target;

/*!
 * \brief Compilation target.
 * \sa Target
 */
class TargetNode : public Object {
 public:
  /*! \brief The kind of the target device */
  TargetKind kind;
  /*! \brief Target host information, must be Target type */
  Optional<ObjectRef> host;
  /*! \brief Tag of the the target, can be empty */
  String tag;
  /*! \brief Keys for this target */
  Array<String> keys;
  /*! \brief Collection of attributes */
  Map<String, ObjectRef> attrs;
  /*!
   * \brief The raw string representation of the target
   * \return the full device string to pass to codegen::Build
   * \note It will be deprecated after the Target RFC is fully landed.
   */
  TVM_DLL const std::string& str() const;
  /*! \return Export target to JSON-like configuration */
  TVM_DLL Map<String, ObjectRef> Export() const;
  /*! \return The Optional<Target> typed target host of the TargetNode */
  TVM_DLL Optional<Target> GetHost() const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("kind", &kind);
    v->Visit("tag", &tag);
    v->Visit("keys", &keys);
    v->Visit("attrs", &attrs);
    v->Visit("host", &host);
  }

  /*!
   * \brief Get an entry from attrs of the target
   * \tparam TObjectRef Type of the attribute
   * \param attr_key The name of the attribute key
   * \param default_value The value returned if the key is not present
   * \return An optional, NullOpt if not found, otherwise the value found
   */
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
  /*!
   * \brief Get an entry from attrs of the target
   * \tparam TObjectRef Type of the attribute
   * \param attr_key The name of the attribute key
   * \param default_value The value returned if the key is not present
   * \return An optional, NullOpt if not found, otherwise the value found
   */
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

  friend class TargetInternal;
};

/*!
 * \brief Managed reference class to TargetNode.
 * \sa TargetNode
 */
class Target : public ObjectRef {
 public:
  /*! \brief Construct a null Target */
  TVM_DLL explicit Target(std::nullptr_t) { data_ = nullptr; }
  /*!
   * \brief Construct a Target given a string
   * \param tag_or_config_or_target_str the string to parse for target
   */
  TVM_DLL explicit Target(const String& tag_or_config_or_target_str);
  /*!
   * \brief Construct a Target using a JSON-like configuration
   * \param config The JSON-like configuration for target
   */
  TVM_DLL explicit Target(const Map<String, ObjectRef>& config);
  /*!
   * \brief Get the current target context from thread local storage.
   * \param allow_not_defined If the context stack is empty and this is set to true, an
   *   undefined Target will be returned. Otherwise, an empty context stack will cause a
   *   runtime error.
   * \return The target that is the current context. The target may not be defined if
   * allow_not_defined is true.
   */
  TVM_DLL static tvm::Target Current(bool allow_not_defined = true);
  /*!
   * \brief Construct a Target given target and host
   * \param target The Target typed object with host field undefined for target
   * \param host The Target typed object for target host
   * \return The Target with given target and host context information
   */
  TVM_DLL explicit Target(Target target, Target host);
  TVM_DEFINE_OBJECT_REF_METHODS(Target, ObjectRef, TargetNode);
  /*!
   * \brief Create a new Target object with given target (w.o host) and target host.
   * \param target The current Target typed object target, with or without host field.
   * \param host The given Target typed object target host
   * \return The new Target object with the given target and host field of given host.
   */
  static Target WithHost(const Target& target, const Target& host);

 private:
  // enable with syntax.
  friend class TargetInternal;
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
/*!
 * \brief Check and update host field of the given legacy target and target host pair.
 *  Note that this function is for legacy target api compatibility issue only, not
 *  recommended for other use.
 * \param target The pointer to a Target typed object with host field to be updated
 * \param host The pointer to a Target typed object for target host to be updated
 */
void CheckAndUpdateHostConsistency(Target* target, Target* host);
/*!
 * \brief Check and update host field of the given legacy heterogeneous targets and
 *  target host.Note that this function is for legacy target api compatibility issue only,
 *  not recommended for other use.
 * \param target The pointer to a Map objects with values being Target objects
 * \param host The Target typed object for target host to be updated
 */
void CheckAndUpdateHostConsistency(Map<Integer, Target>* target, Target* host);
/*!
 * \brief Check and update host field of the given legacy heterogeneous targets and
 *  target host.Note that this function is for legacy target api compatibility issue only,
 *  not recommended for other use.
 * \param target The pointer to a Map objects with keys being Target objects
 * \param host The Target typed object for target host to be updated
 */
void CheckAndUpdateHostConsistency(Map<Target, IRModule>* target, Target* host);

// TODO(@electriclilies): Move to somewhere in backend and add note about appropriate use

/*! \brief Target hash function */
struct TargetStrHash {
  /*!
   * \brief Calculate the hash code of a Target based on the string value of the Target
   This will be removed when maps from Targets to IRModules are removed from the codebase.
   * \param a The given Target
   * \return String hash of the target
   */
  size_t operator()(const Target& target) const {
    return String::HashBytes(target->str().c_str(), target->str().size());
  }
};

/*! \brief Target equality function based on the string value of Target
This will be removed when maps from Targets to IRModules are removed from the
codebase.*/
struct TargetStrEqual {
  /*!
   * \brief Check if the two Targets are equal
   * \param a One Target
   * \param b The other Target
   * \return String equality of the targets
   */
  const bool operator()(const Target& a, const Target& b) const {
    TargetStrHash target_hash = TargetStrHash();
    return target_hash(a) == target_hash(b);
  }
};

/*!
 * \brief Convert a Map<Target, IRModule> to std::unordered_map<Target, IRmodule, TargetStrHash,
 * TargetStrEqual> Target equality is currently based on pointer equality, which is a problem since
 * we have a lot of Map<Target, IRModule> in the codebase. This function converts the map to a
 * version that is keyed based on string value of the Target instead. Note that once we remove
 * Map<Target, IRModule>, this function will be removed.
 * \param input_map The map to convert
 * \return The converted map
 */
std::unordered_map<Target, IRModule, TargetStrHash, TargetStrEqual>
TargetModuleMapToTargetStrModuleMap(Map<Target, IRModule> input_map);

/*!
 * \brief Convert a std::unordered_map<Target, IRmodule, TargetStrHash, TargetStrEqual> to
 * Map<Target, IRModule> This function is a helper that undoes TargetModuleMapToTargetStr. Note that
 * once we remove Map<Target, IRModule>, this function will be removed.
 * \param input_map The map to convert
 * \return The converted map
 */
Map<Target, IRModule> TargetStrModuleMapToTargetModuleMap(
    std::unordered_map<Target, IRModule, TargetStrHash, TargetStrEqual> input_map);

}  // namespace tvm
#endif  // TVM_TARGET_TARGET_H_
