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
 * \file tvm/target/tag.h
 * \brief Target tag registry
 */
#ifndef TVM_TARGET_TAG_H_
#define TVM_TARGET_TAG_H_

#include <tvm/node/attr_registry_map.h>
#include <tvm/node/node.h>
#include <tvm/target/target.h>

#include <utility>

namespace tvm {

/*! \brief A target tag */
class TargetTagNode : public Object {
 public:
  /*! \brief Name of the target */
  String name;
  /*! \brief Config map to generate the target */
  Map<String, ObjectRef> config;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("config", &config);
  }

  static constexpr const char* _type_key = "TargetTag";
  TVM_DECLARE_FINAL_OBJECT_INFO(TargetTagNode, Object);

 private:
  /*! \brief Return the index stored in attr registry */
  uint32_t AttrRegistryIndex() const { return index_; }
  /*! \brief Return the name stored in attr registry */
  String AttrRegistryName() const { return name; }
  /*! \brief Index used for internal lookup of attribute registry */
  uint32_t index_;

  template <typename, typename>
  friend class AttrRegistry;
  template <typename>
  friend class AttrRegistryMapContainerMap;
  friend class TargetTagRegEntry;
};

/*!
 * \brief Managed reference class to TargetTagNode
 * \sa TargetTagNode
 */
class TargetTag : public ObjectRef {
 public:
  /*!
   * \brief Retrieve the Target given it the name of target tag
   * \param target_tag_name Name of the target tag
   * \return The Target requested
   */
  TVM_DLL static Optional<Target> Get(const String& target_tag_name);
  /*!
   * \brief List all names of the existing target tags
   * \return A dictionary that maps tag name to the concrete target it corresponds to
   */
  TVM_DLL static Map<String, Target> ListTags();
  /*!
   * \brief Add a tag into the registry
   * \param name Name of the tag
   * \param config The target config corresponding to the tag
   * \param override Allow overriding existing tags
   * \return Target created with the tag
   */
  TVM_DLL static Target AddTag(String name, Map<String, ObjectRef> config, bool override);

  TVM_DEFINE_OBJECT_REF_METHODS(TargetTag, ObjectRef, TargetTagNode);

 private:
  /*! \brief Mutable access to the container class  */
  TargetTagNode* operator->() { return static_cast<TargetTagNode*>(data_.get()); }
  friend class TargetTagRegEntry;
};

class TargetTagRegEntry {
 public:
  /*!
   * \brief Set the config dict corresponding to the target tag
   * \param config The config dict for target creation
   */
  inline TargetTagRegEntry& set_config(Map<String, ObjectRef> config);
  /*!
   * \brief Add a key-value pair to the config dict
   * \param key The attribute name
   * \param value The attribute value
   */
  inline TargetTagRegEntry& with_config(String key, ObjectRef value);
  /*! \brief Set name of the TargetTag to be the same as registry if it is empty */
  inline TargetTagRegEntry& set_name();
  /*!
   * \brief Register or get a new entry.
   * \param target_tag_name The name of the TargetTag.
   * \return the corresponding entry.
   */
  TVM_DLL static TargetTagRegEntry& RegisterOrGet(const String& target_tag_name);

 private:
  TargetTag tag_;
  String name;

  /*! \brief private constructor */
  explicit TargetTagRegEntry(uint32_t reg_index) : tag_(make_object<TargetTagNode>()) {
    tag_->index_ = reg_index;
  }
  template <typename, typename>
  friend class AttrRegistry;
  friend class TargetTag;
};

inline TargetTagRegEntry& TargetTagRegEntry::set_config(Map<String, ObjectRef> config) {
  tag_->config = std::move(config);
  return *this;
}

inline TargetTagRegEntry& TargetTagRegEntry::with_config(String key, ObjectRef value) {
  tag_->config.Set(key, value);
  return *this;
}

inline TargetTagRegEntry& TargetTagRegEntry::set_name() {
  if (tag_->name.empty()) {
    tag_->name = name;
  }
  return *this;
}

#define TVM_TARGET_TAG_REGISTER_VAR_DEF \
  static DMLC_ATTRIBUTE_UNUSED ::tvm::TargetTagRegEntry& __make_##TargetTag

/*!
 * \def TVM_REGISTER_TARGET_TAG
 * \brief Register a new target tag, or set attribute of the corresponding target tag.
 * \param TargetTagName The name of target tag
 */
#define TVM_REGISTER_TARGET_TAG(TargetTagName)                   \
  TVM_STR_CONCAT(TVM_TARGET_TAG_REGISTER_VAR_DEF, __COUNTER__) = \
      ::tvm::TargetTagRegEntry::RegisterOrGet(TargetTagName).set_name()

}  // namespace tvm

#endif  // TVM_TARGET_TAG_H_
