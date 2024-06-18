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
 * \file tvm/node/attr_registry.h
 * \brief Common global registry for objects that also have additional attrs.
 */
#ifndef TVM_NODE_ATTR_REGISTRY_H_
#define TVM_NODE_ATTR_REGISTRY_H_

#include <tvm/node/attr_registry_map.h>
#include <tvm/runtime/packed_func.h>

#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {

/*!
 * \brief Implementation of registry with attributes.
 *
 * \tparam EntryType The type of the registry entry.
 * \tparam KeyType The actual key that is used to lookup the attributes.
 *                 each entry has a corresponding key by default.
 */
template <typename EntryType, typename KeyType>
class AttrRegistry {
 public:
  using TSelf = AttrRegistry<EntryType, KeyType>;
  /*!
   * \brief Get an entry from the registry.
   * \param name The name of the item.
   * \return The corresponding entry.
   */
  const EntryType* Get(const String& name) const {
    auto it = entry_map_.find(name);
    if (it != entry_map_.end()) return it->second;
    return nullptr;
  }

  /*!
   * \brief Get an entry or register a new one.
   * \param name The name of the item.
   * \return The corresponding entry.
   */
  EntryType& RegisterOrGet(const String& name) {
    auto it = entry_map_.find(name);
    if (it != entry_map_.end()) return *it->second;
    uint32_t registry_index = static_cast<uint32_t>(entries_.size());
    auto entry = std::unique_ptr<EntryType>(new EntryType(registry_index));
    auto* eptr = entry.get();
    eptr->name = name;
    entry_map_[name] = eptr;
    entries_.emplace_back(std::move(entry));
    return *eptr;
  }

  /*!
   * \brief List all the entry names in the registry.
   * \return The entry names.
   */
  Array<String> ListAllNames() const {
    Array<String> names;
    for (const auto& kv : entry_map_) {
      names.push_back(kv.first);
    }
    return names;
  }

  /*!
   * \brief Update the attribute stable.
   * \param attr_name The name of the attribute.
   * \param key The key to the attribute table.
   * \param value The value to be set.
   * \param plevel The support level.
   */
  void UpdateAttr(const String& attr_name, const KeyType& key, runtime::TVMRetValue value,
                  int plevel) {
    using runtime::TVMRetValue;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& op_map = attrs_[attr_name];
    if (op_map == nullptr) {
      op_map.reset(new AttrRegistryMapContainerMap<KeyType>());
      op_map->attr_name_ = attr_name;
    }

    uint32_t index = key->AttrRegistryIndex();
    if (op_map->data_.size() <= index) {
      op_map->data_.resize(index + 1, std::make_pair(TVMRetValue(), 0));
    }
    std::pair<TVMRetValue, int>& p = op_map->data_[index];
    ICHECK(p.second != plevel) << "Attribute " << attr_name << " of " << key->AttrRegistryName()
                               << " is already registered with same plevel=" << plevel;
    ICHECK(value.type_code() != kTVMNullptr) << "Registered packed_func is Null for " << attr_name
                                             << " of operator " << key->AttrRegistryName();
    if (p.second < plevel && value.type_code() != kTVMNullptr) {
      op_map->data_[index] = std::make_pair(value, plevel);
    }
  }

  /*!
   * \brief Reset an attribute table entry.
   * \param attr_name The name of the attribute.
   * \param key The key to the attribute table.
   */
  void ResetAttr(const String& attr_name, const KeyType& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& op_map = attrs_[attr_name];
    if (op_map == nullptr) {
      return;
    }
    uint32_t index = key->AttrRegistryIndex();
    if (op_map->data_.size() > index) {
      op_map->data_[index] = std::make_pair(TVMRetValue(), 0);
    }
  }

  /*!
   * \brief Get an internal attribute map.
   * \param attr_name The name of the attribute.
   * \return The result attribute map.
   */
  const AttrRegistryMapContainerMap<KeyType>& GetAttrMap(const String& attr_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = attrs_.find(attr_name);
    if (it == attrs_.end()) {
      LOG(FATAL) << "Attribute \'" << attr_name << "\' is not registered";
    }
    return *it->second.get();
  }

  /*!
   * \brief Check of attribute has been registered.
   * \param attr_name The name of the attribute.
   * \return The check result.
   */
  bool HasAttrMap(const String& attr_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return attrs_.count(attr_name);
  }

  /*!
   * \return a global singleton of the registry.
   */
  static TSelf* Global() {
    static TSelf* inst = new TSelf();
    return inst;
  }

 private:
  // mutex to avoid registration from multiple threads.
  std::mutex mutex_;
  // entries in the registry
  std::vector<std::unique_ptr<EntryType>> entries_;
  // map from name to entries.
  std::unordered_map<String, EntryType*> entry_map_;
  // storage of additional attribute table.
  std::unordered_map<String, std::unique_ptr<AttrRegistryMapContainerMap<KeyType>>> attrs_;
};

}  // namespace tvm
#endif  // TVM_NODE_ATTR_REGISTRY_H_
