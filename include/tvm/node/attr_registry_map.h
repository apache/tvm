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
 * \file tvm/node/attr_registry_map.h
 * \brief Attribute map used in registry.
 */
#ifndef TVM_NODE_ATTR_REGISTRY_MAP_H_
#define TVM_NODE_ATTR_REGISTRY_MAP_H_

#include <tvm/runtime/container/string.h>

#include <utility>
#include <vector>

namespace tvm {

/*!
 * \brief Generic attribute map.
 * \tparam KeyType the type of the key.
 */
template <typename KeyType>
class AttrRegistryMapContainerMap {
 public:
  /*!
   * \brief Check if the map has key.
   * \param key The key to the map
   * \return 1 if key is contained in map, 0 otherwise.
   */
  int count(const KeyType& key) const {
    if (key.defined()) {
      const uint32_t idx = key->AttrRegistryIndex();
      return idx < data_.size() ? (data_[idx].second != 0) : 0;
    } else {
      return 0;
    }
  }
  /*!
   * \brief get the corresponding value element at key.
   * \param key The key to the map
   * \return the const reference to the content value.
   */
  const runtime::TVMRetValue& operator[](const KeyType& key) const {
    ICHECK(key.defined());
    const uint32_t idx = key->AttrRegistryIndex();
    ICHECK(idx < data_.size() && data_[idx].second != 0)
        << "Attribute " << attr_name_ << " has not been registered for " << key->name;
    return data_[idx].first;
  }
  /*!
   * \brief get the corresponding value element at key with default value.
   * \param key The key to the map
   * \param def_value The default value when the key does not exist.
   * \return the const reference to the content value.
   * \tparam ValueType The content value type.
   */
  template <typename ValueType>
  ValueType get(const KeyType& key, ValueType def_value) const {
    ICHECK(key.defined());
    const uint32_t idx = key->AttrRegistryIndex();
    if (idx < data_.size() && data_[idx].second != 0) {
      return data_[idx].first;
    } else {
      return def_value;
    }
  }

 private:
  /*! \brief The name of the attr field */
  String attr_name_;
  /*! \brief The internal data. */
  std::vector<std::pair<runtime::TVMRetValue, int>> data_;
  /*! \brief The constructor */
  AttrRegistryMapContainerMap() = default;
  template <typename, typename>
  friend class AttrRegistry;
  friend class OpRegEntry;
};

/*!
 * \brief Map<Key, ValueType> used to store meta-data.
 * \tparam KeyType The type of the key
 * \tparam ValueType The type of the value stored in map.
 */
template <typename KeyType, typename ValueType>
class AttrRegistryMap {
 public:
  /*!
   * \brief constructor
   * \param map The internal map.
   */
  explicit AttrRegistryMap(const AttrRegistryMapContainerMap<KeyType>& map) : map_(map) {}
  /*!
   * \brief Check if the map has op as key.
   * \param key The key to the map
   * \return 1 if op is contained in map, 0 otherwise.
   */
  int count(const KeyType& key) const { return map_.count(key); }
  /*!
   * \brief get the corresponding value element at key.
   * \param key The key to the map
   * \return the const reference to the content value.
   */
  ValueType operator[](const KeyType& key) const { return map_[key]; }
  /*!
   * \brief get the corresponding value element at key with default value.
   * \param key The key to the map
   * \param def_value The default value when the key does not exist.
   * \return the const reference to the content value.
   */
  ValueType get(const KeyType& key, ValueType def_value) const { return map_.get(key, def_value); }

 protected:
  /*! \brief The internal map field */
  const AttrRegistryMapContainerMap<KeyType>& map_;
};

}  // namespace tvm
#endif  // TVM_NODE_ATTR_REGISTRY_MAP_H_
