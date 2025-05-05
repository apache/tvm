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
/*
 * \file src/ffi/object.cc
 * \brief Registry to record dynamic types
 */
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace ffi {

/*!
 * \brief Type context that manages the type hierarchy information.
 *
 * \note We do not use mutex to guard updating of TypeContext
 *
 * The assumption is that updating of TypeContext will be done
 * in the main thread during initialization or loading.
 *
 * Then the followup code will leverage the information
 */
class TypeTable {
 public:
  /*! \brief Type information */
  struct Entry : public TypeInfo {
    /*! \brief stored type key */
    std::string type_key_data;
    /*! \brief acenstor information */
    std::vector<int32_t> type_acenstors_data;
    // NOTE: the indices in [index, index + num_reserved_slots) are
    // reserved for the child-class of this type.
    /*! \brief Total number of slots reserved for the type and its children. */
    int32_t num_slots;
    /*! \brief number of allocated child slots. */
    int32_t allocated_slots;
    /*! \brief Whether child can overflow. */
    bool child_slots_can_overflow{true};

    Entry(int32_t type_index, int32_t type_depth, std::string type_key, int32_t num_slots,
          bool child_slots_can_overflow, const Entry* parent) {
      // setup fields in the class
      this->type_key_data = std::move(type_key);
      this->num_slots = num_slots;
      this->allocated_slots = 1;
      this->child_slots_can_overflow = child_slots_can_overflow;
      // set up type acenstors information
      if (type_depth != 0) {
        TVM_FFI_ICHECK_NOTNULL(parent);
        TVM_FFI_ICHECK_EQ(type_depth, parent->type_depth + 1);
        type_acenstors_data.resize(type_depth);
        // copy over parent's type information
        for (int32_t i = 0; i < parent->type_depth; ++i) {
          type_acenstors_data[i] = parent->type_acenstors[i];
        }
        // set last type information to be parent
        type_acenstors_data[parent->type_depth] = parent->type_index;
      }
      // initialize type info: no change to type_key and type_acenstors fields
      // after this line
      this->type_index = type_index;
      this->type_depth = type_depth;
      this->type_key = this->type_key_data.c_str();
      this->type_key_hash = std::hash<std::string>()(this->type_key_data);
      this->type_acenstors = type_acenstors_data.data();
    }
  };

  int32_t GetOrAllocTypeIndex(std::string type_key, int32_t static_type_index, int32_t type_depth,
                              int32_t num_child_slots, bool child_slots_can_overflow,
                              int32_t parent_type_index) {
    auto it = type_key2index_.find(type_key);
    if (it != type_key2index_.end()) {
      return type_table_[it->second]->type_index;
    }

    // get parent's entry
    Entry* parent = [&]() -> Entry* {
      if (parent_type_index < 0) return nullptr;
      // try to allocate from parent's type table.
      TVM_FFI_ICHECK_LT(parent_type_index, type_table_.size())
          << " type_key=" << type_key << ", static_index=" << static_type_index;
      return type_table_[parent_type_index].get();
    }();

    // get allocated index
    int32_t allocated_tindex = [&]() {
      // Step 0: static allocation
      if (static_type_index >= 0) {
        TVM_FFI_ICHECK_LT(static_type_index, type_table_.size());
        TVM_FFI_ICHECK(type_table_[static_type_index] == nullptr)
            << "Conflicting static index " << static_type_index << " between "
            << type_table_[static_type_index]->type_key << " and " << type_key;
        return static_type_index;
      }
      TVM_FFI_ICHECK_NOTNULL(parent);
      int num_slots = num_child_slots + 1;
      if (parent->allocated_slots + num_slots <= parent->num_slots) {
        // allocate the slot from parent's reserved pool
        int32_t allocated_tindex = parent->type_index + parent->allocated_slots;
        // update parent's state
        parent->allocated_slots += num_slots;
        return allocated_tindex;
      }
      // Step 2: allocate from overflow
      TVM_FFI_ICHECK(parent->child_slots_can_overflow)
          << "Reach maximum number of sub-classes for " << parent->type_key;
      // allocate new entries.
      int32_t allocated_tindex = type_counter_;
      type_counter_ += num_slots;
      TVM_FFI_ICHECK_LE(type_table_.size(), type_counter_);
      type_table_.reserve(type_counter_);
      // resize type table
      while (static_cast<int32_t>(type_table_.size()) < type_counter_) {
        type_table_.emplace_back(nullptr);
      }
      return allocated_tindex;
    }();

    // if parent cannot overflow, then this class cannot.
    if (parent != nullptr && !(parent->child_slots_can_overflow)) {
      child_slots_can_overflow = false;
    }
    // total number of slots include the type itself.

    if (parent != nullptr) {
      TVM_FFI_ICHECK_GT(allocated_tindex, parent->type_index);
    }

    type_table_[allocated_tindex] =
        std::make_unique<Entry>(allocated_tindex, type_depth, type_key, num_child_slots + 1,
                                child_slots_can_overflow, parent);
    // update the key2index mapping.
    type_key2index_[type_key] = allocated_tindex;
    return allocated_tindex;
  }

  int32_t TypeKey2Index(const std::string& type_key) {
    auto it = type_key2index_.find(type_key);
    TVM_FFI_ICHECK(it != type_key2index_.end()) << "Cannot find type " << type_key;
    return it->second;
  }

  const TypeInfo* GetTypeInfo(int32_t type_index) {
    const TypeInfo* info = nullptr;
    if (type_index >= 0 && static_cast<size_t>(type_index) < type_table_.size()) {
      info = type_table_[type_index].get();
    }
    TVM_FFI_ICHECK(info != nullptr) << "Cannot find type info for type_index=" << type_index;
    return info;
  }

  void Dump(int min_children_count) {
    std::vector<int> num_children(type_table_.size(), 0);
    // reverse accumulation so we can get total counts in a bottom-up manner.
    for (auto it = type_table_.rbegin(); it != type_table_.rend(); ++it) {
      const Entry* ptr = it->get();
      if (ptr != nullptr && ptr->type_depth != 0) {
        int parent_index = ptr->type_acenstors[ptr->type_depth - 1];
        num_children[parent_index] += num_children[ptr->type_index] + 1;
      }
    }

    for (const auto& ptr : type_table_) {
      if (ptr != nullptr && num_children[ptr->type_index] >= min_children_count) {
        std::cerr << '[' << ptr->type_index << "]\t" << ptr->type_key;
        if (ptr->type_depth != 0) {
          int32_t parent_index = ptr->type_acenstors[ptr->type_depth - 1];
          std::cerr << "\tparent=" << type_table_[parent_index]->type_key;
        } else {
          std::cerr << "\tparent=root";
        }
        std::cerr << "\tnum_child_slots=" << ptr->num_slots - 1
                  << "\tnum_children=" << num_children[ptr->type_index] << std::endl;
      }
    }
  }

  static TypeTable* Global() {
    static TypeTable inst;
    return &inst;
  }

 private:
  TypeTable() {
    type_table_.reserve(TypeIndex::kTVMFFIDynObjectBegin);
    for (int32_t i = 0; i < TypeIndex::kTVMFFIDynObjectBegin; ++i) {
      type_table_.emplace_back(nullptr);
    }
    // initialize the entry for object
    this->GetOrAllocTypeIndex(Object::_type_key, Object::_type_index, Object::_type_depth,
                              Object::_type_child_slots, Object::_type_child_slots_can_overflow,
                              -1);
  }

  int32_t type_counter_{TypeIndex::kTVMFFIDynObjectBegin};
  std::vector<std::unique_ptr<Entry>> type_table_;
  std::unordered_map<std::string, int32_t> type_key2index_;
};

namespace details {

int32_t ObjectGetOrAllocTypeIndex(const char* type_key, int32_t static_type_index,
                                  int32_t type_depth, int32_t num_child_slots,
                                  bool child_slots_can_overflow, int32_t parent_index) {
  return tvm::ffi::TypeTable::Global()->GetOrAllocTypeIndex(type_key, static_type_index, type_depth,
                                                            num_child_slots,
                                                            child_slots_can_overflow, parent_index);
}

const TypeInfo* ObjectGetTypeInfo(int32_t type_index) {
  return tvm::ffi::TypeTable::Global()->GetTypeInfo(type_index);
}
}  // namespace details
}  // namespace ffi
}  // namespace tvm
