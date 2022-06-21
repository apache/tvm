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
 * \file src/runtime/object.cc
 * \brief Object type management system.
 */
#include <tvm/runtime/logging.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "object_internal.h"
#include "runtime_base.h"

namespace tvm {
namespace runtime {

/*! \brief Type information */
struct TypeInfo {
  /*! \brief The current index. */
  uint32_t index{0};
  /*! \brief Index of the parent in the type hierarchy */
  uint32_t parent_index{0};
  // NOTE: the indices in [index, index + num_reserved_slots) are
  // reserved for the child-class of this type.
  /*! \brief Total number of slots reserved for the type and its children. */
  uint32_t num_slots{0};
  /*! \brief number of allocated child slots. */
  uint32_t allocated_slots{0};
  /*! \brief Whether child can overflow. */
  bool child_slots_can_overflow{true};
  /*! \brief name of the type. */
  std::string name;
  /*! \brief hash of the name */
  size_t name_hash{0};
};

/*!
 * \brief Type context that manages the type hierarchy information.
 */
class TypeContext {
 public:
  // NOTE: this is a relatively slow path for child checking
  // Most types are already checked by the fast-path via reserved slot checking.
  bool DerivedFrom(uint32_t child_tindex, uint32_t parent_tindex) {
    // invariance: child's type index is always bigger than its parent.
    if (child_tindex < parent_tindex) return false;
    if (child_tindex == parent_tindex) return true;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ICHECK_LT(child_tindex, type_table_.size());
      while (child_tindex > parent_tindex) {
        child_tindex = type_table_[child_tindex].parent_index;
      }
    }
    return child_tindex == parent_tindex;
  }

  uint32_t GetOrAllocRuntimeTypeIndex(const std::string& skey, uint32_t static_tindex,
                                      uint32_t parent_tindex, uint32_t num_child_slots,
                                      bool child_slots_can_overflow) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = type_key2index_.find(skey);
    if (it != type_key2index_.end()) {
      return it->second;
    }
    // try to allocate from parent's type table.
    ICHECK_LT(parent_tindex, type_table_.size())
        << " skey=" << skey << ", static_index=" << static_tindex;
    TypeInfo& pinfo = type_table_[parent_tindex];
    ICHECK_EQ(pinfo.index, parent_tindex);

    // if parent cannot overflow, then this class cannot.
    if (!pinfo.child_slots_can_overflow) {
      child_slots_can_overflow = false;
    }

    // total number of slots include the type itself.
    uint32_t num_slots = num_child_slots + 1;
    uint32_t allocated_tindex;

    if (static_tindex != TypeIndex::kDynamic) {
      // statically assigned type
      VLOG(3) << "TypeIndex[" << static_tindex << "]: static: " << skey << ", parent "
              << type_table_[parent_tindex].name;
      allocated_tindex = static_tindex;
      ICHECK_LT(static_tindex, type_table_.size());
      ICHECK_EQ(type_table_[allocated_tindex].allocated_slots, 0U)
          << "Conflicting static index " << static_tindex << " between "
          << type_table_[allocated_tindex].name << " and " << skey;
    } else if (pinfo.allocated_slots + num_slots <= pinfo.num_slots) {
      // allocate the slot from parent's reserved pool
      allocated_tindex = parent_tindex + pinfo.allocated_slots;
      VLOG(3) << "TypeIndex[" << allocated_tindex << "]: dynamic: " << skey << ", parent "
              << type_table_[parent_tindex].name;
      // update parent's state
      pinfo.allocated_slots += num_slots;
    } else {
      VLOG(3) << "TypeIndex[" << type_counter_ << "]: dynamic (overflow): " << skey << ", parent "
              << type_table_[parent_tindex].name;
      ICHECK(pinfo.child_slots_can_overflow)
          << "Reach maximum number of sub-classes for " << pinfo.name;
      // allocate new entries.
      allocated_tindex = type_counter_;
      type_counter_ += num_slots;
      ICHECK_LE(type_table_.size(), type_counter_);
      type_table_.resize(type_counter_, TypeInfo());
    }
    ICHECK_GT(allocated_tindex, parent_tindex);
    // initialize the slot.
    type_table_[allocated_tindex].index = allocated_tindex;
    type_table_[allocated_tindex].parent_index = parent_tindex;
    type_table_[allocated_tindex].num_slots = num_slots;
    type_table_[allocated_tindex].allocated_slots = 1;
    type_table_[allocated_tindex].child_slots_can_overflow = child_slots_can_overflow;
    type_table_[allocated_tindex].name = skey;
    type_table_[allocated_tindex].name_hash = std::hash<std::string>()(skey);
    // update the key2index mapping.
    type_key2index_[skey] = allocated_tindex;
    return allocated_tindex;
  }

  std::string TypeIndex2Key(uint32_t tindex) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tindex != 0) {
      // always return the right type key for root
      // for non-root type nodes, allocated slots should not equal 0
      ICHECK(tindex < type_table_.size() && type_table_[tindex].allocated_slots != 0)
          << "Unknown type index " << tindex;
    }
    return type_table_[tindex].name;
  }

  size_t TypeIndex2KeyHash(uint32_t tindex) {
    std::lock_guard<std::mutex> lock(mutex_);
    ICHECK(tindex < type_table_.size() && type_table_[tindex].allocated_slots != 0)
        << "Unknown type index " << tindex;
    return type_table_[tindex].name_hash;
  }

  uint32_t TypeKey2Index(const std::string& skey) {
    auto it = type_key2index_.find(skey);
    ICHECK(it != type_key2index_.end())
        << "Cannot find type " << skey
        << ". Did you forget to register the node by TVM_REGISTER_NODE_TYPE ?";
    return it->second;
  }

  void Dump(int min_children_count) {
    std::vector<int> num_children(type_table_.size(), 0);
    // reverse accumulation so we can get total counts in a bottom-up manner.
    for (auto it = type_table_.rbegin(); it != type_table_.rend(); ++it) {
      if (it->index != 0) {
        num_children[it->parent_index] += num_children[it->index] + 1;
      }
    }

    for (const auto& info : type_table_) {
      if (info.index != 0 && num_children[info.index] >= min_children_count) {
        std::cerr << '[' << info.index << "] " << info.name
                  << "\tparent=" << type_table_[info.parent_index].name
                  << "\tnum_child_slots=" << info.num_slots - 1
                  << "\tnum_children=" << num_children[info.index] << std::endl;
      }
    }
  }

  static TypeContext* Global() {
    static TypeContext inst;
    return &inst;
  }

 private:
  TypeContext() {
    type_table_.resize(TypeIndex::kStaticIndexEnd, TypeInfo());
    type_table_[0].name = "runtime.Object";
  }
  // mutex to avoid registration from multiple threads.
  std::mutex mutex_;
  std::atomic<uint32_t> type_counter_{TypeIndex::kStaticIndexEnd};
  std::vector<TypeInfo> type_table_;
  std::unordered_map<std::string, uint32_t> type_key2index_;
};

uint32_t Object::GetOrAllocRuntimeTypeIndex(const std::string& key, uint32_t static_tindex,
                                            uint32_t parent_tindex, uint32_t num_child_slots,
                                            bool child_slots_can_overflow) {
  return TypeContext::Global()->GetOrAllocRuntimeTypeIndex(
      key, static_tindex, parent_tindex, num_child_slots, child_slots_can_overflow);
}

bool Object::DerivedFrom(uint32_t parent_tindex) const {
  return TypeContext::Global()->DerivedFrom(this->type_index_, parent_tindex);
}

std::string Object::TypeIndex2Key(uint32_t tindex) {
  return TypeContext::Global()->TypeIndex2Key(tindex);
}

size_t Object::TypeIndex2KeyHash(uint32_t tindex) {
  return TypeContext::Global()->TypeIndex2KeyHash(tindex);
}

uint32_t Object::TypeKey2Index(const std::string& key) {
  return TypeContext::Global()->TypeKey2Index(key);
}

TVM_REGISTER_GLOBAL("runtime.ObjectPtrHash").set_body_typed([](ObjectRef obj) {
  return static_cast<int64_t>(ObjectPtrHash()(obj));
});

TVM_REGISTER_GLOBAL("runtime.DumpTypeTable").set_body_typed([](int min_child_count) {
  TypeContext::Global()->Dump(min_child_count);
});
}  // namespace runtime
}  // namespace tvm

int TVMObjectGetTypeIndex(TVMObjectHandle obj, unsigned* out_tindex) {
  API_BEGIN();
  ICHECK(obj != nullptr);
  out_tindex[0] = static_cast<tvm::runtime::Object*>(obj)->type_index();
  API_END();
}

int TVMObjectRetain(TVMObjectHandle obj) {
  API_BEGIN();
  tvm::runtime::ObjectInternal::ObjectRetain(obj);
  API_END();
}

int TVMObjectFree(TVMObjectHandle obj) {
  API_BEGIN();
  tvm::runtime::ObjectInternal::ObjectFree(obj);
  API_END();
}

int TVMObjectDerivedFrom(uint32_t child_type_index, uint32_t parent_type_index, int* is_derived) {
  API_BEGIN();
  *is_derived =
      tvm::runtime::TypeContext::Global()->DerivedFrom(child_type_index, parent_type_index);
  API_END();
}

int TVMObjectTypeKey2Index(const char* type_key, unsigned* out_tindex) {
  API_BEGIN();
  out_tindex[0] = tvm::runtime::ObjectInternal::ObjectTypeKey2Index(type_key);
  API_END();
}

int TVMObjectTypeIndex2Key(unsigned tindex, char** out_type_key) {
  API_BEGIN();
  auto key = tvm::runtime::Object::TypeIndex2Key(tindex);
  *out_type_key = static_cast<char*>(malloc(key.size() + 1));
  strncpy(*out_type_key, key.c_str(), key.size() + 1);
  API_END();
}
