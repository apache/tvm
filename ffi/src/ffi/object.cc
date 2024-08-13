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

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace ffi {

/*! \brief Type information */
struct TypeInfo {
  /*! \brief The current index. */
  int32_t index{0};
  /*! \brief Index of the parent in the type hierarchy */
  int32_t parent_index{0};
  // NOTE: the indices in [index, index + num_reserved_slots) are
  // reserved for the child-class of this type.
  /*! \brief Total number of slots reserved for the type and its children. */
  int32_t num_slots{0};
  /*! \brief number of allocated child slots. */
  int32_t allocated_slots{0};
  /*! \brief Whether child can overflow. */
  bool child_slots_can_overflow{true};
  /*! \brief name of the type. */
  std::string name;
  /*! \brief hash of the name */
  size_t name_hash{0};
};

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
class TypeContext {
 public:
  // NOTE: this is a relatively slow path for child checking
  // Most types are already checked by the fast-path via reserved slot checking.
  bool DerivedFrom(int32_t child_tindex, int32_t parent_tindex) {
    // invariance: child's type index is always bigger than its parent.
    if (child_tindex < parent_tindex) return false;
    if (child_tindex == parent_tindex) return true;
    TVM_FFI_ICHECK_LT(child_tindex, type_table_.size());
    while (child_tindex > parent_tindex) {
      child_tindex = type_table_[child_tindex].parent_index;
    }
    return child_tindex == parent_tindex;
  }

  int32_t GetOrAllocRuntimeTypeIndex(const std::string& skey, int32_t static_tindex,
                                     int32_t parent_tindex, int32_t num_child_slots,
                                     bool child_slots_can_overflow) {
    auto it = type_key2index_.find(skey);
    if (it != type_key2index_.end()) {
      return it->second;
    }
    // try to allocate from parent's type table.
    TVM_FFI_ICHECK_LT(parent_tindex, type_table_.size())
        << " skey=" << skey << ", static_index=" << static_tindex;

    TypeInfo& pinfo = type_table_[parent_tindex];
    TVM_FFI_ICHECK_EQ(pinfo.index, parent_tindex);

    // if parent cannot overflow, then this class cannot.
    if (!pinfo.child_slots_can_overflow) {
      child_slots_can_overflow = false;
    }

    // total number of slots include the type itself.
    int32_t num_slots = num_child_slots + 1;
    int32_t allocated_tindex;

    if (static_tindex > 0) {
      // statically assigned type
      allocated_tindex = static_tindex;
      TVM_FFI_ICHECK_LT(static_tindex, type_table_.size());
      TVM_FFI_ICHECK_EQ(type_table_[allocated_tindex].allocated_slots, 0U)
          << "Conflicting static index " << static_tindex << " between "
          << type_table_[allocated_tindex].name << " and " << skey;
    } else if (pinfo.allocated_slots + num_slots <= pinfo.num_slots) {
      // allocate the slot from parent's reserved pool
      allocated_tindex = parent_tindex + pinfo.allocated_slots;
      // update parent's state
      pinfo.allocated_slots += num_slots;
    } else {
      TVM_FFI_ICHECK(pinfo.child_slots_can_overflow)
          << "Reach maximum number of sub-classes for " << pinfo.name;
      // allocate new entries.
      allocated_tindex = type_counter_;
      type_counter_ += num_slots;
      TVM_FFI_ICHECK_LE(type_table_.size(), type_counter_);
      type_table_.resize(type_counter_, TypeInfo());
    }
    TVM_FFI_ICHECK_GT(allocated_tindex, parent_tindex);
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

  const std::string& TypeIndex2Key(int32_t tindex) {
    if (tindex != 0) {
      // always return the right type key for root
      // for non-root type nodes, allocated slots should not equal 0
      TVM_FFI_ICHECK(tindex < static_cast<int32_t>(type_table_.size()) &&
                     type_table_[tindex].allocated_slots != 0)
          << "Unknown type index " << tindex;
    }
    return type_table_[tindex].name;
  }

  size_t TypeIndex2KeyHash(int32_t tindex) {
    TVM_FFI_ICHECK(tindex < static_cast<int32_t>(type_table_.size()) &&
                   type_table_[tindex].allocated_slots != 0)
        << "Unknown type index " << tindex;
    return type_table_[tindex].name_hash;
  }

  int32_t TypeKey2Index(const std::string& skey) {
    auto it = type_key2index_.find(skey);
    TVM_FFI_ICHECK(it != type_key2index_.end()) << "Cannot find type " << skey;
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
    type_table_.resize(TypeIndex::kTVMFFIDynObjectBegin, TypeInfo());
    type_table_[0].name = "runtime.Object";
  }

  int32_t type_counter_{TypeIndex::kTVMFFIDynObjectBegin};
  std::vector<TypeInfo> type_table_;
  std::unordered_map<std::string, int32_t> type_key2index_;
};

namespace details {

int32_t ObjectGetOrAllocTypeIndex(const char* type_key, int32_t static_tindex,
                                  int32_t parent_tindex, int32_t type_child_slots,
                                  bool type_child_slots_can_overflow) {
  return tvm::ffi::TypeContext::Global()->GetOrAllocRuntimeTypeIndex(
      type_key, static_tindex, parent_tindex, type_child_slots, type_child_slots_can_overflow != 0);
}

bool ObjectDerivedFrom(int32_t child_type_index, int32_t parent_type_index) {
  return static_cast<int>(
      tvm::ffi::TypeContext::Global()->DerivedFrom(child_type_index, parent_type_index));
}
}  // namespace details
}  // namespace ffi
}  // namespace tvm
