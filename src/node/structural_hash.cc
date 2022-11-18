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
 * \file src/node/structural_hash.cc
 */
#include <dmlc/memory_io.h>
#include <tvm/node/functor.h>
#include <tvm/node/node.h>
#include <tvm/node/object_path.h>
#include <tvm/node/reflection.h>
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <unordered_map>

#include "../support/base64.h"
#include "../support/str_escape.h"
#include "../support/utils.h"
#include "ndarray_hash_equal.h"

namespace tvm {

// Define the dispatch functio here since primary user is in this file.
void ReflectionVTable::SHashReduce(const Object* self, SHashReducer reducer) const {
  uint32_t tindex = self->type_index();
  if (tindex >= fshash_reduce_.size() || fshash_reduce_[tindex] == nullptr) {
    LOG(FATAL) << "TypeError: SHashReduce of " << self->GetTypeKey()
               << " is not registered via TVM_REGISTER_NODE_TYPE";
  }
  fshash_reduce_[tindex](self, reducer);
}

// Hash handler that handles free vars
// by assigning an unique counter in the order of their ocurrence.
//
// This algorithm depends on the determinism of the traversal of SHash function.
// In particular, when we traverse unordered_map, we should first sort
// the entries by keys(or hash of keys) before traversing.

class SHashHandlerDefault::Impl {
 public:
  explicit Impl(SHashHandlerDefault* parent) : parent_(parent) {}

  /*! \brief Pending reduce tasks. */
  struct Task {
    /*!
     * \brief The object operand to be hashed.
     *  If the object is nullptr, then the reduced hash is already set
     *  the correct value.
     */
    ObjectRef object;
    /*! \brief The partially reduce hash value.*/
    size_t reduced_hash;
    /*! \brief The expected location in the result stack. */
    size_t result_stack_index = std::numeric_limits<size_t>::max();
    /*! \brief Whether the children has been expanded via SEqualReduce */
    bool children_expanded{false};
    /*! \brief Whether the node is graph node. */
    bool graph_node_hash{false};
    /*! \brief whether to map the free variables. */
    bool map_free_vars;

    Task() = default;
    explicit Task(ObjectRef object, size_t reduced_hash, bool map_free_vars)
        : object(object), reduced_hash(reduced_hash), map_free_vars(map_free_vars) {}
  };

  void MarkGraphNode() {
    // need to push to pending tasks in this case
    ICHECK(!allow_push_to_stack_ && !task_stack_.empty());
    task_stack_.back().graph_node_hash = true;
  }

  bool LookupHashedValue(const ObjectRef& key, size_t* hash_value) {
    auto it = hash_memo_.find(key);
    if (it != hash_memo_.end()) {
      hash_value[0] = it->second;
      return true;
    }
    return false;
  }

  void SHashReduceHashedValue(size_t hashed_value) {
    pending_tasks_.emplace_back(Task(ObjectRef(nullptr), hashed_value, false));
  }

  void SHashReduceFreeVar(const runtime::Object* var, bool map_free_vars) {
    ICHECK(!hash_memo_.count(GetRef<ObjectRef>(var)));
    if (map_free_vars) {
      // use counter value.
      size_t value = std::hash<size_t>()(free_var_counter_++);
      pending_tasks_.emplace_back(Task(ObjectRef(nullptr), value, false));
    } else {
      // use pointer hash
      size_t value = std::hash<const runtime::Object*>()(var);
      pending_tasks_.emplace_back(Task(ObjectRef(nullptr), value, false));
    }
  }

  void SHashReduce(const ObjectRef& object, bool map_free_vars) {
    // Directly push the result
    // Note: it is still important to push the result to pendng tasks
    // so that the reduction order of hash values stays the same.
    if (!object.defined()) {
      pending_tasks_.emplace_back(Task(ObjectRef(nullptr), 0, false));
      return;
    }
    auto it = hash_memo_.find(object);
    if (it != hash_memo_.end()) {
      pending_tasks_.emplace_back(Task(ObjectRef(nullptr), it->second, false));
    } else {
      // Push a pending task with initial value.
      pending_tasks_.emplace_back(Task(object, object->GetTypeKeyHash(), map_free_vars));
    }
  }

  size_t Hash(const ObjectRef& object, bool map_free_vars) {
    ICHECK_EQ(task_stack_.size(), 0U);
    ICHECK_EQ(pending_tasks_.size(), 0U);
    ICHECK_EQ(result_stack_.size(), 0U);

    this->SHashReduce(object, map_free_vars);
    ICHECK_EQ(pending_tasks_.size(), 1U);
    ICHECK(allow_push_to_stack_);
    task_stack_.emplace_back(std::move(pending_tasks_.back()));
    pending_tasks_.clear();

    this->RunTasks();

    ICHECK_EQ(result_stack_.size(), 1U);
    size_t ret = result_stack_.back();
    result_stack_.pop_back();
    return ret;
  }

  void DispatchSHash(const ObjectRef& object, bool map_free_vars) {
    ICHECK(object.defined());
    vtable_->SHashReduce(object.get(), SHashReducer(parent_, map_free_vars));
  }

 protected:
  /*!
   * \brief Pop the top entry of the task stack and push the hash into the result stack.
   */
  void PopTaskStack() {
    const auto& entry = task_stack_.back();
    result_stack_.push_back(entry.reduced_hash);
    task_stack_.pop_back();
  }
  /*!
   * \brief Compute the reduced hash value for the task.
   * \param task The indicated task.
   */
  size_t ReduceHash(const Task& task) {
    size_t stack_begin = task.result_stack_index;
    ICHECK_LE(stack_begin, result_stack_.size());

    // combine in the reverse order of the stack.
    size_t reduced_hash = task.reduced_hash;
    for (size_t i = result_stack_.size(); i != stack_begin; --i) {
      reduced_hash = support::HashCombine(reduced_hash, result_stack_[i - 1]);
    }
    result_stack_.resize(stack_begin);
    return reduced_hash;
  }
  // run the tasks.
  void RunTasks() {
    while (task_stack_.size() != 0) {
      // Caution: entry becomes invalid when the stack changes
      auto& entry = task_stack_.back();
      if (entry.children_expanded) {
        // reduce hash
        entry.reduced_hash = ReduceHash(entry);
        // When all the children has expanded and visited.
        // entry.reduced_hash contains the reduced hash result.
        auto it = hash_memo_.find(entry.object);
        if (it != hash_memo_.end()) {
          // use the pre-computed hash for the object.
          entry.reduced_hash = it->second;
        } else {
          // Append the graph node counter to the hash
          // so that we can distinguish DAG from trees.
          if (entry.graph_node_hash) {
            entry.reduced_hash = support::HashCombine(entry.reduced_hash,
                                                      std::hash<size_t>()(graph_node_counter_++));
          }
          hash_memo_[entry.object] = entry.reduced_hash;
        }
        // send value to parent.
        this->PopTaskStack();
      } else if (!entry.object.defined()) {
        // Directly send value to parent
        this->PopTaskStack();
      } else {
        // check if there are already hash for object.
        auto it = hash_memo_.find(entry.object);
        if (it != hash_memo_.end()) {
          entry.reduced_hash = it->second;
          this->PopTaskStack();
        } else {
          // NOTE: important to modify entry before visit.
          // as entry becomes invalid after we change the stack.
          entry.children_expanded = true;
          entry.result_stack_index = result_stack_.size();

          ICHECK_EQ(pending_tasks_.size(), 0U);
          allow_push_to_stack_ = false;
          // dispatch hash, reduce to the current slot.
          parent_->DispatchSHash(entry.object, entry.map_free_vars);
          allow_push_to_stack_ = true;
          // Move pending tasks to the stack until the marked point.
          while (pending_tasks_.size() != 0) {
            task_stack_.emplace_back(std::move(pending_tasks_.back()));
            pending_tasks_.pop_back();
          }
        }
      }
    }
  }

 private:
  // The owner of this impl
  SHashHandlerDefault* parent_;
  // free var counter.
  size_t free_var_counter_{0};
  // graph node counter.
  size_t graph_node_counter_{0};
  // record current stack top
  bool allow_push_to_stack_{true};
  // list of pending tasks to be pushed to the stack.
  std::vector<Task> pending_tasks_;
  // Internal task stack to executed the task
  std::vector<Task> task_stack_;
  // Internal stack to store the result poped from the task stack.
  std::vector<size_t> result_stack_;
  // reflection vtable
  ReflectionVTable* vtable_ = ReflectionVTable::Global();
  // map from lhs to rhs
  std::unordered_map<ObjectRef, size_t, ObjectPtrHash, ObjectPtrEqual> hash_memo_;
};

SHashHandlerDefault::SHashHandlerDefault() { impl = new Impl(this); }
SHashHandlerDefault::~SHashHandlerDefault() { delete impl; }

void SHashHandlerDefault::SHashReduceHashedValue(size_t hashed_value) {
  return impl->SHashReduceHashedValue(hashed_value);
}

void SHashHandlerDefault::SHashReduce(const ObjectRef& key, bool map_free_vars) {
  impl->SHashReduce(key, map_free_vars);
}

void SHashHandlerDefault::SHashReduceFreeVar(const runtime::Object* var, bool map_free_vars) {
  impl->SHashReduceFreeVar(var, map_free_vars);
}

bool SHashHandlerDefault::LookupHashedValue(const ObjectRef& key, size_t* hashed_value) {
  return impl->LookupHashedValue(key, hashed_value);
}

void SHashHandlerDefault::MarkGraphNode() { impl->MarkGraphNode(); }

size_t SHashHandlerDefault::Hash(const ObjectRef& object, bool map_free_vars) {
  return impl->Hash(object, map_free_vars);
}

void SHashHandlerDefault::DispatchSHash(const ObjectRef& key, bool map_free_vars) {
  impl->DispatchSHash(key, map_free_vars);
}

TVM_REGISTER_GLOBAL("node.StructuralHash")
    .set_body_typed([](const ObjectRef& object, bool map_free_vars) -> int64_t {
      size_t hashed_value = SHashHandlerDefault().Hash(object, map_free_vars);
      return static_cast<int64_t>(hashed_value);
    });

size_t StructuralHash::operator()(const ObjectRef& object) const {
  return SHashHandlerDefault().Hash(object, false);
}

// SEQualReduce traits for runtime containers.
struct StringObjTrait {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;

  static void SHashReduce(const runtime::StringObj* key, SHashReducer hash_reduce) {
    hash_reduce->SHashReduceHashedValue(runtime::String::HashBytes(key->data, key->size));
  }

  static bool SEqualReduce(const runtime::StringObj* lhs, const runtime::StringObj* rhs,
                           SEqualReducer equal) {
    if (lhs == rhs) return true;
    if (lhs->size != rhs->size) return false;
    if (lhs->data == rhs->data) return true;
    return std::memcmp(lhs->data, rhs->data, lhs->size) == 0;
  }
};

struct RefToObjectPtr : public ObjectRef {
  static ObjectPtr<Object> Get(const ObjectRef& ref) { return GetDataPtr<Object>(ref); }
};

TVM_REGISTER_REFLECTION_VTABLE(runtime::StringObj, StringObjTrait)
    .set_creator([](const std::string& bytes) {
      return RefToObjectPtr::Get(runtime::String(bytes));
    })
    .set_repr_bytes([](const Object* n) -> std::string {
      return GetRef<runtime::String>(static_cast<const runtime::StringObj*>(n))
          .
          operator std::string();
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<runtime::StringObj>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const runtime::StringObj*>(node.get());
      p->stream << '"' << support::StrEscape(op->data, op->size) << '"';
    });

struct ADTObjTrait {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;

  static void SHashReduce(const runtime::ADTObj* key, SHashReducer hash_reduce) {
    hash_reduce(key->tag);
    hash_reduce(static_cast<uint64_t>(key->size));
    for (uint32_t i = 0; i < key->size; ++i) {
      hash_reduce((*key)[i]);
    }
  }

  static bool SEqualReduce(const runtime::ADTObj* lhs, const runtime::ADTObj* rhs,
                           SEqualReducer equal) {
    if (lhs == rhs) return true;
    if (lhs->tag != rhs->tag) return false;
    if (lhs->size != rhs->size) return false;

    for (uint32_t i = 0; i < lhs->size; ++i) {
      if (!equal((*lhs)[i], (*rhs)[i])) return false;
    }
    return true;
  }
};

TVM_REGISTER_REFLECTION_VTABLE(runtime::ADTObj, ADTObjTrait);

void NDArrayHash(const runtime::NDArray::Container* arr, SHashReducer* hash_reduce,
                 bool hash_data) {
  ICHECK_EQ(arr->dl_tensor.device.device_type, kDLCPU) << "can only compare CPU tensor";
  ICHECK(runtime::IsContiguous(arr->dl_tensor)) << "Can only hash contiguous tensor";
  (*hash_reduce)(runtime::DataType(arr->dl_tensor.dtype));
  (*hash_reduce)(arr->dl_tensor.ndim);
  for (int i = 0; i < arr->dl_tensor.ndim; ++i) {
    (*hash_reduce)(arr->dl_tensor.shape[i]);
  }
  if (hash_data) {
    (*hash_reduce)
        ->SHashReduceHashedValue(runtime::String::HashBytes(
            static_cast<const char*>(arr->dl_tensor.data), runtime::GetDataSize(arr->dl_tensor)));
  }
}

void NDArrayContainerTrait::SHashReduce(const runtime::NDArray::Container* key,
                                        SHashReducer hash_reduce) {
  NDArrayHash(key, &hash_reduce, /*bool hash_data*/ true);
}

TVM_REGISTER_REFLECTION_VTABLE(runtime::NDArray::Container, NDArrayContainerTrait)
    .set_creator([](const std::string& blob) {
      dmlc::MemoryStringStream mstrm(const_cast<std::string*>(&blob));
      support::Base64InStream b64strm(&mstrm);
      b64strm.InitPosition();
      runtime::NDArray temp;
      ICHECK(temp.Load(&b64strm));
      return RefToObjectPtr::Get(temp);
    })
    .set_repr_bytes([](const Object* n) -> std::string {
      std::string blob;
      dmlc::MemoryStringStream mstrm(&blob);
      support::Base64OutStream b64strm(&mstrm);
      const auto* ndarray = static_cast<const runtime::NDArray::Container*>(n);
      runtime::SaveDLTensor(&b64strm, &ndarray->dl_tensor);
      b64strm.Finish();
      return blob;
    });

struct ArrayNodeTrait {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;

  static void SHashReduce(const ArrayNode* key, SHashReducer hash_reduce) {
    hash_reduce(static_cast<uint64_t>(key->size()));
    for (size_t i = 0; i < key->size(); ++i) {
      hash_reduce(key->at(i));
    }
  }

  static bool SEqualReduce(const ArrayNode* lhs, const ArrayNode* rhs, SEqualReducer equal) {
    if (equal.IsPathTracingEnabled()) {
      return SEqualReduceTraced(lhs, rhs, equal);
    }

    if (lhs->size() != rhs->size()) return false;
    for (size_t i = 0; i < lhs->size(); ++i) {
      if (!equal(lhs->at(i), rhs->at(i))) return false;
    }
    return true;
  }

 private:
  static bool SEqualReduceTraced(const ArrayNode* lhs, const ArrayNode* rhs,
                                 const SEqualReducer& equal) {
    size_t min_size = std::min(lhs->size(), rhs->size());
    const ObjectPathPair& array_paths = equal.GetCurrentObjectPaths();

    for (size_t index = 0; index < min_size; ++index) {
      ObjectPathPair element_paths = {array_paths->lhs_path->ArrayIndex(index),
                                      array_paths->rhs_path->ArrayIndex(index)};
      if (!equal(lhs->at(index), rhs->at(index), element_paths)) {
        return false;
      }
    }

    if (lhs->size() == rhs->size()) {
      return true;
    }

    // If the array length is mismatched, don't report it immediately.
    // Instead, defer the failure until we visit all children.
    //
    // This is for human readability. For example, say we have two sequences
    //
    //    (1)     a b c d e f g h i j k l m
    //    (2)     a b c d e g h i j k l m
    //
    // If we directly report a mismatch at the end of the array right now,
    // the user will see that array (1) has an element `m` at index 12 but array (2)
    // has no index 12 because it's too short:
    //
    //    (1)     a b c d e f g h i j k l m
    //                                    ^error here
    //    (2)     a b c d e g h i j k l m
    //                                    ^ error here
    //
    // This is not very helpful. Instead, if we defer reporting this mismatch until all elements
    // are fully visited, we can be much more helpful with pointing out the location:
    //
    //    (1)     a b c d e f g h i j k l m
    //                      ^
    //                   error here
    //
    //    (2)     a b c d e g h i j k l m
    //                      ^
    //                  error here
    if (lhs->size() > min_size) {
      equal->DeferFail({array_paths->lhs_path->ArrayIndex(min_size),
                        array_paths->rhs_path->MissingArrayElement(min_size)});
    } else {
      equal->DeferFail({array_paths->lhs_path->MissingArrayElement(min_size),
                        array_paths->rhs_path->ArrayIndex(min_size)});
    }

    // Can return `true` pretending that everything is good since we have deferred the failure.
    return true;
  }
};
TVM_REGISTER_REFLECTION_VTABLE(ArrayNode, ArrayNodeTrait)
    .set_creator([](const std::string&) -> ObjectPtr<Object> {
      return ::tvm::runtime::make_object<ArrayNode>();
    });

struct MapNodeTrait {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;

  static void SHashReduceForOMap(const MapNode* key, SHashReducer hash_reduce) {
    // SHash's var handling depends on the determinism of traversal.
    // NOTE: only book-keep the mapped hash keys.
    // This resolves common use cases where we want to store
    // Map<Var, Value> where Var is defined in the function
    // parameters.
    using KV = std::pair<size_t, ObjectRef>;
    std::vector<KV> temp;
    for (const auto& kv : *key) {
      size_t hashed_value;
      if (hash_reduce->LookupHashedValue(kv.first, &hashed_value)) {
        temp.emplace_back(hashed_value, kv.second);
      }
    }
    // sort by the hash key of the keys.
    std::sort(temp.begin(), temp.end(),
              [](const KV& lhs, const KV& rhs) { return lhs.first < rhs.first; });
    // add size to the hash
    hash_reduce(static_cast<uint64_t>(key->size()));
    // hash the content
    for (size_t i = 0; i < temp.size();) {
      size_t k = i + 1;
      for (; k < temp.size() && temp[k].first == temp[i].first; ++k) {
      }
      // ties are rare, but we need to skip them to make the hash determinsitic
      if (k == i + 1) {
        hash_reduce->SHashReduceHashedValue(temp[i].first);
        hash_reduce(temp[i].second);
      }
      i = k;
    }
  }

  static void SHashReduceForSMap(const MapNode* key, SHashReducer hash_reduce) {
    // NOTE: only book-keep the mapped hash keys.
    // This resolves common use cases where we want to store
    // Map<Var, Value> where Var is defined in the function
    // parameters.
    using KV = std::pair<String, ObjectRef>;
    std::vector<KV> temp;
    for (const auto& kv : *key) {
      temp.push_back(std::make_pair(Downcast<String>(kv.first), kv.second));
    }
    // sort by the hash key of the keys.
    std::sort(temp.begin(), temp.end(),
              [](const KV& lhs, const KV& rhs) { return lhs.first < rhs.first; });
    // NOTE: we won't have ties
    // add size to the hash after sorting.
    hash_reduce(static_cast<uint64_t>(key->size()));
    // hash the content
    for (size_t i = 0; i < temp.size(); ++i) {
      hash_reduce(temp[i].first);
      hash_reduce(temp[i].second);
    }
  }

  static void SHashReduce(const MapNode* key, SHashReducer hash_reduce) {
    bool is_str_map = std::all_of(key->begin(), key->end(), [](const auto& v) {
      return v.first->template IsInstance<StringObj>();
    });
    if (is_str_map) {
      SHashReduceForSMap(key, hash_reduce);
    } else {
      SHashReduceForOMap(key, hash_reduce);
    }
  }

  static bool SEqualReduceForOMap(const MapNode* lhs, const MapNode* rhs, SEqualReducer equal) {
    for (const auto& kv : *lhs) {
      // Only allow equal checking if the keys are already mapped
      // This resolves common use cases where we want to store
      // Map<Var, Value> where Var is defined in the function
      // parameters.
      ObjectRef rhs_key = equal->MapLhsToRhs(kv.first);
      if (!rhs_key.defined()) return false;
      auto it = rhs->find(rhs_key);
      if (it == rhs->end()) return false;
      if (!equal(kv.second, it->second)) return false;
    }
    return true;
  }

  static bool SEqualReduceForSMap(const MapNode* lhs, const MapNode* rhs, SEqualReducer equal) {
    for (const auto& kv : *lhs) {
      auto it = rhs->find(kv.first);
      if (it == rhs->end()) return false;
      if (!equal(kv.second, it->second)) return false;
    }
    return true;
  }

  static bool IsStringMap(const MapNode* map) {
    return std::all_of(map->begin(), map->end(),
                       [](const auto& v) { return v.first->template IsInstance<StringObj>(); });
  }

  static bool SEqualReduceTracedForOMap(const MapNode* lhs, const MapNode* rhs,
                                        const SEqualReducer& equal) {
    const ObjectPathPair& map_paths = equal.GetCurrentObjectPaths();

    std::vector<const Object*> seen_rhs_keys;

    // First, check that every key from `lhs` is also in `rhs`,
    // and their values are mapped to each other.
    for (const auto& kv : *lhs) {
      ObjectPath lhs_path = map_paths->lhs_path->MapValue(kv.first);

      ObjectRef rhs_key = equal->MapLhsToRhs(kv.first);
      if (!rhs_key.defined()) {
        equal.RecordMismatchPaths({lhs_path, map_paths->rhs_path->MissingMapEntry()});
        return false;
      }

      auto it = rhs->find(rhs_key);
      if (it == rhs->end()) {
        equal.RecordMismatchPaths({lhs_path, map_paths->rhs_path->MissingMapEntry()});
        return false;
      }

      if (!equal(kv.second, it->second, {lhs_path, map_paths->rhs_path->MapValue(it->first)})) {
        return false;
      }

      seen_rhs_keys.push_back(it->first.get());
    }

    std::sort(seen_rhs_keys.begin(), seen_rhs_keys.end());

    // Second, check that we have visited every `rhs` key when iterating over `lhs`.
    for (const auto& kv : *rhs) {
      if (!std::binary_search(seen_rhs_keys.begin(), seen_rhs_keys.end(), kv.first.get())) {
        equal.RecordMismatchPaths(
            {map_paths->lhs_path->MissingMapEntry(), map_paths->rhs_path->MapValue(kv.first)});
        return false;
      }
    }

    ICHECK(lhs->size() == rhs->size());
    return true;
  }

  static bool SEqualReduceTracedForSMap(const MapNode* lhs, const MapNode* rhs,
                                        const SEqualReducer& equal) {
    const ObjectPathPair& map_paths = equal.GetCurrentObjectPaths();

    // First, check that every key from `lhs` is also in `rhs`, and their values are equal.
    for (const auto& kv : *lhs) {
      ObjectPath lhs_path = map_paths->lhs_path->MapValue(kv.first);
      auto it = rhs->find(kv.first);
      if (it == rhs->end()) {
        equal.RecordMismatchPaths({lhs_path, map_paths->rhs_path->MissingMapEntry()});
        return false;
      }

      if (!equal(kv.second, it->second, {lhs_path, map_paths->rhs_path->MapValue(it->first)})) {
        return false;
      }
    }

    // Second, make sure every key from `rhs` is also in `lhs`.
    for (const auto& kv : *rhs) {
      ObjectPath rhs_path = map_paths->rhs_path->MapValue(kv.first);
      if (!lhs->count(kv.first)) {
        equal.RecordMismatchPaths({map_paths->lhs_path->MissingMapEntry(), rhs_path});
        return false;
      }
    }

    ICHECK(lhs->size() == rhs->size());
    return true;
  }

  static bool SEqualReduceTraced(const MapNode* lhs, const MapNode* rhs,
                                 const SEqualReducer& equal) {
    if (IsStringMap(lhs)) {
      return SEqualReduceTracedForSMap(lhs, rhs, equal);
    } else {
      return SEqualReduceTracedForOMap(lhs, rhs, equal);
    }
  }

  static bool SEqualReduce(const MapNode* lhs, const MapNode* rhs, SEqualReducer equal) {
    if (equal.IsPathTracingEnabled()) {
      return SEqualReduceTraced(lhs, rhs, equal);
    }

    if (rhs->size() != lhs->size()) return false;
    if (rhs->size() == 0) return true;
    bool ls = IsStringMap(lhs);
    bool rs = IsStringMap(rhs);
    if (ls != rs) {
      return false;
    }
    return (ls && rs) ? SEqualReduceForSMap(lhs, rhs, equal) : SEqualReduceForOMap(lhs, rhs, equal);
  }
};
TVM_REGISTER_REFLECTION_VTABLE(MapNode, MapNodeTrait)
    .set_creator([](const std::string&) -> ObjectPtr<Object> { return MapNode::Empty(); });

struct ReportNodeTrait {
  static void VisitAttrs(runtime::profiling::ReportNode* report, AttrVisitor* attrs) {
    attrs->Visit("calls", &report->calls);
    attrs->Visit("device_metrics", &report->device_metrics);
    attrs->Visit("configuration", &report->configuration);
  }
  static constexpr std::nullptr_t SEqualReduce = nullptr;
  static constexpr std::nullptr_t SHashReduce = nullptr;
};
TVM_REGISTER_REFLECTION_VTABLE(runtime::profiling::ReportNode, ReportNodeTrait);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<runtime::profiling::ReportNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const runtime::profiling::ReportNode*>(node.get());
      p->stream << op->AsTable();
    });

struct CountNodeTrait {
  static void VisitAttrs(runtime::profiling::CountNode* n, AttrVisitor* attrs) {
    attrs->Visit("value", &n->value);
  }
  static constexpr std::nullptr_t SEqualReduce = nullptr;
  static constexpr std::nullptr_t SHashReduce = nullptr;
};
TVM_REGISTER_REFLECTION_VTABLE(runtime::profiling::CountNode, CountNodeTrait);
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<runtime::profiling::CountNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const runtime::profiling::CountNode*>(node.get());
      p->stream << op->GetTypeKey() << "(" << op->value << ")";
    });
struct DurationNodeTrait {
  static void VisitAttrs(runtime::profiling::DurationNode* n, AttrVisitor* attrs) {
    attrs->Visit("microseconds", &n->microseconds);
  }
  static constexpr std::nullptr_t SEqualReduce = nullptr;
  static constexpr std::nullptr_t SHashReduce = nullptr;
};
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<runtime::profiling::DurationNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const runtime::profiling::DurationNode*>(node.get());
      p->stream << op->GetTypeKey() << "(" << op->microseconds << ")";
    });
TVM_REGISTER_REFLECTION_VTABLE(runtime::profiling::DurationNode, DurationNodeTrait);
struct PercentNodeTrait {
  static void VisitAttrs(runtime::profiling::PercentNode* n, AttrVisitor* attrs) {
    attrs->Visit("percent", &n->percent);
  }
  static constexpr std::nullptr_t SEqualReduce = nullptr;
  static constexpr std::nullptr_t SHashReduce = nullptr;
};
TVM_REGISTER_REFLECTION_VTABLE(runtime::profiling::PercentNode, PercentNodeTrait);
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<runtime::profiling::PercentNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const runtime::profiling::PercentNode*>(node.get());
      p->stream << op->GetTypeKey() << "(" << op->percent << ")";
    });
struct RatioNodeTrait {
  static void VisitAttrs(runtime::profiling::RatioNode* n, AttrVisitor* attrs) {
    attrs->Visit("ratio", &n->ratio);
  }
  static constexpr std::nullptr_t SEqualReduce = nullptr;
  static constexpr std::nullptr_t SHashReduce = nullptr;
};
TVM_REGISTER_REFLECTION_VTABLE(runtime::profiling::RatioNode, RatioNodeTrait);
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<runtime::profiling::RatioNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const runtime::profiling::RatioNode*>(node.get());
      p->stream << op->GetTypeKey() << "(" << op->ratio << ")";
    });

}  // namespace tvm
