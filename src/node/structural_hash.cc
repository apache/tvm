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
#include <tvm/node/functor.h>
#include <tvm/node/node.h>
#include <tvm/node/reflection.h>
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <unordered_map>

#include "../support/str_escape.h"
#include "../support/utils.h"

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

class VarCountingSHashHandler : public SHashReducer::Handler {
 public:
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

  VarCountingSHashHandler() {}

  void MarkGraphNode() final {
    // need to push to pending tasks in this case
    ICHECK(!allow_push_to_stack_ && !task_stack_.empty());
    task_stack_.back().graph_node_hash = true;
  }

  bool LookupHashedValue(const ObjectRef& key, size_t* hash_value) final {
    auto it = hash_memo_.find(key);
    if (it != hash_memo_.end()) {
      hash_value[0] = it->second;
      return true;
    }
    return false;
  }

  void SHashReduceHashedValue(size_t hashed_value) final {
    pending_tasks_.emplace_back(Task(ObjectRef(nullptr), hashed_value, false));
  }

  void SHashReduceFreeVar(const runtime::Object* var, bool map_free_vars) final {
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

  void SHashReduce(const ObjectRef& object, bool map_free_vars) final {
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
          this->DispatchSHash(entry.object, entry.map_free_vars);
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

  // The default equal as registered in the structural equal vtable.
  void DispatchSHash(const ObjectRef& object, bool map_free_vars) {
    ICHECK(object.defined());
    vtable_->SHashReduce(object.get(), SHashReducer(this, map_free_vars));
  }

 private:
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

TVM_REGISTER_GLOBAL("node.StructuralHash")
    .set_body_typed([](const ObjectRef& object, bool map_free_vars) -> int64_t {
      size_t hashed_value = VarCountingSHashHandler().Hash(object, map_free_vars);
      return static_cast<int64_t>(hashed_value);
    });

size_t StructuralHash::operator()(const ObjectRef& object) const {
  return VarCountingSHashHandler().Hash(object, false);
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

struct NDArrayContainerTrait {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;

  static void SHashReduce(const runtime::NDArray::Container* key, SHashReducer hash_reduce) {
    ICHECK_EQ(key->dl_tensor.device.device_type, kDLCPU) << "can only compare CPU tensor";
    ICHECK(runtime::IsContiguous(key->dl_tensor)) << "Can only hash contiguous tensor";
    hash_reduce(runtime::DataType(key->dl_tensor.dtype));
    hash_reduce(key->dl_tensor.ndim);
    for (int i = 0; i < key->dl_tensor.ndim; ++i) {
      hash_reduce(key->dl_tensor.shape[i]);
    }
    hash_reduce->SHashReduceHashedValue(runtime::String::HashBytes(
        static_cast<const char*>(key->dl_tensor.data), runtime::GetDataSize(key->dl_tensor)));
  }

  static bool SEqualReduce(const runtime::NDArray::Container* lhs,
                           const runtime::NDArray::Container* rhs, SEqualReducer equal) {
    if (lhs == rhs) return true;

    auto ldt = lhs->dl_tensor.dtype;
    auto rdt = rhs->dl_tensor.dtype;
    ICHECK_EQ(lhs->dl_tensor.device.device_type, kDLCPU) << "can only compare CPU tensor";
    ICHECK_EQ(rhs->dl_tensor.device.device_type, kDLCPU) << "can only compare CPU tensor";
    ICHECK(runtime::IsContiguous(lhs->dl_tensor)) << "Can only compare contiguous tensor";
    ICHECK(runtime::IsContiguous(rhs->dl_tensor)) << "Can only compare contiguous tensor";

    if (lhs->dl_tensor.ndim != rhs->dl_tensor.ndim) return false;
    for (int i = 0; i < lhs->dl_tensor.ndim; ++i) {
      if (!equal(lhs->dl_tensor.shape[i], rhs->dl_tensor.shape[i])) return false;
    }
    if (ldt.code == rdt.code && ldt.lanes == rdt.lanes && ldt.bits == rdt.bits) {
      size_t data_size = runtime::GetDataSize(lhs->dl_tensor);
      return std::memcmp(lhs->dl_tensor.data, rhs->dl_tensor.data, data_size) == 0;
    } else {
      return false;
    }
  }
};

TVM_REGISTER_REFLECTION_VTABLE(runtime::NDArray::Container, NDArrayContainerTrait);

struct ArrayNodeTrait {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;

  static void SHashReduce(const ArrayNode* key, SHashReducer hash_reduce) {
    hash_reduce(static_cast<uint64_t>(key->size()));
    for (size_t i = 0; i < key->size(); ++i) {
      hash_reduce(key->at(i));
    }
  }

  static bool SEqualReduce(const ArrayNode* lhs, const ArrayNode* rhs, SEqualReducer equal) {
    if (lhs->size() != rhs->size()) return false;
    for (size_t i = 0; i < lhs->size(); ++i) {
      if (!equal(lhs->at(i), rhs->at(i))) return false;
    }
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

  static bool SEqualReduce(const MapNode* lhs, const MapNode* rhs, SEqualReducer equal) {
    if (rhs->size() != lhs->size()) return false;
    if (rhs->size() == 0) return true;
    bool ls = std::all_of(lhs->begin(), lhs->end(),
                          [](const auto& v) { return v.first->template IsInstance<StringObj>(); });
    bool rs = std::all_of(rhs->begin(), rhs->end(),
                          [](const auto& v) { return v.first->template IsInstance<StringObj>(); });
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

}  // namespace tvm
