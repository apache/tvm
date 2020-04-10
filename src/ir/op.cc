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
 * \file src/ir/op.cc
 * \brief Primitive operators and intrinsics.
 */
#include <tvm/ir/op.h>
#include <tvm/ir/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/packed_func.h>

#include <memory>
#include <mutex>

namespace dmlc {
// enable registry
DMLC_REGISTRY_ENABLE(::tvm::OpRegistry);
}  // namespace dmlc

namespace tvm {

using runtime::TVMRetValue;
using runtime::TVMArgs;
using runtime::PackedFunc;

::dmlc::Registry<OpRegistry>* OpRegistry::Registry() {
  return ::dmlc::Registry<OpRegistry>::Get();
}

// single manager of operator information.
struct OpManager {
  // mutex to avoid registration from multiple threads.
  std::mutex mutex;
  // global operator counter
  std::atomic<int> op_counter{0};
  // storage of additional attribute table.
  std::unordered_map<std::string, std::unique_ptr<GenericOpMap>> attr;
  // frontend functions
  std::vector<PackedFunc*> frontend_funcs;
  // get singleton of the op manager
  static OpManager* Global() {
    static OpManager* inst = new OpManager();
    return inst;
  }
};

// find operator by name
const Op& Op::Get(const std::string& name) {
  const OpRegistry* reg = dmlc::Registry<OpRegistry>::Find(name);
  CHECK(reg != nullptr) << "Operator " << name << " is not registered";
  return reg->op();
}

OpRegistry::OpRegistry() {
  OpManager* mgr = OpManager::Global();
  ObjectPtr<OpNode> n = make_object<OpNode>();
  n->index_ = mgr->op_counter++;
  op_ = Op(n);
}

// Get attribute map by key
const GenericOpMap& Op::GetGenericAttr(const std::string& key) {
  OpManager* mgr = OpManager::Global();
  std::lock_guard<std::mutex> lock(mgr->mutex);
  auto it = mgr->attr.find(key);
  if (it == mgr->attr.end()) {
    LOG(FATAL) << "Operator attribute \'" << key << "\' is not registered";
  }
  return *it->second.get();
}

// Check if a key is present in the registry.
bool Op::HasGenericAttr(const std::string& key) {
  OpManager* mgr = OpManager::Global();
  std::lock_guard<std::mutex> lock(mgr->mutex);
  auto it = mgr->attr.find(key);
  if (it == mgr->attr.end()) {
    return false;
  }
  return true;
}

// Resets attr of the OpMap.
void OpRegistry::reset_attr(const std::string& key) {
  OpManager* mgr = OpManager::Global();
  std::lock_guard<std::mutex> lock(mgr->mutex);
  std::unique_ptr<GenericOpMap>& op_map = mgr->attr[key];
  if (op_map == nullptr) {
    return;
  }
  uint32_t index = op_->index_;
  if (op_map->data_.size() > index) {
    op_map->data_[index] = std::make_pair(TVMRetValue(), 0);
  }
}

void OpRegistry::UpdateAttr(const std::string& key,
                            TVMRetValue value,
                            int plevel) {
  OpManager* mgr = OpManager::Global();
  std::lock_guard<std::mutex> lock(mgr->mutex);
  std::unique_ptr<GenericOpMap>& op_map = mgr->attr[key];
  if (op_map == nullptr) {
    op_map.reset(new GenericOpMap());
    op_map->attr_name_ = key;
  }
  uint32_t index = op_->index_;
  if (op_map->data_.size() <= index) {
    op_map->data_.resize(index + 1, std::make_pair(TVMRetValue(), 0));
  }
  std::pair<TVMRetValue, int>& p = op_map->data_[index];
  CHECK(p.second != plevel)
      << "Attribute " << key << " of operator " << this->name
      << " is already registered with same plevel=" << plevel;
  CHECK(value.type_code() != kTVMNullptr)
      << "Registered packed_func is Null for " << key
      << " of operator " << this->name;
  if (p.second < plevel && value.type_code() != kTVMNullptr) {
    op_map->data_[index] = std::make_pair(value, plevel);
  }
}

// Frontend APIs
TVM_REGISTER_GLOBAL("relay.op._ListOpNames")
.set_body_typed([]() {
    Array<runtime::String> ret;
    for (const std::string& name : dmlc::Registry<OpRegistry>::ListAllNames()) {
      ret.push_back(name);
    }
    return ret;
  });

TVM_REGISTER_GLOBAL("relay.op._GetOp").set_body_typed(Op::Get);

TVM_REGISTER_GLOBAL("relay.op._OpGetAttr")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Op op = args[0];
    std::string attr_name = args[1];
    auto op_map = Op::GetAttr<TVMRetValue>(attr_name);
    if (op_map.count(op)) {
      *rv = op_map[op];
    }
  });

TVM_REGISTER_GLOBAL("relay.op._OpSetAttr")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Op op = args[0];
    std::string attr_name = args[1];
    runtime::TVMArgValue value = args[2];
    int plevel = args[3];
    auto& reg =
        OpRegistry::Registry()->__REGISTER_OR_GET__(op->name).set_name();
    reg.set_attr(attr_name, value, plevel);
  });

TVM_REGISTER_GLOBAL("relay.op._OpResetAttr")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Op op = args[0];
    std::string attr_name = args[1];
    auto& reg =
        OpRegistry::Registry()->__REGISTER_OR_GET__(op->name);
    reg.reset_attr(attr_name);
  });

TVM_REGISTER_GLOBAL("relay.op._Register")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::string op_name = args[0];
    std::string attr_key = args[1];
    runtime::TVMArgValue value = args[2];
    int plevel = args[3];
    auto& reg =
        OpRegistry::Registry()->__REGISTER_OR_GET__(op_name).set_name();
    // enable resgiteration and override of certain properties
    if (attr_key == "num_inputs" && plevel > 128) {
      reg.set_num_inputs(value);
    } else if (attr_key == "attrs_type_key" && plevel > 128) {
      LOG(FATAL) << "attrs type key no longer supported";
    } else {
      // normal attr table override.
      if (args[2].type_code() == kTVMPackedFuncHandle) {
        // do an eager copy of the PackedFunc
        PackedFunc f = args[2];
        // If we get a function from frontend, avoid deleting it.
        OpManager::Global()->frontend_funcs.push_back(new PackedFunc(f));
        reg.set_attr(attr_key, f, plevel);
      } else {
        reg.set_attr(attr_key, args[2], plevel);
      }
    }
  });

// helper to get internal dev function in objectref.
struct Op2ObjectPtr : public ObjectRef {
  static ObjectPtr<Object> Get(const Op& op) {
    return GetDataPtr<Object>(op);
  }
};

ObjectPtr<Object> CreateOp(const std::string& name) {
  // Hack use TVMRetValue as exchange
  auto op = Op::Get(name);
  CHECK(op.defined()) << "Cannot find op \'" << name << '\'';
  return Op2ObjectPtr::Get(op);
}

TVM_REGISTER_NODE_TYPE(OpNode)
.set_creator(CreateOp)
.set_repr_bytes([](const Object* n) {
    return static_cast<const OpNode*>(n)->name;
  });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<OpNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const OpNode*>(ref.get());
    p->stream << "Op(" << node->name << ")";
  });

}  // namespace tvm
