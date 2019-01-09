/*!
 *  Copyright (c) 2018 by Contributors
 * \file src/tvm/relay/op.cc
 * \brief Resolve incomplete types to complete types.
 */
#include <tvm/relay/op.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>

#include <memory>
#include <mutex>

namespace dmlc {
// enable registry
DMLC_REGISTRY_ENABLE(::tvm::relay::OpRegistry);
}  // namespace dmlc

namespace tvm {
namespace relay {

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
    static OpManager inst;
    return &inst;
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
  NodePtr<OpNode> n = make_node<OpNode>();
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
  if (p.second < plevel) {
    op_map->data_[index] = std::make_pair(value, plevel);
  }
}

// Frontend APIs
TVM_REGISTER_API("relay.op._ListOpNames")
.set_body_typed<Array<tvm::Expr>()>([]() {
    Array<tvm::Expr> ret;
    for (const std::string& name :
             dmlc::Registry<OpRegistry>::ListAllNames()) {
      ret.push_back(tvm::Expr(name));
    }
    return ret;
  });

TVM_REGISTER_API("relay.op._GetOp").set_body_typed<Op(std::string)>(Op::Get);

TVM_REGISTER_API("relay.op._OpGetAttr")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      Op op = args[0];
      std::string attr_name = args[1];
      auto op_map = Op::GetAttr<TVMRetValue>(attr_name);
      if (op_map.count(op)) {
        *rv = op_map[op];
      }
    });

TVM_REGISTER_API("relay.op._Register")
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
      reg.set_attrs_type_key(value);
    } else {
      // normal attr table override.
      if (args[2].type_code() == kFuncHandle) {
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

NodePtr<Node> CreateOp(const std::string& name) {
  auto op = Op::Get(name);
  CHECK(op.defined()) << "Cannot find op \'" << name << '\'';
  return op.node_;
}

TVM_REGISTER_NODE_TYPE(OpNode)
.set_creator(CreateOp)
.set_global_key([](const Node* n) {
    return static_cast<const OpNode*>(n)->name;
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<OpNode>([](const OpNode* node, tvm::IRPrinter* p) {
    p->stream << "Op(" << node->name << ")";
  });

}  // namespace relay
}  // namespace tvm
