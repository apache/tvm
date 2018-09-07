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

#include "./../pass/type_subst.h"

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
  // get singleton of the
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
  std::shared_ptr<OpNode> n = std::make_shared<OpNode>();
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

void OpRegistry::UpdateAttr(const std::string& key, TVMRetValue value,
                            int plevel) {
  OpManager* mgr = OpManager::Global();
  std::lock_guard<std::mutex> lock(mgr->mutex);
  std::unique_ptr<GenericOpMap>& op_map = mgr->attr[key];
  if (op_map == nullptr) {
    op_map.reset(new GenericOpMap());
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

bool IsGeneric(const FuncType& func_ty) {
  return func_ty->type_params.size() != 0;
}

using namespace runtime;

Module CompileOpsToModule(const std::vector<std::string>& op_names) {
  PackedFunc compile_ops = GetPackedFunc("relay.op._compile_ops");
  tvm::Array<tvm::Array<NodeRef>> args;

  auto compiler_map = Op::GetAttr<PackedFunc>("FRelayOpCompiler");

  for (auto op_name : op_names) {
    Op op = Op::Get(op_name);

    if (!IsGeneric(op->op_type)) {
      auto compiler = compiler_map[op];
      std::cout << "ABOVE CALL" << std::endl;
      tvm::Array<NodeRef> pair = compiler(op->name, op->op_type);
      std::cout << "BELOW CALL" << std::endl;
      // TODO(@jroesch): I can't pass strings across what should be the
      // interface here.
      tvm::Array<NodeRef> triple = {LocalVarNode::make(op->name), pair[0],
                                    pair[1]};
      args.push_back(triple);
    } else {
      throw dmlc::Error("it is impossible to compile generic operators.");
    }
  }

  // Nothing to do, bail out earlier.
  // TVM will complain if we try to generate a module of size 0.
  if (args.size() == 0) {
    return Module(nullptr);
  }

  return compile_ops(args);
}

TVM_REGISTER_API("relay.op._CompileOpsToModule")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      std::vector<std::string> names;
      for (auto i = 0; i < args.num_args; i++) {
        names.push_back(args[i]);
      }
      std::cout << "Right here" << std::endl;
      *ret = CompileOpsToModule(names);
    });

Op SpecializeOp(const std::string& op_name, const std::string& new_op_name,
                Array<Type> type_args) {
  auto registry = ::tvm::relay::OpRegistry::Registry();
  auto op_reg = registry->__REGISTER_OR_GET__(op_name);
  auto new_op_reg = registry->__REGISTER__(new_op_name).set_name();

  auto fn_ty = op_reg.op()->op_type;

  tvm::Map<TypeParam, Type> subst_map;

  CHECK(fn_ty->type_params.size() == type_args.size());

  // Build a subsitituion map up from the function type and type arguments.
  // Eventually allow the type vars to be passed in.
  for (size_t i = 0; i < type_args.size(); i++) {
    subst_map.Set(fn_ty->type_params[i], type_args[i]);
  }

  Type inst_ty = FuncTypeNode::make(fn_ty->arg_types, fn_ty->ret_type, {}, {});
  inst_ty = TypeSubst(fn_ty, subst_map);
  FuncType new_op_ty = GetRef<FuncType>(inst_ty.as<FuncTypeNode>());
  new_op_reg.op()->op_type = new_op_ty;

  // Now we want to copy over some attributes.
  PackedFunc compiler =
      Op::GetAttr<PackedFunc>("FRelayOpCompiler")[op_reg.op()];
  new_op_reg.set_attr<PackedFunc>("FRelayOpCompiler", compiler);

  return new_op_reg.op();
}

TVM_REGISTER_API("relay.op._SpecializeOp")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      *ret = SpecializeOp(args[0], args[1], args[2]);
    });

}  // namespace relay
}  // namespace tvm
