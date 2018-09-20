/*!
 *  Copyright (c) 2018 by Contributors
 * \file api_registry.cc
 */
#include <tvm/api_registry.h>

namespace tvm {

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<EnvFuncNode>([](const EnvFuncNode *op, IRPrinter *p) {
    p->stream << "EnvFunc(" << op->name << ")";
});

NodePtr<EnvFuncNode> CreateEnvNode(const std::string& name) {
  auto* f = runtime::Registry::Get(name);
  CHECK(f != nullptr) << "Cannot find global function \'" << name << '\'';
  NodePtr<EnvFuncNode> n = make_node<EnvFuncNode>();
  n->func = *f;
  n->name = name;
  return n;
}

EnvFunc EnvFunc::Get(const std::string& name) {
  return EnvFunc(CreateEnvNode(name));
}

TVM_REGISTER_API("_EnvFuncGet")
.set_body_typed<EnvFunc(const std::string& name)>(EnvFunc::Get);

TVM_REGISTER_API("_EnvFuncCall")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    EnvFunc env = args[0];
    CHECK_GE(args.size(), 1);
    env->func.CallPacked(TVMArgs(args.values + 1,
                                 args.type_codes + 1,
                                 args.size() - 1), rv);
  });

TVM_REGISTER_API("_EnvFuncGetPackedFunc")
.set_body_typed<PackedFunc(const EnvFunc& n)>([](const EnvFunc&n) {
    return n->func;
  });

TVM_REGISTER_NODE_TYPE(EnvFuncNode)
.set_creator(CreateEnvNode)
.set_global_key([](const Node* n) {
    return static_cast<const EnvFuncNode*>(n)->name;
  });

}  // namespace tvm
