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
 * \file src/tir/ir/function.cc
 * \brief The function data structure.
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace tir {

LinkedParam::LinkedParam(int64_t id, ::tvm::runtime::NDArray param) {
  auto n = make_object<LinkedParamNode>();
  n->id = id;
  n->param = param;
  data_ = std::move(n);
}

// Get the function type of a PrimFunc
PrimFunc::PrimFunc(Array<tir::Var> params, Stmt body, Type ret_type,
                   Map<tir::Var, Buffer> buffer_map, DictAttrs attrs, Span span) {
  // Assume void-return type for now
  // TODO(tvm-team) consider type deduction from body.
  if (!ret_type.defined()) {
    ret_type = VoidType();
  }
  auto n = make_object<PrimFuncNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->buffer_map = std::move(buffer_map);
  n->attrs = std::move(attrs);
  n->checked_type_ = n->func_type_annotation();
  n->span = std::move(span);
  data_ = std::move(n);
}

FuncType PrimFuncNode::func_type_annotation() const {
  Array<Type> param_types;
  for (auto param : this->params) {
    param_types.push_back(GetType(param));
  }
  return FuncType(param_types, ret_type, {}, {});
}

TVM_REGISTER_NODE_TYPE(PrimFuncNode);

class TensorIntrinManager {
 public:
  Map<String, tir::TensorIntrin> reg;

  static TensorIntrinManager* Global() {
    static TensorIntrinManager* inst = new TensorIntrinManager();
    return inst;
  }
};

TensorIntrin::TensorIntrin(PrimFunc desc_func, PrimFunc intrin_func) {
  // check the number of func var is equal
  CHECK_EQ(desc_func->params.size(), intrin_func->params.size());
  CHECK_EQ(desc_func->buffer_map.size(), intrin_func->buffer_map.size());

  // check both functions' bodies are directly block
  const auto* desc_realize = Downcast<BlockRealize>(desc_func->body)->block->body.as<BlockRealizeNode>();
  const auto* intrin_realize = Downcast<BlockRealize>(intrin_func->body)->block->body.as<BlockRealizeNode>();
  CHECK(desc_realize != nullptr) << "description function's body expect a directly block";
  CHECK(intrin_realize != nullptr) << "intrinsic function's body expect a directly block";

  const Block& desc_block = desc_realize->block;
  const Block& intrin_block = intrin_realize->block;

  // check block var number and iter type
  CHECK_EQ(desc_block->iter_vars.size(), intrin_block->iter_vars.size())
      << "Two blocks should have the same number of block vars";
  for (size_t i = 0; i < desc_block->iter_vars.size(); i++) {
    const IterVar& desc_var = desc_block->iter_vars[i];
    const IterVar& intrin_var = intrin_block->iter_vars[i];
    CHECK(desc_var->iter_type == intrin_var->iter_type)
        << "Block iter_type mismatch between " << desc_var->iter_type << " and "
        << intrin_var->iter_type;
  }

  auto n = make_object<TensorIntrinNode>();
  n->description = std::move(desc_func);
  n->implementation = std::move(intrin_func);
  data_ = std::move(n);
}

TensorIntrin TensorIntrin::Register(String name, PrimFunc desc_func, PrimFunc intrin_func) {
  TensorIntrinManager* manager = TensorIntrinManager::Global();
  ICHECK_EQ(manager->reg.count(name), 0)
      << "ValueError: TensorIntrin '" << name << "' has already been registered";
  TensorIntrin intrin(desc_func, intrin_func);
  manager->reg.Set(name, intrin);
  return intrin;
}

TensorIntrin TensorIntrin::Get(String name) {
  const TensorIntrinManager* manager = TensorIntrinManager::Global();
  ICHECK_EQ(manager->reg.count(name), 1)
      << "ValueError: TensorIntrin '" << name << "' is not registered";
  return manager->reg.at(name);
}

TVM_REGISTER_NODE_TYPE(TensorIntrinNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimFuncNode>([](const ObjectRef& ref, ReprPrinter* p) {
      // TODO(tvm-team) redirect to Text printer once we have a good text format.
      auto* node = static_cast<const PrimFuncNode*>(ref.get());
      p->stream << "PrimFunc(" << node->params << ") ";
      if (node->attrs.defined()) {
        p->stream << "attrs=" << node->attrs;
      }
      p->stream << " {\n";
      p->indent += 2;
      p->Print(node->body);
      p->indent -= 2;
      p->stream << "}\n";
    });

TVM_REGISTER_GLOBAL("tir.PrimFunc")
    .set_body_typed([](Array<tir::Var> params, Stmt body, Type ret_type,
                       Map<tir::Var, Buffer> buffer_map, DictAttrs attrs, Span span) {
      return PrimFunc(params, body, ret_type, buffer_map, attrs, span);
    });


TVM_REGISTER_GLOBAL("tir.TensorIntrin")
    .set_body_typed([](PrimFunc desc_func, PrimFunc intrin_func) {
      return TensorIntrin(desc_func, intrin_func);
    });

TVM_REGISTER_GLOBAL("tir.TensorIntrinRegister").set_body_typed(TensorIntrin::Register);
TVM_REGISTER_GLOBAL("tir.TensorIntrinGet").set_body_typed(TensorIntrin::Get);

}  // namespace tir
}  // namespace tvm
