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
 * \file extract_constants.cc
 * \brief Collects PrimFunc's constant data into mod's 'tvm::attr::kConstantsArray' attrs array,
 * sets irmod_storage_idx as index in this array.
 * For more information, see the RFC:
 * https://github.com/apache/tvm-rfcs/blob/main/rfcs/0022-tir-non-scalar-constants.md
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

using ConstArrayType = Array<runtime::NDArray>;
class Applicator : public tir::StmtMutator {
 protected:
  // returns index of the a in constant_array_, if not found - appends
  size_t DeDup(const runtime::NDArray& a) {
    tvm::SEqualReducer eql;
    auto it = std::find_if(
        constant_array_.begin(), constant_array_.end(), [&eql, a](const runtime::NDArray& v) {
          return NDArrayContainerTrait::SEqualReduce(a.as<runtime::NDArray::Container>(),
                                                     v.as<runtime::NDArray::Container>(), eql);
        });
    if (it != constant_array_.end()) {
      return it - constant_array_.begin();
    }
    constant_array_.push_back(std::move(a));
    return constant_array_.size() - 1;
  }

 public:
  Stmt Apply(tir::Stmt body, const ConstArrayType& constant_array) {
    constant_array_ = constant_array;
    return this->VisitStmt(body);
  }

  Stmt VisitStmt_(const tir::AllocateConstNode* acn) override {
    // Check whether the data already defined within the module's attrs
    // and add array index.
    ICHECK(acn->data) << "data field should be defined";
    auto node = CopyOnWrite(acn);
    node->irmod_storage_idx = Optional<Integer>(Integer(DeDup(node->data.value())));
    return Stmt(node);
  }

  ConstArrayType constant_array_;
};

namespace transform {

tvm::transform::Pass ExtractPrimFuncConstants() {
  auto prim_func_pass = [=](PrimFunc foo, IRModule m, tvm::transform::PassContext ctx) {
    auto* func = foo.CopyOnWrite();
    if (!m->attrs.defined()) {
      m->attrs = DictAttrs(Map<String, ObjectRef>());
    }
    auto* attrs = m->attrs.CopyOnWrite();
    ConstArrayType constant_array_ =
        (attrs->dict.count(tvm::attr::kConstants))
            ? Downcast<ConstArrayType>(attrs->dict[tvm::attr::kConstants])
            : ConstArrayType();
    Applicator a = Applicator();
    func->body = a.Apply(func->body, constant_array_);
    const ConstArrayType constant_list = a.constant_array_;
    if (constant_list.size()) {
      attrs->dict.Set(tvm::attr::kConstants, constant_list);
    }
    return GetRef<PrimFunc>(func);
  };

  auto pass_func = [=](IRModule module, tvm::transform::PassContext pc) {
    auto m = GetRef<IRModule>(module.CopyOnWrite());
    for (const auto& kv : m->functions) {
      if (auto func = kv.second.as<PrimFunc>()) {
        m->Update(kv.first, prim_func_pass(func.value(), m, pc));
      }
    }
    return m;
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tir.ExtractPrimFuncConstants", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ExtractPrimFuncConstants")
    .set_body_typed(ExtractPrimFuncConstants);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
