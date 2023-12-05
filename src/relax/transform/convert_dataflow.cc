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
 * \file tvm/relax/transform/convert_dataflow.cc
 * \brief Pass for extracting groups of pure operations without
 *   dataflow into dataflow blocks.
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/utils.h>

namespace tvm {
namespace relax {

class DataflowBlockExtractor : public ExprMutator {
 public:
  explicit DataflowBlockExtractor(size_t min_size) : ExprMutator(), min_size_(min_size) {}

  Expr VisitExpr_(const SeqExprNode* seq) override {
    Array<BindingBlock> new_blocks;
    Expr new_body = VisitExpr(seq->body);
    bool changed = !new_body.same_as(seq->body);
    bool dataflow_streak = false;
    Array<Binding> dataflow_bindings;
    Array<Binding> non_dataflow_bindings;

    for (auto block : seq->blocks) {
      BindingBlock new_block = this->VisitBindingBlock(block);
      changed = changed || !new_block.same_as(block);

      // For an existing dataflow block, we add to the current streak
      // or start a new streak in case there will be more dataflow operations
      // coming up
      if (new_block.as<DataflowBlock>()) {
        if (!dataflow_streak) {
          dataflow_streak = true;
        }
        dataflow_bindings.insert(dataflow_bindings.end(), new_block->bindings.begin(),
                                 new_block->bindings.end());
        continue;
      }

      // for a binding block, attempt to extract dataflow blocks inside
      auto binding_block = Downcast<BindingBlock>(new_block);
      for (size_t i = 0; i < binding_block->bindings.size(); i++) {
        auto binding = binding_block->bindings[i];
        Expr value = GetBoundValue(binding);
        // dataflow values: not an if node and not an impure call
        bool is_dataflow = (!value.as<IfNode>()) &&
                           (!(value.as<CallNode>() && IsImpureCall(Downcast<Call>(value))));
        if (!dataflow_streak) {
          // we can start a dataflow streak
          if (is_dataflow) {
            dataflow_streak = true;
            dataflow_bindings = {binding};
          } else {
            non_dataflow_bindings.push_back(binding);
          }
        } else {
          if (is_dataflow) {
            // extend the streak
            dataflow_bindings.push_back(binding);
          } else {
            // this is the end of the streak
            dataflow_streak = false;

            // if the df block is below the minimum length, combine the blocks
            // and reset the dataflow collection
            if (dataflow_bindings.size() < min_size_) {
              non_dataflow_bindings.insert(non_dataflow_bindings.end(), dataflow_bindings.begin(),
                                           dataflow_bindings.end());
              dataflow_bindings = {};
            } else {
              // otherwise insert both collections
              changed = true;
              new_blocks.push_back(BindingBlock(non_dataflow_bindings));
              new_blocks.push_back(DataflowBlock(dataflow_bindings));
              non_dataflow_bindings = {};
              dataflow_bindings = {};
            }
            non_dataflow_bindings.push_back(binding);
          }
        }
      }
    }

    // handle any remaining bindings
    if (dataflow_bindings.size() < min_size_) {
      non_dataflow_bindings.insert(non_dataflow_bindings.end(), dataflow_bindings.begin(),
                                   dataflow_bindings.end());
      new_blocks.push_back(BindingBlock(non_dataflow_bindings));
    } else {
      changed = true;
      new_blocks.push_back(BindingBlock(non_dataflow_bindings));
      new_blocks.push_back(DataflowBlock(dataflow_bindings));
    }

    if (!changed) {
      return GetRef<SeqExpr>(seq);
    }
    return SeqExpr(new_blocks, new_body);
  }

 private:
  size_t min_size_;
};

Expr ConvertToDataflow(const Expr& input, size_t min_size) {
  DataflowBlockExtractor extractor(min_size);
  return extractor.VisitExpr(input);
}

namespace transform {

Pass ConvertToDataflow(int min_size) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(ConvertToDataflow(f, min_size));
      };
  auto pass = CreateFunctionPass(pass_func, 0, "ConvertToDataflow", {});
  // Canonicalize bindings is included afterwards in order to transform any
  // normal vars in DF blocks that are not used outside the DF block into
  // dataflow vars. This allows us to avoid reimplementing that functionality.
  return tvm::transform::Sequential({pass, CanonicalizeBindings()});
}

TVM_REGISTER_GLOBAL("relax.transform.ConvertToDataflow").set_body_typed(ConvertToDataflow);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
