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

#include "create_primfunc.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ir/name_supply.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../../tir/ir/functor_common.h"
#include "../../tir/transforms/ir_utils.h"
#include "../schedule/graph.h"

namespace tvm {
namespace tir {

/*! \brief The helper mutator that transforms ProducerLoad to BufferLoad */
class ProducerToBufferTransformer : public StmtExprMutator {
 public:
  explicit ProducerToBufferTransformer(const std::unordered_map<te::Tensor, Buffer>& tensor2buffers)
      : tensor2buffers_(tensor2buffers) {}

  PrimExpr VisitExpr_(const ProducerLoadNode* op) final {
    auto visited_op = Downcast<ProducerLoad>(StmtExprMutator::VisitExpr_(op));
    te::Tensor tensor = Downcast<te::Tensor>(visited_op->producer);
    auto it = tensor2buffers_.find(tensor);
    ICHECK(it != tensor2buffers_.end()) << "IndexError: Cannot find the tensor " << tensor;
    const Buffer& buffer = it->second;
    return BufferLoad(buffer, visited_op->indices);
  }

 private:
  /*! \brief The Map from Operations to buffers */
  const std::unordered_map<te::Tensor, Buffer>& tensor2buffers_;
};

/*! \brief Helper data structure to store information. */
struct CreateFuncInfo {
  /*! \brief The Tensor arg_list. */
  Array<te::Tensor> arg_list;
  /*! \brief The map from each Tensor to its corresponding buffer. */
  std::unordered_map<te::Tensor, Buffer> tensor2buffers;
  /*! \brief The transformer from ProducerLoad to BufferLoad. */
  ProducerToBufferTransformer transformer;
  /*! \brief The buffers should be allocated at function root. */
  Array<Buffer> root_alloc;
  /*! \brief The NameSupply to make block name unique. */
  NameSupply name_supply = NameSupply("");

  String FreshName(String base_name) { return name_supply->FreshName(base_name); }

  explicit CreateFuncInfo(Array<te::Tensor> arg_list)
      : arg_list(std::move(arg_list)), transformer(tensor2buffers) {}

  bool IsArg(const te::Tensor& tensor) const {
    return std::any_of(arg_list.begin(), arg_list.end(),
                       [&tensor](const te::Tensor& arg) { return tensor == arg; });
  }
};

class LayoutFreePlaceholdersNormalizer : public StmtMutator {
 public:
  PrimFunc Process(PrimFunc func) {
    for (int i = 0, n = func->params.size(); i < n; ++i) {
      if (const auto* v = func->params[i].as<VarNode>()) {
        if (Optional<Buffer> buffer = func->buffer_map.Get(GetRef<Var>(v))) {
          buffer2index_[buffer.value()] = i;
        }
      }
    }
    PrimFuncNode* f = func.CopyOnWrite();
    f->body = VisitStmt(std::move(f->body));
    if (this->layout_free_buffer_indices_.empty()) {
      return func;
    }
    Array<Integer> indices;
    indices.reserve(this->layout_free_buffer_indices_.size());
    for (int i : this->layout_free_buffer_indices_) {
      indices.push_back(Integer(i));
    }
    return WithAttr(std::move(func), tir::attr::layout_free_buffers, indices);
  }

  Stmt VisitStmt_(const BlockNode* _block) final {
    Block block = Downcast<Block>(StmtMutator::VisitStmt_(_block));
    if (Optional<ObjectRef> ann = block->annotations.Get(topi_attr)) {
      Array<Buffer> new_buffers;
      for (Buffer buffer : Downcast<Array<Buffer>>(ann)) {
        auto it = buffer2index_.find(buffer);
        if (it != buffer2index_.end()) {
          layout_free_buffer_indices_.insert(it->second);
        } else {
          new_buffers.push_back(buffer);
        }
      }
      block.CopyOnWrite()->annotations.Set(topi_attr, new_buffers);
    }
    return std::move(block);
  }

  std::unordered_map<tir::Buffer, int, ObjectPtrHash, ObjectPtrEqual> buffer2index_;
  std::set<int> layout_free_buffer_indices_;
  String topi_attr = "layout_free_placeholders";
};

BlockRealize GenerateBlockFromTensors(const te::ComputeOp& compute_op,
                                      const Array<te::Tensor>& tensors, Array<PrimExpr> bindings,
                                      PrimExpr expr_body, CreateFuncInfo* info,
                                      arith::Analyzer* analyzer) {
  // Step 1. Push_back data_par axis and reduce_axis into block_vars.
  Array<IterVar> iter_vars;
  std::unordered_map<const VarNode*, PrimExpr> var_map;
  iter_vars.reserve(compute_op->axis.size() + compute_op->reduce_axis.size());
  auto f_push_block_vars = [&iter_vars, &var_map, &analyzer](const Array<IterVar>& iters) {
    for (IterVar iter_var : iters) {
      // Create new var
      Var new_var(iter_var->var->name_hint, iter_var->var->dtype);
      var_map[iter_var->var.get()] = new_var;

      const PrimExpr& dom_min = analyzer->Simplify(iter_var->dom->min);
      const PrimExpr& dom_extent = analyzer->Simplify(iter_var->dom->extent);
      iter_vars.push_back(IterVar(Range::FromMinExtent(dom_min, dom_extent), new_var,
                                  iter_var->iter_type, iter_var->thread_tag, iter_var->span));
    }
  };
  f_push_block_vars(compute_op->axis);
  f_push_block_vars(compute_op->reduce_axis);

  // Step 2.
  //  - Declare buffers
  //  - Update `op2buffers`
  //  - Add the non-argument tensors to `alloc_buffer` of the root block
  Array<Buffer> buffers;
  for (const te::Tensor& tensor : tensors) {
    Buffer buffer = decl_buffer(tensor->shape, tensor->dtype, tensor->GetNameHint(), "global");
    info->tensor2buffers[tensor] = buffer;
    buffers.push_back(buffer);

    if (!info->IsArg(tensor)) {
      info->root_alloc.push_back(info->tensor2buffers[tensor]);
    }
  }

  // Step 3. Calculate indices for BufferStore
  Array<PrimExpr> indices;
  indices.reserve(compute_op->axis.size());
  for (const IterVar& iter_var : compute_op->axis) {
    auto it = var_map.find(iter_var->var.get());
    ICHECK(it != var_map.end());
    indices.push_back(it->second);
  }

  // Step 4. Create block body.
  String block_name{nullptr};
  Optional<Stmt> init = NullOpt;
  Stmt body;
  if (const auto* reduce = expr_body.as<ReduceNode>()) {
    // Case 1. Reduce compute
    block_name = info->FreshName(compute_op->name);
    int n_buffers = buffers.size();

    Array<PrimExpr> lhs;
    Array<PrimExpr> rhs;
    lhs.reserve(n_buffers);
    rhs.reserve(n_buffers);

    // Make the LHS operands and RHS operands:
    //  - A LHS operand is the buffer storing the reduction result, with corresponding indices.
    //  - A RHS operand is the value to be reduced.
    for (int i = 0; i < n_buffers; ++i) {
      const PrimExpr& left = BufferLoad(buffers[i], indices);
      const PrimExpr& right =
          analyzer->Simplify(Substitute(info->transformer(reduce->source[i]), var_map));
      lhs.push_back(left);
      rhs.push_back(right);
      ICHECK_EQ(left->dtype, right->dtype);
    }

    Array<Var> temp_vars;
    Array<Stmt> body_stmts;
    Array<Stmt> init_stmts;
    temp_vars.reserve(n_buffers);
    body_stmts.reserve(n_buffers);
    init_stmts.reserve(n_buffers);

    // - When there is only one buffer, we directly create a BufferStore which stores "combiner(lhs,
    //   rhs)" into the target buffer position.
    // - In case there are multiple buffers, to avoid incorrect results, we create some intermediate
    //   variables and use LetStmts to bind the variables with "combiner(lhs, rhs)". After that, we
    //   then store the value of the variables into the target buffer positions.
    for (int i = 0; i < n_buffers; ++i) {
      const Buffer& buffer = buffers[i];
      init_stmts.push_back(BufferStore(buffer, reduce->combiner->identity_element[i], indices));
      PrimExpr value{nullptr};
      if (n_buffers > 1) {
        temp_vars.push_back(Var("v_" + buffer->name, PrimType(lhs[i].dtype())));
        value = temp_vars.back();
      } else {
        value = reduce->combiner.get()->operator()(lhs, rhs)[i];
      }
      body_stmts.push_back(BufferStore(buffer, value, indices));
    }

    init = SeqStmt::Flatten(init_stmts);
    body = SeqStmt::Flatten(body_stmts);
    if (n_buffers > 1) {
      // When there are multiple buffers, we wrap the body with LetStmts.
      for (int i = n_buffers - 1; i >= 0; --i) {
        PrimExpr value = reduce->combiner.get()->operator()(lhs, rhs)[i];
        body = LetStmt(temp_vars[i], std::move(value), std::move(body));
      }
    }
  } else {
    // Case 2. Data parallel compute
    ICHECK_EQ(tensors.size(), 1);
    block_name = info->FreshName(tensors[0]->GetNameHint());
    const PrimExpr& compute_body = Substitute(info->transformer(expr_body), var_map);
    body = BufferStore(info->tensor2buffers[tensors[0]], analyzer->Simplify(compute_body), indices);
  }

  // Step 5. Add script_parsing_detect_access attr for auto complete the whole IR.
  Map<String, ObjectRef> annotations;
  auto mutate_attr = [&info](const ObjectRef& value) -> ObjectRef {
    if (const auto* tensor_value = value.as<te::TensorNode>()) {
      return info->tensor2buffers.at(GetRef<te::Tensor>(tensor_value));
    } else {
      return value;
    }
  };

  for (const auto& pair : compute_op->attrs) {
    const String& key = pair.first;
    const ObjectRef& value = pair.second;
    // TensorIR will not allow Tensor data structure
    if (value->IsInstance<ArrayNode>()) {
      const auto array_value = Downcast<Array<ObjectRef>>(value);
      annotations.Set(key, array_value.Map(mutate_attr));
    } else {
      annotations.Set(key, mutate_attr(value));
    }
  }
  // Set script_parsing_detect_access
  annotations.Set(tir::attr::script_parsing_detect_access, IntImm(DataType::Int(32), 3));
  if (iter_vars.empty()) {
    IterVar iter(Range::FromMinExtent(0, 1), Var("vi", DataType::Int(32)), IterVarType::kDataPar);
    PrimExpr binding(0);
    iter_vars.push_back(iter);
    bindings.push_back(binding);
  }

  // Step 6. Create Block and BlockRealize.
  return BlockRealize(/*iter_values=*/std::move(bindings),
                      /*predicate=*/Bool(true),
                      /*block=*/
                      Block(/*iter_vars=*/std::move(iter_vars),
                            /*reads=*/{},
                            /*writes=*/{},
                            /*name_hint=*/block_name,
                            /*body=*/std::move(body),
                            /*init=*/std::move(init),
                            /*alloc_buffers=*/{},
                            /*match_buffers=*/{},
                            /*annotations=*/std::move(annotations)));
}

Stmt GenerateStmtFromCompute(const te::ComputeOp& compute_op, CreateFuncInfo* info,
                             arith::Analyzer* analyzer) {
  // Step 1. Creating loop vars for block bindings.
  Array<IterVar> axes = compute_op->axis;
  axes.insert(axes.end(), compute_op->reduce_axis.begin(), compute_op->reduce_axis.end());
  Array<PrimExpr> bindings;
  for (size_t i = 0; i < axes.size(); ++i) {
    const IterVar& axis = axes[i];
    int bits = std::max(axis->dom->min.dtype().bits(), axis->dom->extent.dtype().bits());
    bindings.push_back(Var("i" + std::to_string(i), runtime::DataType::Int(bits)));
  }
  // Step 2. Generate block bodies.
  Array<Stmt> seq_stmt;
  if (compute_op->body[0]->IsInstance<ReduceNode>()) {
    auto f_reducer_equal = [](const ReduceNode* a, const ReduceNode* b) -> bool {
      return a->combiner.same_as(b->combiner) &&    //
             a->source.same_as(b->source) &&        //
             a->axis.same_as(b->axis) &&            //
             a->condition.same_as(b->condition) &&  //
             ((a->init.empty() && b->init.empty()) || a->init.same_as(b->init));
    };

    PrimExpr expr_body = compute_op->body[0];
    Array<te::Tensor> tensors = {compute_op.output(0)};
    const tir::ReduceNode* reduce = expr_body.as<tir::ReduceNode>();
    // specially handle reduction inline for multiplre reductions.
    for (size_t k = 1; k < compute_op->body.size(); ++k) {
      const tir::ReduceNode* reduce_ = compute_op->body[k].as<tir::ReduceNode>();
      ICHECK(reduce_);
      ICHECK(f_reducer_equal(reduce_, reduce))
          << "The Reduce inputs of ComputeOp should have the same attribute except value_index";
      tensors.push_back(compute_op.output(k));
    }

    seq_stmt.push_back(GenerateBlockFromTensors(compute_op, tensors, bindings, std::move(expr_body),
                                                info, analyzer));
  } else {
    for (int i = 0; i < compute_op->num_outputs(); ++i) {
      const te::Tensor& tensor = compute_op.output(i);
      PrimExpr expr_body = compute_op->body[i];
      seq_stmt.push_back(GenerateBlockFromTensors(compute_op, {tensor}, bindings,
                                                  std::move(expr_body), info, analyzer));
    }
  }

  Stmt body = SeqStmt::Flatten(seq_stmt);

  // Step 3. Generate loop nesting.
  for (size_t i = axes.size(); i > 0; --i) {
    const IterVar& axis = axes[i - 1];
    PrimExpr dom_min = analyzer->Simplify(axis->dom->min);
    PrimExpr dom_extent = analyzer->Simplify(axis->dom->extent);
    const Var& loop_var = Downcast<Var>(bindings[i - 1]);
    body = For(loop_var, dom_min, dom_extent, ForKind::kSerial, body);
  }

  return body;
}

Stmt GenerateStmtFromExternOp(const te::ExternOp& extern_op, CreateFuncInfo* info) {
  // Step 1. Check all inputs are visited before and update var_map.
  std::unordered_map<const VarNode*, PrimExpr> var_map;
  ICHECK_EQ(extern_op->inputs.size(), extern_op->input_placeholders.size());
  for (size_t i = 0; i < extern_op->inputs.size(); ++i) {
    const Buffer& placeholder = extern_op->input_placeholders[i];
    const te::Tensor& input_tensor = extern_op->inputs[i];
    auto it = info->tensor2buffers.find(input_tensor);
    ICHECK(it != info->tensor2buffers.end());
    var_map[placeholder->data.get()] = it->second->data;
  }

  // Step 2. Update info with its output tensor and placeholder buffer.
  ICHECK_EQ(extern_op->num_outputs(), extern_op->output_placeholders.size());
  for (int i = 0; i < extern_op->num_outputs(); ++i) {
    const Buffer& placeholder = extern_op->output_placeholders[i];
    const te::Tensor& output_tensor = extern_op.output(i);
    info->tensor2buffers[output_tensor] = placeholder;
    if (!info->IsArg(output_tensor)) {
      info->root_alloc.push_back(placeholder);
    }
  }

  // Step 3. Collect Access Region
  Array<BufferRegion> reads, writes;
  for (const te::Tensor& tensor : extern_op->inputs) {
    // We have ICHECK before so it is not needed here.
    reads.push_back(BufferRegion::FullRegion(info->tensor2buffers[tensor]));
  }
  for (const Buffer& buffer : extern_op->output_placeholders) {
    writes.push_back(BufferRegion::FullRegion(buffer));
  }

  Stmt body = Substitute(extern_op->body, var_map);

  // Step 4. Generate opaque block as body.
  return BlockRealize(/*iter_values=*/{},
                      /*predicate=*/Bool(true),
                      /*block=*/
                      Block(/*iter_vars=*/{},
                            /*reads=*/std::move(reads),
                            /*writes=*/std::move(writes),
                            /*name_hint=*/info->FreshName(extern_op->name),
                            /*body=*/std::move(body),
                            /*init=*/NullOpt,
                            /*alloc_buffers=*/{},
                            /*match_buffers=*/{},
                            /*annotations=*/extern_op->attrs));
}

Array<te::Operation> CollectOrderedOps(const Array<te::Tensor>& arg_list) {
  Array<te::Operation> arg_ops;
  for (const te::Tensor& arg : arg_list) {
    arg_ops.push_back(arg->op);
  }
  te::ReadGraph g = te::CreateReadGraph(arg_ops);
  Array<te::Operation> order = te::PostDFSOrder(arg_ops, g);

  for (const te::Operation& op : order) {
    if (!(op->IsInstance<te::PlaceholderOpNode>() || op->IsInstance<te::ComputeOpNode>() ||
          op->IsInstance<te::ExternOpNode>()))
      LOG(FATAL) << "TypeError: Unsupported Operation: " << op->GetTypeKey() << ". "
                 << "Only te.placeholder and te.compute are allowed for now.";
  }
  return order;
}

void InitializeBufferBinds(const Array<te::Operation>& ordered_ops, CreateFuncInfo* info) {
  // Process any TE operations which contain user defined buffers
  for (const auto& op : ordered_ops) {
    // Initialize the tensor2buffer binds map with buffers defined by the te.extern
    if (const auto* extern_op = op.as<te::ExternOpNode>()) {
      ICHECK_EQ(extern_op->inputs.size(), extern_op->input_placeholders.size());
      for (size_t i = 0; i < extern_op->inputs.size(); ++i) {
        const te::Tensor& input = extern_op->inputs[i];
        const Buffer& buffer = extern_op->input_placeholders[i];
        info->tensor2buffers[input] = buffer;
      }
    }
  }
}

void RewriteStageToBlock(const te::Operation& op, CreateFuncInfo* info, Array<Stmt>* root_stmts,
                         arith::Analyzer* analyzer) {
  if (const auto* placeholder = op.as<te::PlaceholderOpNode>()) {
    // Case 1. PlaceholderOp (te.placeholder)
    ICHECK_EQ(op->num_outputs(), 1);
    const te::Tensor& tensor = op.output(0);
    // Check op is in op list
    ICHECK(info->IsArg(tensor));
    // Declare a buffer for any argument tensors without a pre-existing
    // buffer declaration recorded in the tensor2buffer binds map
    if (info->tensor2buffers.count(tensor) == 0) {
      const Buffer& buffer =
          decl_buffer(placeholder->shape, placeholder->dtype, placeholder->name, "global");
      info->tensor2buffers[tensor] = buffer;
    }
  } else if (const auto* compute_op = op.as<te::ComputeOpNode>()) {
    // Case 2. ComputeOp (te.compute)
    root_stmts->push_back(
        GenerateStmtFromCompute(GetRef<te::ComputeOp>(compute_op), info, analyzer));
  } else if (const auto extern_op = op.as<te::ExternOpNode>()) {
    // Case 3. ExternOp (te.extern)
    root_stmts->push_back(GenerateStmtFromExternOp(GetRef<te::ExternOp>(extern_op), info));
  } else {
    ICHECK(false) << "TypeError: Unsupported Operation: " << op->GetTypeKey() << ". "
                  << "Only te.placeholder and te.compute are allowed for now.";
  }
}

PrimFunc GenerateAndCompletePrimFunc(const Array<te::Tensor>& arg_list,
                                     const Array<Stmt>& root_stmts, CreateFuncInfo* info) {
  Array<Var> parameters;
  Map<Var, Buffer> buffer_map;
  for (const te::Tensor& tensor : arg_list) {
    Var arg("var_" + tensor->GetNameHint(), PrimType(DataType::Handle()));
    parameters.push_back(arg);
    auto it = info->tensor2buffers.find(tensor);
    ICHECK(it != info->tensor2buffers.end());
    buffer_map.Set(arg, it->second);
  }
  PrimFunc func = WithAttrs(PrimFunc(/*params=*/std::move(parameters),
                                     /*body=*/SeqStmt::Flatten(root_stmts),
                                     /*ret_type=*/VoidType(),
                                     /*buffer_map=*/std::move(buffer_map)),
                            {{"global_symbol", String("main")}, {"tir.noalias", Bool(true)}});
  const auto* complete = runtime::Registry::Get("script.Complete");
  ICHECK(complete);
  func = (*complete)(std::move(func), info->root_alloc);
  return func;
}

PrimFunc CreatePrimFuncWithConstants(const Array<te::Tensor>& arg_list,
                                     const Array<runtime::NDArray>& constants) {
  // Infomations used in CreatePrimFunc and its sub-functions.
  CreateFuncInfo info(arg_list);
  // Root body stmts.
  Array<Stmt> root_stmts;
  // Analyzer
  arith::Analyzer analyzer;

  // Step 1. Create ordered array of operations and validate they are supported.
  Array<te::Operation> order = CollectOrderedOps(arg_list);

  // Step 2. Initialize buffer binds map
  InitializeBufferBinds(order, &info);

  // Step 3. Rewrite compute stages into blocks.
  for (const te::Operation& op : order) {
    RewriteStageToBlock(op, &info, &root_stmts, &analyzer);
  }

  // Step 4. Create func and complete prim func.
  auto func = GenerateAndCompletePrimFunc(arg_list, root_stmts, &info);
  func = tir::BindParams(func, constants);
  return LayoutFreePlaceholdersNormalizer().Process(std::move(func));
}

PrimFunc CreatePrimFunc(const Array<te::Tensor>& arg_list) {
  return CreatePrimFuncWithConstants(arg_list, {});
}

TVM_REGISTER_GLOBAL("te.CreatePrimFunc").set_body_typed(CreatePrimFunc);

}  // namespace tir
}  // namespace tvm
