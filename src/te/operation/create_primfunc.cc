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

#include <tvm/runtime/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>

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

/*! \brief Helper data structural to store informations. */
struct CreateFuncInfo {
  /*! \brief The Tensor arg_list. */
  Array<te::Tensor> arg_list;
  /*! \brief The map from each Tensor to its corresponding buffer. */
  std::unordered_map<te::Tensor, Buffer> tensor2buffers;
  /*! \brief The transformer from ProducerLoad to BufferLoad. */
  ProducerToBufferTransformer transformer;
  /*! \brief The buffers should be allocated at function root. */
  Array<Buffer> root_alloc;
  /*! \brief The count map to make block name unique. */
  std::unordered_map<String, int> name_count;

  explicit CreateFuncInfo(Array<te::Tensor> arg_list)
      : arg_list(std::move(arg_list)), transformer(tensor2buffers) {}

  bool IsArg(const te::Tensor& tensor) const {
    return std::any_of(arg_list.begin(), arg_list.end(),
                       [&tensor](const te::Tensor& arg) { return tensor == arg; });
  }

  String GetUniqueName(const String& prefix) {
    String unique_prefix = prefix;
    auto it = name_count.find(prefix);
    while (name_count.count(unique_prefix)) {
      unique_prefix = prefix + "_" + std::to_string(++it->second);
    }
    name_count[unique_prefix] = 0;
    return unique_prefix;
  }
};

BlockRealize GenerateBlockFromTensor(const te::ComputeOp& compute_op, const te::Tensor& tensor,
                                     Array<PrimExpr> bindings, PrimExpr expr_body,
                                     CreateFuncInfo* info) {
  // Step 1. Push_back data_par axis and reduce_axis into block_vars.
  Array<IterVar> iter_vars;
  std::unordered_map<const VarNode*, PrimExpr> var_map;
  iter_vars.reserve(compute_op->axis.size() + compute_op->reduce_axis.size());
  auto f_push_block_vars = [&iter_vars, &var_map](const Array<IterVar>& iters) {
    for (IterVar iter_var : iters) {
      // Create new var
      Var new_var(iter_var->var->name_hint, iter_var->var->dtype);
      var_map[iter_var->var.get()] = new_var;

      IterVarNode* iter_var_node = iter_var.CopyOnWrite();
      iter_var_node->dom = Range::FromMinExtent(iter_var->dom->min, iter_var->dom->extent);
      iter_var_node->var = new_var;
      iter_vars.push_back(iter_var);
    }
  };
  f_push_block_vars(compute_op->axis);
  f_push_block_vars(compute_op->reduce_axis);

  // Step 2. Declare buffer and update op2buffers
  Buffer buffer = decl_buffer(tensor->shape, tensor->dtype, tensor->GetNameHint(), "global");
  info->tensor2buffers[tensor] = buffer;

  // Step 3. Add Buffer to root_alloc
  if (!info->IsArg(tensor)) {
    info->root_alloc.push_back(buffer);
  }

  // Step 4. Calculate indices for BufferStore
  Array<PrimExpr> indices;
  indices.reserve(compute_op->axis.size());
  for (const IterVar& iter_var : compute_op->axis) {
    auto it = var_map.find(iter_var->var.get());
    ICHECK(it != var_map.end());
    indices.push_back(it->second);
  }

  // Step 5. Create block body.
  Optional<Stmt> init = NullOpt;
  Stmt body;
  if (const auto* reduce = expr_body.as<ReduceNode>()) {
    // Case 1. Reduce compute
    ICHECK_EQ(reduce->source.size(), 1);
    const PrimExpr& lhs = BufferLoad(buffer, indices);
    const PrimExpr& rhs = Substitute(info->transformer(reduce->source[0]), var_map);
    ICHECK(lhs->dtype == rhs->dtype);
    body = BufferStore(buffer, reduce->combiner.get()->operator()({lhs}, {rhs})[0], indices);
    init = BufferStore(buffer, reduce->combiner->identity_element[0], indices);
  } else {
    // Case 2. Data parallel compute
    body = BufferStore(buffer, Substitute(info->transformer(expr_body), var_map), indices);
  }

  // Step 6. Add script_parsing_detect_access attr for auto complete the whole IR.
  Map<String, ObjectRef> annotations = compute_op->attrs;
  annotations.Set(tir::attr::script_parsing_detect_access, IntImm(DataType::Int(32), 3));

  // Step 7. Create Block and BlockRealize.
  return BlockRealize(/*iter_values=*/std::move(bindings),
                      /*predicate=*/Bool(true),
                      /*block=*/
                      Block(/*iter_vars=*/std::move(iter_vars),
                            /*reads=*/{},
                            /*writes=*/{},
                            /*name_hint=*/info->GetUniqueName(tensor->GetNameHint()),
                            /*body=*/std::move(body),
                            /*init=*/std::move(init),
                            /*alloc_buffers=*/{},
                            /*match_buffers=*/{},
                            /*annotations=*/std::move(annotations)));
}

Stmt GenerateStmtFromCompute(const te::ComputeOp& compute_op, CreateFuncInfo* info) {
  // Step 1. Creating loop vars for block bindings.
  Array<IterVar> axes = compute_op->axis;
  axes.insert(axes.end(), compute_op->reduce_axis.begin(), compute_op->reduce_axis.end());
  Array<PrimExpr> bindings;
  for (size_t i = 0; i < axes.size(); ++i) {
    bindings.push_back(Var("i" + std::to_string(i)));
  }
  // Step 2. Generate block bodies.
  Array<Stmt> seq_stmt;
  for (int i = 0; i < compute_op->num_outputs(); ++i) {
    const te::Tensor& tensor = compute_op.output(i);
    PrimExpr expr_body = compute_op->body[i];
    seq_stmt.push_back(
        GenerateBlockFromTensor(compute_op, tensor, bindings, std::move(expr_body), info));
  }
  Stmt body = SeqStmt::Flatten(seq_stmt);

  // Step 3. Generate loop nesting.
  for (size_t i = axes.size(); i > 0; --i) {
    const IterVar& axis = axes[i - 1];
    const Var& loop_var = Downcast<Var>(bindings[i - 1]);
    body = For(loop_var, axis->dom->min, axis->dom->extent, ForKind::kSerial, body);
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
                            /*name_hint=*/info->GetUniqueName(extern_op->name),
                            /*body=*/std::move(body),
                            /*init=*/NullOpt,
                            /*alloc_buffers=*/{},
                            /*match_buffers=*/{},
                            /*annotations=*/extern_op->attrs));
}

/*! \brief Use Tensor Expression to create a schedulable TensorIR func. */
PrimFunc CreatePrimFunc(const Array<te::Tensor>& arg_list) {
  // Step 1. Create tensor read graph.
  Array<te::Operation> arg_ops;
  for (const te::Tensor& arg : arg_list) {
    arg_ops.push_back(arg->op);
  }
  te::ReadGraph g = te::CreateReadGraph(arg_ops);
  Array<te::Operation> order = te::PostDFSOrder(arg_ops, g);

  // Step 2. Checking all Operations are supported.
  for (const te::Operation& op : order) {
    if (!(op->IsInstance<te::PlaceholderOpNode>() || op->IsInstance<te::ComputeOpNode>() ||
          op->IsInstance<te::ExternOpNode>()))
      LOG(FATAL) << "TypeError: Unsupported Operation: " << op->GetTypeKey() << ". "
                 << "Only te.placeholder and te.compute are allowed for now.";
  }

  // Infomations used in CreatePrimFunc and its sub-funtions.
  CreateFuncInfo info(arg_list);
  // Root body stmts.
  Array<Stmt> root_stmts;

  // Step 3. Rewrite compute stages into blocks.
  for (const te::Operation& op : order) {
    if (const auto* placeholder = op.as<te::PlaceholderOpNode>()) {
      // Case 1. PlaceholderOp (te.placeholder)
      ICHECK_EQ(op->num_outputs(), 1);
      const te::Tensor& tensor = op.output(0);
      // Check op is in op list
      ICHECK(info.IsArg(tensor));
      const Buffer& buffer =
          decl_buffer(placeholder->shape, placeholder->dtype, placeholder->name, "global");
      info.tensor2buffers[tensor] = buffer;
    } else if (const auto* compute_op = op.as<te::ComputeOpNode>()) {
      // Case 2. ComputeOp (te.compute)
      root_stmts.push_back(GenerateStmtFromCompute(GetRef<te::ComputeOp>(compute_op), &info));
    } else if (const auto extern_op = op.as<te::ExternOpNode>()) {
      // Case 3. ExternOp (te.extern)
      root_stmts.push_back(GenerateStmtFromExternOp(GetRef<te::ExternOp>(extern_op), &info));
    } else {
      ICHECK(false) << "TypeError: Unsupported Operation: " << op->GetTypeKey() << ". "
                    << "Only te.placeholder and te.compute are allowed for now.";
    }
  }

  // Step 4. Create func and complete it.
  Array<Var> parameters;
  Map<Var, Buffer> buffer_map;
  for (const te::Tensor& tensor : arg_list) {
    Var arg("var_" + tensor->GetNameHint(), PrimType(DataType::Handle()));
    parameters.push_back(arg);
    auto it = info.tensor2buffers.find(tensor);
    ICHECK(it != info.tensor2buffers.end());
    buffer_map.Set(arg, it->second);
  }
  PrimFunc func = PrimFunc(/*params=*/std::move(parameters),
                           /*body=*/SeqStmt::Flatten(root_stmts),
                           /*ret_type=*/VoidType(),
                           /*buffer_map=*/std::move(buffer_map));

  const auto* complete = runtime::Registry::Get("script.Complete");
  ICHECK(complete);

  return (*complete)(func, info.root_alloc);
}  // namespace tir

TVM_REGISTER_GLOBAL("te.CreatePrimFunc").set_body_typed([](const Array<te::Tensor>& tensors) {
  return CreatePrimFunc(tensors);
});

}  // namespace tir
}  // namespace tvm
