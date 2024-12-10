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
#include <tvm/tir/analysis.h>
#include <tvm/tir/data_type_rewriter.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../support/array.h"
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

/*! \brief The helper mutator to rewrite buffer and buffer var accessed by block body */
class BufferSubstituter : public StmtExprMutator {
 public:
  explicit BufferSubstituter(const std::unordered_map<const VarNode*, PrimExpr>& var_map,
                             const std::unordered_map<const BufferNode*, Buffer>& buffer_map)
      : var_map_(var_map), buffer_map_(buffer_map) {}

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_map_.find(op);
    if (it != var_map_.end()) {
      return it->second;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = buffer_map_.find(load->buffer.get());
    if (it != buffer_map_.end()) {
      return BufferLoad(it->second, load->indices, load->predicate, load->span);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = buffer_map_.find(store->buffer.get());
    if (it != buffer_map_.end()) {
      return BufferStore(it->second, store->value, store->indices, store->predicate, store->span);
    }
    return store;
  }

 private:
  const std::unordered_map<const VarNode*, PrimExpr>& var_map_;
  const std::unordered_map<const BufferNode*, Buffer>& buffer_map_;
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
  NameSupply name_supply;

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
      if (auto v = func->params[i].as<Var>()) {
        if (Optional<Buffer> buffer = func->buffer_map.Get(v.value())) {
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
    BlockNode* n = block.CopyOnWrite();
    if (Optional<ObjectRef> ann = n->annotations.Get(topi_attr)) {
      Array<Buffer> new_buffers;
      for (Buffer buffer : Downcast<Array<Buffer>>(ann)) {
        auto it = buffer2index_.find(buffer);
        if (it != buffer2index_.end()) {
          layout_free_buffer_indices_.insert(it->second);
        } else {
          new_buffers.push_back(buffer);
        }
      }
      if (new_buffers.empty()) {
        n->annotations.erase(topi_attr);
      } else {
        n->annotations.Set(topi_attr, new_buffers);
      }
    }
    for (const String& attr : this->blocklist) {
      auto it = n->annotations.find(attr);
      if (it != n->annotations.end()) {
        n->annotations.erase(attr);
      }
    }
    return std::move(block);
  }

  std::unordered_map<tir::Buffer, int, ObjectPtrHash, ObjectPtrEqual> buffer2index_;
  std::set<int> layout_free_buffer_indices_;
  String topi_attr = "layout_free_placeholders";
  std::vector<String> blocklist = {"const_matrix", "auto_scheduler_simplify_const_tensor_indices",
                                   "workload"};
};

/**!
 * \brief The iter levels specify nested structure wrt iteration domain dependencies.
 * (1) Each iter should reside in exactly one level.
 * (2) The domain of low level iter should be either free or ony depend on iters in high level.
 **/
using NestedIterLevels = std::vector<std::vector<IterVar>>;

NestedIterLevels GenerateNestedIterLevels(const Array<IterVar>& axes, arith::Analyzer* analyzer) {
  int global_max_depth = 0;
  std::unordered_map<Var, int> depth;
  std::unordered_map<Var, IterVar> var2iter;
  for (const auto& axis : axes) {
    var2iter[axis->var] = axis;
  }

  std::function<int(const IterVar&)> traverse = [&](const IterVar& axis) -> int {
    auto depth_it = depth.find(axis->var);
    if (depth_it != depth.end()) {  // cache
      return depth_it->second;
    }
    std::vector<Var> dep_vars;
    for (const Var& v : UndefinedVars(analyzer->Simplify(axis->dom->min))) {
      dep_vars.push_back(v);
    }
    for (const Var& v : UndefinedVars(analyzer->Simplify(axis->dom->extent))) {
      dep_vars.push_back(v);
    }
    int cur_depth = 0;
    for (const Var& v : dep_vars) {
      auto it = var2iter.find(v);
      if (it == var2iter.end()) {
        // not axis var dependency, maybe a symbolic shape var or others.
        continue;
      }
      int depth = traverse(it->second);
      cur_depth = std::max(cur_depth, depth + 1);
    }
    depth.emplace_hint(depth_it, axis->var, cur_depth);
    global_max_depth = std::max(global_max_depth, cur_depth);
    return cur_depth;
  };

  for (const auto& axis : axes) {
    traverse(axis);
  }
  NestedIterLevels levels;
  levels.resize(global_max_depth + 1);
  for (const auto& axis : axes) {
    const Var& var = axis->var;
    levels[depth[var]].push_back(axis);
  }
  return levels;
}

/*!
 * \brief Generate output buffers from compute op's output tensors, and bind to context func info.
 * \param compute_op The target compute op.
 * \param info Generation context info.
 * \returns The output buffer objects, ordered by compute op's outputs.
 **/
Array<Buffer> GenerateOutputBuffers(const te::ComputeOp& compute_op, CreateFuncInfo* info) {
  // Step 1. Collect output tensors in TE operation.
  Array<te::Tensor> tensors;
  if (compute_op->body[0]->IsInstance<ReduceNode>()) {
    auto f_reducer_equal = [](const ReduceNode* a, const ReduceNode* b) -> bool {
      StructuralEqual eq;
      return eq(a->combiner, b->combiner) &&    //
             eq(a->source, b->source) &&        //
             eq(a->axis, b->axis) &&            //
             eq(a->condition, b->condition) &&  //
             eq(a->init, b->init);
    };
    PrimExpr expr_body = compute_op->body[0];
    tensors.push_back(compute_op.output(0));
    const tir::ReduceNode* reduce = expr_body.as<tir::ReduceNode>();
    // specially handle reduction inline for multiplre reductions.
    for (size_t k = 1; k < compute_op->body.size(); ++k) {
      const tir::ReduceNode* reduce_ = compute_op->body[k].as<tir::ReduceNode>();
      ICHECK(reduce_);
      ICHECK(f_reducer_equal(reduce_, reduce))
          << "The Reduce inputs of ComputeOp should have the same attribute except value_index, "
          << "but the first argument has body " << GetRef<PrimExpr>(reduce_) << ", while the " << k
          << "-th argument has body " << GetRef<PrimExpr>(reduce);
      tensors.push_back(compute_op.output(k));
    }
  } else {
    for (size_t k = 0; k < compute_op->body.size(); ++k) {
      tensors.push_back(compute_op.output(k));
    }
  }
  // Step 2. Prepare buffers for compute outputs
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
  return buffers;
}

/*!
 * \brief Generate block annotation dict from compute op attrs.
 * \param compute_op The target compute op.
 * \param info Generation context info.
 * \returns The block annotation dict.
 **/
Map<String, ObjectRef> GenerateBlockAnnotations(const te::ComputeOp& compute_op,
                                                CreateFuncInfo* info) {
  Map<String, ObjectRef> annotations;
  auto mutate_attr = [&info](const ObjectRef& value) -> ObjectRef {
    if (auto tensor_value = value.as<te::Tensor>()) {
      return info->tensor2buffers.at(tensor_value.value());
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
  return annotations;
}

/*!
 * \brief Generate init stmt for reduction.
 * \param indices Target store indices for the block.
 * \param buffers Target store buffers for the block.
 * \param reduce Reduce description node.
 * \param var_map Var re-mapping for TE compute axes.
 * \param info Generation context info.
 * \returns Init stmt.
 **/
Stmt GenerateInitStmt(const Array<PrimExpr>& indices, const Array<Buffer>& buffers,
                      const ReduceNode* reduce, const Map<Var, PrimExpr>& var_map,
                      CreateFuncInfo* info) {
  // helper to transform the expr and remap iters to the block domain
  auto f_transform_and_remap = [&](const PrimExpr& e) {
    return Substitute(info->transformer(e), var_map);
  };
  Optional<Stmt> init = NullOpt;
  Stmt body;
  int n_buffers = buffers.size();
  Array<Stmt> init_stmts;
  init_stmts.reserve(n_buffers);
  for (int i = 0; i < n_buffers; ++i) {
    const Buffer& buffer = buffers[i];
    PrimExpr identity = f_transform_and_remap(reduce->combiner->identity_element[i]);
    init_stmts.push_back(BufferStore(buffer, identity, indices));
  }
  return SeqStmt::Flatten(init_stmts);
}

/*!
 * \brief Generate body execution stmt.
 * \param indices Target store indices for the block.
 * \param buffers Target store buffers for the block.
 * \param var_map Var re-mapping for TE compute axes.
 * \param expr_body Target computation expression.
 * \param info Generation context info.
 * \param analyzer Arithmetic analyzer in context.
 * \returns Init stmt.
 **/
Stmt GenerateBodyStmt(const Array<PrimExpr>& indices, const Array<Buffer>& buffers,
                      const Map<Var, PrimExpr>& var_map, PrimExpr expr_body, CreateFuncInfo* info,
                      arith::Analyzer* analyzer) {
  // helper to transform the expr and remap iters to the block domain
  auto f_transform_and_remap = [&](const PrimExpr& e) {
    return Substitute(info->transformer(e), var_map);
  };
  Stmt body;
  if (const auto* reduce = expr_body.as<ReduceNode>()) {
    // Case 1. Reduce compute
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
      const PrimExpr& right = analyzer->Simplify(f_transform_and_remap(reduce->source[i]));
      lhs.push_back(left);
      rhs.push_back(right);
      ICHECK_EQ(left->dtype, right->dtype);
    }

    Array<Var> temp_vars;
    Array<Stmt> body_stmts;
    temp_vars.reserve(n_buffers);
    body_stmts.reserve(n_buffers);

    // - When there is only one buffer, we directly create a BufferStore which stores "combiner(lhs,
    //   rhs)" into the target buffer position.
    // - In case there are multiple buffers, to avoid incorrect results, we create some intermediate
    //   variables and use LetStmts to bind the variables with "combiner(lhs, rhs)". After that, we
    //   then store the value of the variables into the target buffer positions.
    for (int i = 0; i < n_buffers; ++i) {
      const Buffer& buffer = buffers[i];
      PrimExpr value{nullptr};
      if (n_buffers > 1) {
        temp_vars.push_back(Var("v_" + buffer->name, PrimType(lhs[i].dtype())));
        value = temp_vars.back();
      } else {
        PrimExpr combined = reduce->combiner.get()->operator()(lhs, rhs)[i];
        value = f_transform_and_remap(combined);
      }
      body_stmts.push_back(BufferStore(buffer, value, indices));
    }
    body = SeqStmt::Flatten(body_stmts);
    if (n_buffers > 1) {
      // When there are multiple buffers, we wrap the body with LetStmts.
      for (int i = n_buffers - 1; i >= 0; --i) {
        PrimExpr value = f_transform_and_remap(reduce->combiner.get()->operator()(lhs, rhs)[i]);
        body = LetStmt(temp_vars[i], std::move(value), std::move(body));
      }
    }
  } else {
    // Case 2. Data parallel compute
    ICHECK_EQ(buffers.size(), 1);
    const PrimExpr& compute_body = f_transform_and_remap(expr_body);
    body = BufferStore(buffers[0], analyzer->Simplify(compute_body), indices);
  }
  return std::move(body);
}

/*! \brief Record loops, block vars and binding in the single level scope. */
struct NestedScopeInfo {
  // loop var and range in the scope.
  std::vector<std::pair<Var, Range>> loop_vars;
  // block iters for current level's block.
  Array<IterVar> block_iters;
  // block bindings for current level's block.
  Array<PrimExpr> bindings;
  // store indices for current level's block.
  Array<PrimExpr> store_indices;
  // mapping from original TE compute axes to new block vars.
  Map<Var, PrimExpr> axes_remap;

  // helper to add new block var
  void AddBlockIter(const Optional<IterVar>& origin_axis, const IterVar& iter,
                    const PrimExpr& value) {
    block_iters.push_back(iter);
    bindings.push_back(value);
    if (origin_axis.defined()) {
      if (iter->iter_type != IterVarType::kCommReduce) {
        store_indices.push_back(iter->var);
      }
      axes_remap.Set(origin_axis.value()->var, iter->var);
    }
  }

  // helper to renew leaf block var defs to ensure SSA.
  void Renew(const Array<IterVar>& origin_axes) {
    block_iters.MutateByApply([](const IterVar& itervar) {
      auto n = make_object<IterVarNode>(*itervar.get());
      n->var = n->var.copy_with_suffix("");
      return IterVar(n);
    });
    for (size_t i = 0; i < origin_axes.size(); ++i) {
      Var block_var = block_iters[i]->var;
      if (origin_axes[i]->iter_type != IterVarType::kCommReduce) {
        store_indices.Set(i, block_var);
      }
      axes_remap.Set(origin_axes[i]->var, block_var);
    }
  }
};

Stmt GenerateStmtFromCompute(const te::ComputeOp& compute_op, CreateFuncInfo* info,
                             arith::Analyzer* analyzer) {
  // Step 1. Collect all iter axes in original TE compute op
  Array<IterVar> axes = compute_op->axis;
  axes.insert(axes.end(), compute_op->reduce_axis.begin(), compute_op->reduce_axis.end());

  // Step 2. Prepare nested iteration scopes.
  // For each axis, we generate loop and the first block binding at the level it belongs to.
  // In lower levels, we just create new block var and bind it to the previous level block var.
  auto axes_levels = GenerateNestedIterLevels(axes, analyzer);
  ICHECK(!axes_levels.empty());
  std::vector<NestedScopeInfo> scopes;
  scopes.reserve(axes_levels.size());
  std::unordered_set<Var> defined_axes;
  for (size_t i = 0; i < axes_levels.size(); ++i) {
    NestedScopeInfo cur_scope;
    for (size_t j = 0; j < axes.size(); ++j) {
      const IterVar& axis = axes[j];
      DataType index_type =
          DataType::Int(std::max(axis->dom->min.dtype().bits(), axis->dom->extent.dtype().bits()));
      bool first_times_define =
          std::find(axes_levels[i].begin(), axes_levels[i].end(), axis) != axes_levels[i].end();
      if (first_times_define) {
        Var loop_var = Var(axis->var->name_hint, index_type);
        Var block_var("v_" + axis->var->name_hint, index_type);
        PrimExpr min = axis->dom->min;
        PrimExpr extent = axis->dom->extent;
        if (i > 0) {
          const auto& scope_repl = scopes[i - 1].axes_remap;
          min = Substitute(min, scope_repl);
          extent = Substitute(extent, scope_repl);
        }
        Range dom = Range::FromMinExtent(analyzer->Simplify(min), analyzer->Simplify(extent));
        IterVar new_block_iter(dom, block_var, axis->iter_type, axis->thread_tag, axis->span);
        cur_scope.loop_vars.emplace_back(loop_var, dom);
        cur_scope.AddBlockIter(axis, new_block_iter, loop_var);
        defined_axes.insert(axis->var);
      } else if (defined_axes.count(axis->var)) {
        ICHECK_GT(i, 0);
        ICHECK(scopes[i - 1].axes_remap.count(axis->var));
        PrimExpr prev_binding = scopes[i - 1].axes_remap.at(axis->var);
        Var block_var("v_" + axis->var->name_hint, index_type);
        Range dom = Range::FromMinExtent(prev_binding, make_const(index_type, 1));
        IterVar new_block_iter(dom, block_var, axis->iter_type, axis->thread_tag, axis->span);
        cur_scope.AddBlockIter(axis, new_block_iter, prev_binding);
      }
    }
    if (i == axes_levels.size() - 1 && cur_scope.block_iters.empty()) {
      // for the leaf scope, we ensure at least one block var exists
      IterVar dummy(Range::FromMinExtent(0, 1), Var("vi", DataType::Int(32)),
                    IterVarType::kDataPar);
      cur_scope.AddBlockIter(NullOpt, dummy, 0);
    }
    scopes.push_back(cur_scope);
  }

  // Step 3. Generate output buffers for each output tensor
  Array<Buffer> buffers = GenerateOutputBuffers(compute_op, info);

  // Step 4. Generate leaf block stmts.
  Array<Stmt> seq_stmt;
  auto leaf = scopes.back();
  Map<String, ObjectRef> annotations = GenerateBlockAnnotations(compute_op, info);
  const ReduceNode* reduce = compute_op->body[0].as<ReduceNode>();
  if (reduce) {
    PrimExpr expr_body = compute_op->body[0];
    Stmt init = GenerateInitStmt(leaf.store_indices, buffers, reduce, leaf.axes_remap, info);
    Stmt body =
        GenerateBodyStmt(leaf.store_indices, buffers, leaf.axes_remap, expr_body, info, analyzer);
    seq_stmt.push_back(BlockRealize(/*iter_values=*/leaf.bindings,
                                    /*predicate=*/Bool(true),
                                    /*block=*/
                                    Block(/*iter_vars=*/leaf.block_iters,
                                          /*reads=*/{},
                                          /*writes=*/{},
                                          /*name_hint=*/info->FreshName(compute_op->name),
                                          /*body=*/body,
                                          /*init=*/init,
                                          /*alloc_buffers=*/{},
                                          /*match_buffers=*/{},
                                          /*annotations=*/annotations)));

  } else {
    for (int i = 0; i < compute_op->num_outputs(); ++i) {
      if (i > 0) {
        // Renew block var defs to ensure SSA
        leaf.Renew(axes);
      }
      PrimExpr expr_body = compute_op->body[i];
      Stmt body = GenerateBodyStmt(leaf.store_indices, {buffers[i]}, leaf.axes_remap, expr_body,
                                   info, analyzer);
      seq_stmt.push_back(BlockRealize(/*iter_values=*/leaf.bindings,
                                      /*predicate=*/Bool(true),
                                      /*block=*/
                                      Block(/*iter_vars=*/leaf.block_iters,
                                            /*reads=*/{},
                                            /*writes=*/{},
                                            /*name_hint=*/info->FreshName(buffers[i]->name),
                                            /*body=*/body,
                                            /*init=*/NullOpt,
                                            /*alloc_buffers=*/{},
                                            /*match_buffers=*/{},
                                            /*annotations=*/annotations)));
    }
  }
  Stmt body = SeqStmt::Flatten(seq_stmt);

  // Step 4. Generate nested parent scopes.
  for (size_t i = scopes.size(); i > 0; --i) {
    const auto& cur = scopes[i - 1];
    if (i < scopes.size()) {
      auto block_name = info->FreshName(compute_op->name + "_l" + std::to_string(i));
      const auto& block_iters = cur.block_iters;

      Optional<Stmt> init{NullOpt};
      if (reduce && std::any_of(block_iters.begin(), block_iters.end(), [](const IterVar& iter) {
            return iter->iter_type == IterVarType::kCommReduce;
          })) {
        // if the reduce axis defined in non-leaf scopes, the nested block is also
        // a reduction block, thus we should also insert init stmt in the parent level.
        init = GenerateInitStmt(cur.store_indices, buffers, reduce, cur.axes_remap, info);
      }

      // wrap nested block
      body = BlockRealize(/*iter_values=*/cur.bindings,
                          /*predicate=*/Bool(true),
                          /*block=*/
                          Block(/*iter_vars=*/block_iters,
                                /*reads=*/{},
                                /*writes=*/{},
                                /*name_hint=*/block_name,
                                /*body=*/body,
                                /*init=*/init,
                                /*alloc_buffers=*/{},
                                /*match_buffers=*/{},
                                /*annotations=*/annotations));
    }
    for (size_t j = cur.loop_vars.size(); j > 0; --j) {
      const auto& [loop_var, dom] = cur.loop_vars[j - 1];
      body = For(loop_var, dom->min, dom->extent, ForKind::kSerial, body);
    }
  }
  return body;
}

Stmt GenerateStmtFromExternOp(const te::ExternOp& extern_op, CreateFuncInfo* info) {
  // Step 1. Check all inputs are visited before and update var_map.
  std::unordered_map<const VarNode*, PrimExpr> var_map;
  std::unordered_map<const BufferNode*, Buffer> input_buffer_map;
  ICHECK_EQ(extern_op->inputs.size(), extern_op->input_placeholders.size());
  for (size_t i = 0; i < extern_op->inputs.size(); ++i) {
    const Buffer& placeholder = extern_op->input_placeholders[i];
    const te::Tensor& input_tensor = extern_op->inputs[i];
    auto it = info->tensor2buffers.find(input_tensor);
    ICHECK(it != info->tensor2buffers.end());
    var_map[placeholder->data.get()] = it->second->data;
    input_buffer_map[placeholder.get()] = it->second;
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

  // The access region does not need to be collected here, as it will
  // be generated with the later application of "script.Complete" in
  // GenerateAndCompletePrimFunc.  Waiting until later also handles
  // the case where there is only a single BlockNode, which then
  // becomes the root Block of the function, and should not have
  // reads/writes filled in.

  BufferSubstituter substituter(var_map, input_buffer_map);
  Stmt body = substituter(extern_op->body);

  // Step 4. Generate opaque block as body.
  return BlockRealize(/*iter_values=*/{},
                      /*predicate=*/Bool(true),
                      /*block=*/
                      Block(/*iter_vars=*/{},
                            /*reads=*/{},
                            /*writes=*/{},
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
    ICHECK(info->IsArg(tensor)) << "The operation " << op << " produces tensor " << tensor
                                << ", but this tensor does not appear as a function argument.  "
                                << "The function accepts arguments " << info->arg_list;
    // Declare a buffer for any argument tensors without a pre-existing
    // buffer declaration recorded in the tensor2buffer binds map
    if (info->tensor2buffers.count(tensor) == 0) {
      const Buffer& buffer =
          decl_buffer(placeholder->shape, placeholder->dtype, placeholder->name, "global");
      info->tensor2buffers[tensor] = buffer;
    }
  } else if (auto compute_op = op.as<te::ComputeOp>()) {
    // Case 2. ComputeOp (te.compute)
    root_stmts->push_back(GenerateStmtFromCompute(compute_op.value(), info, analyzer));
  } else if (const auto extern_op = op.as<te::ExternOp>()) {
    // Case 3. ExternOp (te.extern)
    root_stmts->push_back(GenerateStmtFromExternOp(extern_op.value(), info));
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
                                     const Array<runtime::NDArray>& constants,
                                     std::optional<DataType> index_dtype_override) {
  // Information used in CreatePrimFunc and its sub-functions.
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
  if (index_dtype_override.has_value()) {
    func = IndexDataTypeNormalizer(index_dtype_override.value()).Rewrite(std::move(func));
  }
  auto result = LayoutFreePlaceholdersNormalizer().Process(std::move(func));
  return result;
}

PrimFunc CreatePrimFunc(const Array<te::Tensor>& arg_list,
                        std::optional<DataType> index_dtype_override) {
  return CreatePrimFuncWithConstants(arg_list, {}, index_dtype_override);
}

TVM_REGISTER_GLOBAL("te.CreatePrimFunc").set_body([](TVMArgs args, TVMRetValue* ret) {
  Array<ObjectRef> arg_list = args[0];
  std::optional<DataType> index_dtype_override{std::nullopt};
  // Add conversion to make std::optional compatible with FFI.
  if (args[1].type_code() != kTVMNullptr) {
    index_dtype_override = args[1].operator DataType();
  }
  *ret = CreatePrimFunc(arg_list, index_dtype_override);
});

// Relax version impl
PrimFunc GenerateAndCompletePrimFunc(const Array<ObjectRef>& arg_tir_var_list,
                                     const Array<Stmt>& root_stmts, CreateFuncInfo* info) {
  Array<Var> parameters;
  Map<Var, Buffer> buffer_map;
  for (const ObjectRef& arg : arg_tir_var_list) {
    if (auto opt_tensor = arg.as<te::Tensor>()) {
      te::Tensor tensor = opt_tensor.value();
      Var arg("var_" + tensor->GetNameHint(), PrimType(DataType::Handle()));
      parameters.push_back(arg);
      auto it = info->tensor2buffers.find(tensor);
      ICHECK(it != info->tensor2buffers.end());
      buffer_map.Set(arg, it->second);
    } else if (auto var = arg.as<tir::Var>()) {
      parameters.push_back(var.value());
    }
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

PrimFunc CreatePrimFuncWithConstants(const Array<ObjectRef>& arg_list,
                                     const Array<runtime::NDArray>& constants,
                                     std::optional<DataType> index_dtype_override) {
  Array<te::Tensor> tensor_arg_list;
  for (const ObjectRef& x : arg_list) {
    if (auto tensor_node = x.as<te::TensorNode>()) {
      te::Tensor tensor = GetRef<te::Tensor>(tensor_node);
      tensor_arg_list.push_back(tensor);
    }
  }
  // Infomations used in CreatePrimFunc and its sub-functions.
  CreateFuncInfo info(tensor_arg_list);
  // Root body stmts.
  Array<Stmt> root_stmts;
  // Analyzer
  arith::Analyzer analyzer;

  // Step 1. Create ordered array of operations and validate they are supported.
  Array<te::Operation> order = CollectOrderedOps(tensor_arg_list);

  // Step 2. Initialize buffer binds map
  InitializeBufferBinds(order, &info);

  // Step 3. Rewrite compute stages into blocks.
  for (const te::Operation& op : order) {
    RewriteStageToBlock(op, &info, &root_stmts, &analyzer);
  }
  auto func = GenerateAndCompletePrimFunc(arg_list, root_stmts, &info);
  func = tir::BindParams(func, constants);
  if (index_dtype_override.has_value()) {
    func = IndexDataTypeNormalizer(index_dtype_override.value()).Rewrite(std::move(func));
  }
  auto result = LayoutFreePlaceholdersNormalizer().Process(std::move(func));
  return result;
}

PrimFunc CreatePrimFunc(const Array<ObjectRef>& arg_list,
                        std::optional<DataType> index_dtype_override) {
  return CreatePrimFuncWithConstants(arg_list, {}, index_dtype_override);
}

}  // namespace tir
}  // namespace tvm
