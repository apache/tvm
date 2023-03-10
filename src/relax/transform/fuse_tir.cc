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
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/stmt_functor.h>

#include "../../relay/analysis/graph_partitioner.h"
#include "../../support/arena.h"
#include "../../tir/ir/functor_common.h"

namespace tvm {
namespace tir {

// TODO(Siyuan): move it to somewhere under tir folder
/*!
 * \brief Substitute a given source buffer with a given target buffer in statements or expressions.
 */
class FuseTIRBufferSubstitor : private StmtExprMutator {
 public:
  static Stmt Substitute(const Map<Buffer, Buffer>& buffer_map, Stmt stmt) {
    return FuseTIRBufferSubstitor(buffer_map)(std::move(stmt));
  }

 private:
  explicit FuseTIRBufferSubstitor(const Map<Buffer, Buffer>& buffer_map) {
    for (const auto& kv : buffer_map) {
      const Buffer& src = kv.first;
      const Buffer& tgt = kv.second;
      buffer_var_map_[src->data.get()] = tgt;
    }
  }

  PrimExpr VisitExpr_(const VarNode* _op) final {
    auto it = buffer_var_map_.find(_op);
    if (it != buffer_var_map_.end()) {
      return it->second->data;
    } else {
      return GetRef<PrimExpr>(_op);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* _op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_op));
    auto it = buffer_var_map_.find(load->buffer->data.get());
    if (it != buffer_var_map_.end()) {
      auto n = make_object<BufferLoadNode>(*load.get());
      n->buffer = it->second;
      return BufferLoad(n);
    } else {
      return std::move(load);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* _op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_op));
    auto it = buffer_var_map_.find(store->buffer->data.get());
    if (it != buffer_var_map_.end()) {
      auto n = CopyOnWrite(store.get());
      n->buffer = it->second;
      return BufferStore(n);
    } else {
      return std::move(store);
    }
  }

  PrimExpr VisitExpr_(const LoadNode* _op) final {
    Load load = Downcast<Load>(StmtExprMutator::VisitExpr_(_op));
    auto it = buffer_var_map_.find(load->buffer_var.get());
    if (it != buffer_var_map_.end()) {
      auto n = make_object<LoadNode>(*load.get());
      n->buffer_var = it->second->data;
      return Load(n);
    } else {
      return std::move(load);
    }
  }

  Stmt VisitStmt_(const StoreNode* _op) final {
    Store store = Downcast<Store>(StmtExprMutator::VisitStmt_(_op));
    auto it = buffer_var_map_.find(store->buffer_var.get());
    if (it != buffer_var_map_.end()) {
      auto n = CopyOnWrite(store.get());
      n->buffer_var = it->second->data;
      return Store(n);
    } else {
      return std::move(store);
    }
  }

  Stmt VisitStmt_(const BlockNode* _op) final {
    Block block = Downcast<Block>(StmtMutator::VisitStmt_(_op));

    // Define the mutation functions.
    auto f_mutate_match_buffers = [this](const MatchBufferRegion& match_buffer) {
      const Buffer& src_buffer = match_buffer->source->buffer;
      auto it = buffer_var_map_.find(src_buffer->data.get());
      if (it != buffer_var_map_.end()) {
        return MatchBufferRegion(match_buffer->buffer,
                                 BufferRegion(it->second, match_buffer->source->region));
      } else {
        return match_buffer;
      }
    };

    auto f_mutate_read_write_region = [this](const BufferRegion& buffer_region) {
      auto it = buffer_var_map_.find(buffer_region->buffer->data.get());
      return it == buffer_var_map_.end() ? buffer_region
                                         : BufferRegion(it->second, buffer_region->region);
    };

    // Step 1. Mutate `match_buffers`.
    Array<MatchBufferRegion> match_buffers =
        MutateArray(block->match_buffers, f_mutate_match_buffers);
    // Step 2. Mutate the read/write region.
    Array<BufferRegion> reads = MutateArray(block->reads, f_mutate_read_write_region);
    Array<BufferRegion> writes = MutateArray(block->writes, f_mutate_read_write_region);

    reads = UnionAccessRegion(reads);
    writes = UnionAccessRegion(writes);

    if (reads.same_as(block->reads) &&    //
        writes.same_as(block->writes) &&  //
        match_buffers.same_as(block->match_buffers)) {
      return std::move(block);
    } else {
      auto n = CopyOnWrite(block.get());
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      n->match_buffers = std::move(match_buffers);
      return Block(n);
    }
  }

 private:
  /*! \brief Mapping from src buffer.data to tgt buffer. */
  std::unordered_map<const tir::VarNode*, tir::Buffer> buffer_var_map_;
  /*! \brief The structural equality checker */
  StructuralEqual structural_equal_;

  Array<tir::BufferRegion> UnionAccessRegion(const Array<BufferRegion>& regions) const {
    // For now we only allow Buffer access the same elements.
    // e.g. `[A[vi, vj], A[vi, vj]]` is a legal pattern but need to union to `A[vi, vj]`
    // However, `A[vi, vj], A[vi, vj + 1]` is not allow for now.
    // Note: the order of return region should remain the same as the first occurance of the region
    Array<BufferRegion> ret;
    std::unordered_map<const BufferNode*, Region> buffer_region_set;

    for (const BufferRegion& region : regions) {
      auto it = buffer_region_set.find(region->buffer.get());
      if (it == buffer_region_set.end()) {
        ret.push_back(region);
        buffer_region_set[region->buffer.get()] = region->region;
      } else {
        ICHECK(structural_equal_(region->region, it->second));
      }
    }

    if (ret.size() == regions.size()) {
      return regions;
    } else {
      return ret;
    }
  }
};

/*! \brief A mutator which detect block name duplication and deduplicate the names. */
class BlockNameDeduplicator : public tir::StmtMutator {
 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = Downcast<Block>(tir::StmtMutator::VisitStmt_(op));

    String name = GetUniqueName(block->name_hint);

    if (name == block->name_hint) {
      return std::move(block);
    } else {
      ObjectPtr<BlockNode> n = CopyOnWrite(block.get());
      n->name_hint = std::move(name);
      return Stmt(n);
    }
  }

  String GetUniqueName(const String& prefix) {
    String unique_prefix = prefix;
    auto it = name_count_.find(prefix);
    while (name_count_.count(unique_prefix)) {
      unique_prefix = prefix + "_" + std::to_string(++it->second);
    }
    name_count_[unique_prefix] = 0;
    return unique_prefix;
  }

  // TODO(relax-team): It should detects the number suffix and do renaming properly
  // e.g. GetUniqueName("name1") should return "name2" instead of "name10".
  /*! \brief The count map to make block name unique. */
  std::unordered_map<String, int> name_count_;
};

}  // namespace tir

namespace relax {

class FusedTIRConstructor : public ExprVisitor {
 public:
  /*!
   * \brief Construct a fused TIR PrimFunc from a relax sub-function
   * \param mod The IRModule
   * \param gv The global var of relax subfunction to be fused into one PrimFunc
   * \return The fused TIR PrimFunc
   */
  static tir::PrimFunc GetFusedTIR(const IRModule& mod, const GlobalVar& gv) {
    FusedTIRConstructor visitor(mod, gv->name_hint);
    BaseFunc f = mod->Lookup(gv);
    CHECK(f->IsInstance<relax::FunctionNode>())
        << "Expected relax functions, but got: " << f->GetTypeKey();
    CHECK(f->HasNonzeroAttr(relax::attr::kPrimitive))
        << "Expected a function with attr `kPrimitive`";
    visitor(Downcast<relax::Function>(f));
    return visitor.fused_tir_;
  }

 private:
  explicit FusedTIRConstructor(const IRModule& mod, const String& func_name)
      : mod_(mod), func_name_(func_name) {}

  void VisitExpr_(const FunctionNode* func) final {
    // Step 1. Create buffers for function params
    for (const Var& relax_param : func->params) {
      auto ret = CreateParamsAndBuffers(GetStructInfo(relax_param),  //
                                        relax_param->name_hint());
      const Array<tir::Var>& params = ret.first;
      const Array<tir::Buffer>& buffers = ret.second;
      ICHECK_EQ(params.size(), buffers.size());
      for (size_t i = 0; i < params.size(); ++i) {
        func_info_.buffer_map.Set(params[i], buffers[i]);
        func_info_.params.push_back(params[i]);
      }
      func_info_.expr2buffers.Set(relax_param, buffers);
    }

    // Step 2. Visit Function body and create intermediate buffers
    ExprVisitor::VisitExpr_(func);

    // Step 3. Create and remap buffers for function output
    ICHECK(func->body->IsInstance<SeqExprNode>())
        << "Function body is expected to be a SeqExpr, but got: " << func->body->GetTypeKey();
    Expr body = Downcast<SeqExpr>(func->body)->body;
    auto it = func_info_.expr2buffers.find(body);
    ICHECK(it != func_info_.expr2buffers.end())
        << "Fail to detect output buffers for function body";
    const Array<tir::Buffer>& buffers = (*it).second;
    for (size_t i = 0; i < buffers.size(); ++i) {
      tir::Var param = tir::Var("p_output" + std::to_string(i), PrimType(DataType::Handle()));
      func_info_.buffer_map.Set(param, buffers[i]);
      func_info_.params.push_back(param);
      func_info_.output_buffers.insert(buffers[i].get());
    }

    // Step 4. Create PrimFunc
    fused_tir_ = ConstructFunc();
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    // Update expr2buffers by visiting values.
    this->VisitExpr(binding->value);
    auto it = func_info_.expr2buffers.find(binding->value);
    if (it != func_info_.expr2buffers.end()) {
      // assign binding var to the buffers of the value
      func_info_.expr2buffers.Set(binding->var, (*it).second);
    } else {
      LOG(FATAL) << "Unsupported binding value: " << binding->value;
    }
  }

  void VisitBinding_(const MatchCastNode* match_cast) final {
    LOG(FATAL) << "MatchCast is unsupported in primitive functions";
  }

  void VisitExpr_(const CallNode* call) final {
    ExprVisitor::VisitExpr_(call);
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    ICHECK(call->op == call_tir_op_)
        << "Only call_tir is supported in primitive function, but got: " << GetRef<Expr>(call);

    // Step 1. Get Global var and PrimFunc
    GlobalVar gv = Downcast<GlobalVar>(call->args[0]);
    tir::PrimFunc prim_func_ = Downcast<tir::PrimFunc>(mod_->Lookup(gv));

    // Step 2. Renew all vars/buffer definitions and blocks to avoid duplication
    tir::PrimFunc prim_func = tir::RenewDefs(prim_func_);

    // Step 3. Check functions are all schedulable funcs. i.e. the body of func is root block
    // TODO(Siyuan): support un-schedulable functions.
    ICHECK(prim_func->body->IsInstance<tir::BlockRealizeNode>())
        << "Only schedulable functions (whose body is the root block) can be fused";
    const tir::BlockRealize& root_realize = Downcast<tir::BlockRealize>(prim_func->body);
    const tir::Block& root_block = root_realize->block;

    // Step 4. Add all the original alloc_buffers and body to the fused function.
    func_info_.alloc_buffers.insert(func_info_.alloc_buffers.end(),
                                    root_block->alloc_buffers.begin(),
                                    root_block->alloc_buffers.end());
    func_info_.bodies.push_back(root_block->body);

    // Step 5. Map input arguments to buffer
    MapInputBuffer(prim_func, call->args[1]);
    size_t num_output_buffers = GetCallTIROutputSize(call);
    AllocateIntermediateBuffer(GetRef<Expr>(call), prim_func, num_output_buffers);
    // Update fused func name
    func_info_.global_name += "_" + gv->name_hint;
  }

  void VisitExpr_(const TupleGetItemNode* tuple_get_item) final {
    ExprVisitor::VisitExpr_(tuple_get_item);
    auto it = func_info_.expr2buffers.find(tuple_get_item->tuple);
    if (it != func_info_.expr2buffers.end()) {
      int begin_buf_idx = 0;
      int end_buf_idx = 0;
      const TupleType& tuple_type = Downcast<TupleType>(tuple_get_item->tuple->checked_type());
      for (int i = 0; i < tuple_get_item->index; ++i) {
        begin_buf_idx += GetTotalTensorSize(tuple_type->fields[i]);
      }
      end_buf_idx = begin_buf_idx + GetTotalTensorSize(tuple_type->fields[tuple_get_item->index]);
      func_info_.expr2buffers.Set(
          GetRef<Expr>(tuple_get_item),
          {(*it).second.begin() + begin_buf_idx, (*it).second.begin() + end_buf_idx});
    }
  }

  void VisitExpr_(const TupleNode* tuple) final {
    ExprVisitor::VisitExpr_(tuple);
    Array<tir::Buffer> buffers;
    for (const Expr& expr : tuple->fields) {
      auto it = func_info_.expr2buffers.find(expr);
      if (it != func_info_.expr2buffers.end()) {
        buffers.insert(buffers.end(), (*it).second.begin(), (*it).second.end());
      }
    }
    if (!buffers.empty()) {
      func_info_.expr2buffers.Set(GetRef<Expr>(tuple), buffers);
    }
  }

  void VisitExpr_(const ConstantNode* op) final {
    LOG(FATAL) << "Relax.Constant is not supported in primitive functions.";
  }

  /*!
   * \brief Get the number of outputs for a call_tir node.
   * \return The number of outputs.
   */
  static size_t GetCallTIROutputSize(const CallNode* call) {
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    ICHECK(call->op.same_as(call_tir_op_));
    ICHECK_EQ(call->sinfo_args.size(), 1);
    if (const auto* tuple_sinfo = call->sinfo_args[0].as<TupleStructInfoNode>()) {
      return tuple_sinfo->fields.size();
    } else {
      return 1;
    }
  }

  /*! \brief Map old TIR func param buffer to new buffer, and then update `buffer_subst_map` */
  void MapArgsToBuffer(const Array<Expr> args, const Array<tir::Buffer>& buffers) {
    size_t buffer_idx = 0;
    for (const Expr& arg : args) {
      if (const auto* v = arg.as<VarNode>()) {
        auto it = func_info_.expr2buffers.find(GetRef<Var>(v));
        // Substitute the buffer with the already allocated one if it is an intermediate var
        if (it != func_info_.expr2buffers.end()) {
          for (const tir::Buffer& target_buffer : (*it).second) {
            ICHECK_LT(buffer_idx, buffers.size());
            const tir::Buffer& buffer = buffers[buffer_idx];
            // TODO(relax-team): Add support for symbolic shape fusion
            for (const PrimExpr& shape_expr : buffer->shape) {
              ICHECK(shape_expr.as<IntImmNode>()) << "Only support constant shape fusion for now";
            }
            func_info_.buffer_subst_map.Set(buffer, target_buffer);
            buffer_idx++;
          }
        }
      }
    }
    // Make sure every buffers are maped.
    ICHECK_EQ(buffer_idx, buffers.size());
  }

  /*!
   * \brief Update buffer mapping `func_info_.buffer_subst_map` for input args
   * \param func The old TIR PrimFunc
   * \param output_size The number of output params. All output params are at the end of param list.
   */
  void MapInputBuffer(const tir::PrimFunc& func, const relax::Expr& args) {
    Array<Expr> arg_list;
    Array<tir::Buffer> buffer_list;
    if (const auto* arg_tuple = args.as<TupleNode>()) {
      arg_list = arg_tuple->fields;
    } else {
      arg_list = {args};
    }

    ICHECK_GE(func->params.size(), arg_list.size());
    for (size_t i = 0; i < arg_list.size(); ++i) {
      const tir::Var& param = func->params[i];
      const tir::Buffer& buffer = func->buffer_map.at(param);
      buffer_list.push_back(buffer);
    }

    MapArgsToBuffer(arg_list, buffer_list);
  }

  /*!
   * \brief Allocate buffer(s) and update `func_info.expr2buffers` if the PrimFunc output(s) are
   * intermediate results.
   * \param expr The relax Expr, which can be binding vars or binding values.
   * \param func The old TIR PrimFunc
   * \param output_size The number of output params. All output params are at the end of param list.
   */
  void AllocateIntermediateBuffer(const Expr& expr, const tir::PrimFunc& func, size_t output_size) {
    size_t n = func->params.size();
    ICHECK_GE(n, output_size);
    // Allocate intermediate buffer
    Array<tir::Buffer> alloc_buffers;
    for (size_t i = 0; i < output_size; ++i) {
      const tir::Var& param = func->params[n - output_size + i];
      const tir::Buffer& buffer = func->buffer_map.at(param);
      func_info_.alloc_buffers.push_back(buffer);
      alloc_buffers.push_back(buffer);
    }
    // Update expr2buffers
    func_info_.expr2buffers.Set(expr, alloc_buffers);
  }

  /*!
   * \brief Create an TIR func params and buffers with specified relax type and shape
   * \param struct_info The struct info
   * \param name_hint The name hint for params and buffers
   * \param index The index used for unique name_hint if type is Tuple.
   *              -1 means no need to add postfix since the relax param is not a Tuple.
   * \return The created TIR func params and buffers
   */
  static std::pair<Array<tir::Var>, Array<tir::Buffer>> CreateParamsAndBuffers(
      StructInfo struct_info, const String& name_hint, int index = -1) {
    Array<tir::Var> params;
    Array<tir::Buffer> buffers;
    if (const auto* tensor = struct_info.as<TensorStructInfoNode>()) {
      // Case 1. the relax param is a DynTensor, we directly create a tir var and buffer
      const auto* shape_expr = tensor->shape.as<ShapeExprNode>();
      ICHECK(shape_expr) << "FuseTIR expects all parameters are Tensors with symbolic shape.";

      String name = index == -1 ? name_hint : name_hint + "_" + std::to_string(index);
      DataType dtype = tensor->dtype;
      tir::Buffer buffer = tir::decl_buffer(shape_expr->values, dtype, name);
      // Differentiate buffer name and param name by adding prefix `v_` to param
      // Every symbol should be unique in TVMScript, and Buffer is used more than param
      // So we decide to make sure buffer names have better readability.
      tir::Var param = tir::Var("p_" + name, PrimType(DataType::Handle()));
      params.push_back(std::move(param));
      buffers.push_back(std::move(buffer));
    } else if (const auto* tuple = struct_info.as<TupleStructInfoNode>()) {
      // Case 2. the relax param is a Tuple, we recursively visit each field until it's a DynTensor
      // Enable postfix
      if (index == -1) index = 0;
      for (size_t i = 0; i < tuple->fields.size(); ++i) {
        auto ret = CreateParamsAndBuffers(tuple->fields[i], name_hint, index);
        const Array<tir::Var>& ret_params = ret.first;
        const Array<tir::Buffer>& ret_buffers = ret.second;
        ICHECK_EQ(ret_params.size(), ret_buffers.size());
        // Adding tuple field results to the end of params and buffers.
        params.insert(params.end(), ret_params.begin(), ret_params.end());
        buffers.insert(buffers.end(), ret_buffers.begin(), ret_buffers.end());
        index += ret_params.size();
      }
    } else {
      ICHECK(false) << "shapes are expected to be ShapeExprNode or TupleNode";
    }
    return std::make_pair(params, buffers);
  }

  /*!
   * \brief Construct fused TIR func with collected FuseFuncInfo
   * \return The fused TIR
   */
  tir::PrimFunc ConstructFunc() {
    Map<String, ObjectRef> attr_map;
    attr_map.Set("tir.noalias", tir::const_true());
    ICHECK(func_info_.global_name != "fused");
    // Remove output buffers from func_info_.alloc_buffers
    Array<tir::Buffer> alloc_buffers;
    for (const tir::Buffer& buf : func_info_.alloc_buffers) {
      if (func_info_.output_buffers.count(buf.get()) == 0) {
        alloc_buffers.push_back(buf);
      }
    }
    tir::Stmt body = tir::BlockNameDeduplicator()(tir::SeqStmt::Flatten(func_info_.bodies));
    body = tir::FuseTIRBufferSubstitor::Substitute(func_info_.buffer_subst_map, body);
    body = tir::Block({}, {}, {}, "root", std::move(body), NullOpt, alloc_buffers);
    body = tir::BlockRealize({}, Bool(true), Downcast<tir::Block>(body));
    tir::PrimFunc func(func_info_.params, body, VoidType(), func_info_.buffer_map,
                       DictAttrs(attr_map));
    return func;
  }

  /*! \brief Get DynTensor numbers from recursive Tuples. */
  static size_t GetTotalTensorSize(const Type& type) {
    if (type.as<DynTensorTypeNode>()) {
      return 1;
    } else if (const auto* tuple_type = type.as<TupleTypeNode>()) {
      size_t num = 0;
      for (const Type& type : tuple_type->fields) {
        num += GetTotalTensorSize(type);
      }
      return num;
    } else {
      LOG(FATAL) << "DynTensorType and TupleType are expect, but got: " << type;
      return 0;
    }
  }

  /********** Function Info **********/

  /*! \brief auxiliary information for FuseTIR */
  struct FuseFuncInfo {
    /*! \brief The arguments for calling prim_func */
    Array<Expr> arguments;
    /*!
     * \brief The map from each dataflow var (intermediate var) to the corresponding buffers
     * allocated in the fused func
     */
    Map<Expr, Array<tir::Buffer>> expr2buffers;
    /*! \brief The buffers to allocate in the fused func*/
    Array<tir::Buffer> alloc_buffers;
    /*! \brief The bodies of the original funcs, which is also the body of the fused func. */
    Array<tir::Stmt> bodies;
    /*! \brief The params of the fused function*/
    Array<tir::Var> params;
    /*!
     * \brief The map from buffer in original functions to corresponding buffer in the fused
     * function
     */
    Map<tir::Buffer, tir::Buffer> buffer_subst_map;
    /*! \brief The `buffer_map` in the fused function*/
    Map<tir::Var, tir::Buffer> buffer_map;
    /*! \brief The output buffers in the function buffer_map*/
    std::unordered_set<const tir::BufferNode*> output_buffers;
    /*! \brief The name of the fused function */
    std::string global_name = "fused";
  };

  /*! \brief The IRModule */
  const IRModule& mod_;
  /*! \brief The name hint for the input func. */
  String func_name_;
  /*! \brief The helper info to fuse TIR prim_func */
  FuseFuncInfo func_info_;
  /*! \brief The tir function after fusion*/
  tir::PrimFunc fused_tir_;
};

/*!
 * \brief The helper class to fuse TIR functions and build a new module which calls the fused TIR.
 */
class TIRFuseMutator : public ExprMutator {
 public:
  static IRModule Transform(const IRModule& mod) {
    // Since TIRFuseMutator will delete bunch of PrimFunc, we create an empty block builder.
    TIRFuseMutator mutator(mod);
    // Step 1. Fuse all primitive relax functions, store the result in `fused_tir_funcs_`
    for (const auto& kv : mod->functions) {
      const GlobalVar& gv = kv.first;
      const BaseFunc& func = kv.second;
      // Only fuse primitive relax functions
      if (func->IsInstance<relax::FunctionNode>() && func->HasNonzeroAttr(attr::kPrimitive)) {
        tir::PrimFunc fused_tir = FusedTIRConstructor::GetFusedTIR(mod, gv);
        mutator.fused_tir_funcs_.Set(gv, fused_tir);
      }
    }

    // Step 2. Update all non-primitive relax functions and add it, with the dependent function,
    // into the new IRModule
    for (const auto& kv : mod->functions) {
      const GlobalVar& gv = kv.first;
      const BaseFunc& func = kv.second;
      if (func->IsInstance<relax::FunctionNode>() && !func->HasNonzeroAttr(attr::kPrimitive)) {
        relax::Function update_func = Downcast<Function>(mutator.VisitExpr(func));
        mutator.builder_->AddFunction(update_func, gv->name_hint);
      }
    }
    return mutator.builder_->GetContextIRModule();
  }

 private:
  explicit TIRFuseMutator(const IRModule& mod) : mod_(mod) {}

  using ExprMutator::VisitExpr_;

  // Get shape from call tir
  static Expr GetCallTIRShape(StructInfo sinfo) {
    if (auto* tuple = sinfo.as<TupleStructInfoNode>()) {
      Array<Expr> fields = tuple->fields.Map([&](StructInfo x) { return GetCallTIRShape(x); });
      return Tuple(fields);
    } else {
      auto* tensor = sinfo.as<TensorStructInfoNode>();
      ICHECK(tensor) << "FuseTIR can only take tensor or tuple type";
      auto* shape_expr = tensor->shape.as<ShapeExprNode>();
      ICHECK(shape_expr) << "FuseTIR requires all intermediate values have shape";
      return GetRef<ShapeExpr>(shape_expr);
    }
  }

  Expr VisitExpr_(const CallNode* op) final {
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    Call call = Downcast<Call>(builder_->Normalize(ExprMutator::VisitExpr_(op)));

    if (call->op->IsInstance<GlobalVarNode>()) {
      // Case 1. It is a relax cross function call
      GlobalVar old_gv = Downcast<GlobalVar>(call->op);
      auto it = fused_tir_funcs_.find(old_gv);
      if (it != fused_tir_funcs_.end()) {
        const tir::PrimFunc& fused_tir = (*it).second;
        // Case 1.1. It calls a primitive relax function, update the call into a call_tir
        GlobalVar fused_tir_gv = this->builder_->AddFunction(fused_tir, old_gv->name_hint);
        // Step a. Flatten all args since call_tir does not support Tuple value.
        Array<Expr> arg_list;
        for (const Expr& arg : call->args) {
          Array<Expr> flattened = FlattenArg(arg);
          arg_list.insert(arg_list.end(), flattened.begin(), flattened.end());
        }
        // Step b. Create call_tir
        Array<Expr> call_args = {fused_tir_gv, Tuple(arg_list)};
        return Call(call_tir_op_, call_args, call->attrs, {GetStructInfo(call)});
      } else {
        // Case 1.2. The callee function is not primitive, nothing to do.
        return call;
      }
    } else if (call->op == call_tir_op_) {
      // Case 2. It is a call_tir, re-emit the PrimFunc.
      if (const auto* gv = call->args[0].as<GlobalVarNode>()) {
        tir::PrimFunc func = Downcast<tir::PrimFunc>(mod_->Lookup(GetRef<GlobalVar>(gv)));
        GlobalVar new_gv = this->builder_->AddFunction(func, gv->name_hint);
        return Call(call->op, {new_gv, call->args[1]}, call->attrs, call->sinfo_args, call->span);
      }
    }

    // Case 3. CallNode in other types. Leave it as it is.
    return call;
  }

  /********** Helper Functions **********/

  /*! \brief Flatten the call args if it's Tuple by emitting `TupleGetItem`. */
  Array<Expr> FlattenArg(const Expr& arg) {
    if (const auto* tuple_sinfo = GetStructInfoAs<TupleStructInfoNode>(arg)) {
      Array<Expr> arg_list;
      for (size_t i = 0; i < tuple_sinfo->fields.size(); ++i) {
        Expr new_arg = builder_->Emit(TupleGetItem(arg, i));
        Array<Expr> flattened = FlattenArg(new_arg);
        arg_list.insert(arg_list.end(), flattened.begin(), flattened.end());
      }
      return arg_list;
    } else {
      return {arg};
    }
  }

 private:
  /*! \brief The IRModule */
  const IRModule& mod_;
  /*! \brief The map from global var of primitive relax function to generated prim func. */
  Map<GlobalVar, tir::PrimFunc> fused_tir_funcs_;
};

IRModule FuseTIR(IRModule mod) {
  mod = TIRFuseMutator::Transform(mod);
  return mod;
}

namespace transform {

Pass FuseTIR() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) { return relax::FuseTIR(m); };
  return CreateModulePass(/*pass_function=*/pass_func,  //
                          /*opt_level=*/0,              //
                          /*pass_name=*/"FuseTIR",      //
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.FuseTIR").set_body_typed(FuseTIR);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
