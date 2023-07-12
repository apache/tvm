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
 * \file src/relax/transform/static_plan_block_memory.cc
 * \brief The static memory planning pass on BindingBlock level.
 * \details
 * The core data structure of the planning pass is StorageToken, which denotes
 * reusable memory in this planning pass.
 *
 * The memory planning pass contains three stages:
 *
 * The first stage is initialization. A storage token object will be created
 * for each builtin alloc_tensor as long as the allocated storage satisfies
 * the requirements (which are described in the code). The reference counter
 * (i.e., the times of reference) for each token is recorded.
 *
 * The second stage is allocation planning. We maintain a pool of available
 * allocated storage, in the form of storage tokens. For the storage token of
 * each builtin alloc_tensor, we check if there is appropriate available token
 * in the pool under certain criterion. If there is, we reuse that storage
 * for this alloc_tensor. Otherwise, we decide to allocate a storage for the
 * alloc_tensor.
 *
 * The third stage is IR rewrite. Based on the decision made in the second
 * stage, we insert memory alloc_storage, alloc_tensor, kill_tensor, and
 * kill_storage accordingly. Specifically, we
 * - insert alloc_storage before the site that each storage token is firstly
 * used,
 * - insert memory alloc_tensor for each builtin alloc_tensor,
 * - insert kill_tensor after the site that a tensor created by alloc_tensor
 * is last referenced, and
 * - insert kill_storage at the end of each binding block, for all the storage
 * tokens that are allocated inside the binding block, as the memory planning
 * only works on block level.
 *
 * The memory planning pass "supports" dynamic shape in the way of TIR variable
 * upper bound annotation. To be more specific, we can annotate the attribute
 * "tir_var_upper_bound" to Relax functions. The attribute value is a dict from
 * strings to integers, denoting the name of TIR variables to the upper bound
 * values of the TIR vars. **The annotated upper bound attribute only applies
 * to TIR vars in the function signature for clarity.**
 *
 * For example, we can annotate a Relax function with
 *   `R.func_attr({"tir_var_upper_bound": {"n": 1024}})`.
 * It means the maximum value of variable that names "n" in the function
 * signature will have upper bound 1024. And we will use 1024 as its value
 * during memory planning.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/stmt_functor.h>

#include <map>
#include <set>
#include <vector>

namespace tvm {
namespace relax {

/*!
 * \brief A representation of a block of reusable memory required at runtime.
 * \details Only the tensors whose memory can be "possibly reused" will have
 * their storage token. In other words, we do not have storage token for tensor
 * - that is a function parameter,
 * - that is a function return value,
 * - one of whose use site is a BindingBlock different from its allocation site,
 * - that is used as a condition or branch return of a IfNode,
 * - that is used as the body of a SeqExprNode,
 * - that is used as arguments in a Call whose op is not a PrimFunc.
 *
 * In practice, we do create a storage token for such tensor at first. But at
 * any time we find a tensor satisfying any of the conditions above, we erase
 * its storage token.
 */
class StorageTokenNode : public Object {
 public:
  /*! \brief Reference counter. */
  int ref_counter{0};
  /*! \brief Number of bytes that this token requires. */
  int64_t bytes;
  /*! \brief The dtype of this token. */
  DataType dtype;
  /*! \brief The storage id, reserved for debug and demo use. */
  int storage_id{-1};

  static constexpr const char* _type_key = "relax.transform.StorageToken";
  TVM_DECLARE_BASE_OBJECT_INFO(StorageTokenNode, Object);
};

/*!
 * \brief Managed reference to StorageTokenNode.
 * \sa StorageTokenNode
 */
class StorageToken : public ObjectRef {
 public:
  explicit StorageToken(Array<PrimExpr> shape, DataType dtype) {
    // Compute the tensor size from the shape.
    int64_t size = 1;
    for (const PrimExpr& dim_len : shape) {
      const auto* int_len = dim_len.as<IntImmNode>();
      ICHECK_NOTNULL(int_len);
      size *= int_len->value;
    }

    ObjectPtr<StorageTokenNode> n = make_object<StorageTokenNode>();
    n->bytes = size * dtype.bytes() * dtype.lanes();
    n->dtype = dtype;
    data_ = std::move(n);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(StorageToken, ObjectRef, StorageTokenNode);
};

// We use NestedMsg to store the tokens used by each Expr.
using Tokens = NestedMsg<StorageToken>;

/*!
 * \brief Memory manager for flattened 1d memory (buffers)
 * \note We can generalize this implementation to multi-dimensional memory
 * following the same flow in the future.
 */
class TokenAllocator1D {
 public:
  /*!
   * \brief Request a storage token from the available token pool for a
   * given prototype, or report no appropriate available token in the pool.
   * \param prototype The requesting prototype storage token.
   * \return The request result token. Return NullOpt if there is no
   * appropriate available token in the pool.
   */
  Optional<StorageToken> RequestReuse(StorageToken prototype) {
    // Step 0. Sanity check: the prototype token is supposed not to be allocated with actual storage
    ICHECK_EQ(prototype->storage_id, -1) << "The token is expected not to be allocated before.";
    // If the prototype has no reference at all, feel free to allocate new storage.
    // The unused binding can be removed by cleaning passes.
    if (prototype->ref_counter == 0) {
      return NullOpt;
    }

    // Step 1. Get the available pool of the token dtype.
    std::multimap<int64_t, StorageToken>& pool = available_pool_[prototype->dtype];

    // Step 2. Get the range of memory blocks in [size / match_range_, size * match_range_)
    int64_t size = prototype->bytes;
    auto begin = pool.lower_bound(size / match_range_);
    auto mid = pool.lower_bound(size);
    auto end = pool.upper_bound(size * match_range_);
    // Step 3. Search for memory block that equals or is larger than the requested size.
    if (mid != end) {
      StorageToken available_token = mid->second;
      ICHECK_EQ(available_token->ref_counter, 0)
          << "Available tokens are expected to have 0 reference.";
      ICHECK_LE(size, available_token->bytes);
      available_token->ref_counter = prototype->ref_counter;
      pool.erase(mid);
      return available_token;
    }
    // Step 4. Then search for memory block that is smaller than the requested size.
    if (mid != begin) {
      --mid;
      StorageToken available_token = mid->second;
      ICHECK_EQ(available_token->ref_counter, 0)
          << "Available tokens are expected to have 0 reference.";
      ICHECK_GE(size, available_token->bytes);
      // Enlarge the token size.
      available_token->bytes = size;
      available_token->ref_counter = prototype->ref_counter;
      pool.erase(mid);
      return available_token;
    }
    // Return `NullOpt` indicating that no satisfiable storage token is found in the available pool.
    return NullOpt;
  }

  /*!
   * \brief Allocate a storage token for the input prototype token.
   * \param prototype The prototype token.
   * \param storage_id The id of this token.
   */
  StorageToken Alloc(StorageToken prototype, int storage_id) {
    // Sanity check: the prototype token is supposed not to be allocated with actual storage yet
    ICHECK_EQ(prototype->storage_id, -1) << "The token is expected not to be allocated before.";
    prototype->storage_id = storage_id;
    full_pool_.push_back(prototype);
    return prototype;
  }

  /*!
   * \brief Release the input token, putting it into the available pool.
   * \param token The token to be released.
   */
  void Release(StorageToken token) {
    // Sanity check: the token has been allocated with actual storage, and should have 0 reference.
    ICHECK_GE(token->storage_id, 0)
        << "The token to be released is expected to be allocated before";
    ICHECK_EQ(token->ref_counter, 0) << "The token to be released is expected to have 0 reference.";
    available_pool_[token->dtype].insert({token->bytes, token});
  }

 private:
  /*! \brief A constant scale representing the token search range. */
  const int match_range_{16};
  /*! \brief The pool of available storage tokens for each dtype. */
  std::unordered_map<DataType, std::multimap<int64_t, StorageToken>> available_pool_;
  /*! \brief All the storage tokens that have been allocated with actual storage. */
  std::vector<StorageToken> full_pool_;
};

/*! \brief Check if the input op is "relax.reshape". */
bool IsReshape(const Expr& op) { return op.same_as(Op::Get("relax.reshape")); }

/*! \brief The base class for the storage allocation visitor. */
class StorageAllocatorBaseVisitor : public ExprVisitor {
 protected:
  using ExprVisitor::VisitExpr_;

  void VisitBindingBlock_(const BindingBlockNode* block) override {
    // We maintain a block stack for token allocation-site and use-site check.
    block_stack_.push_back(block);
    ExprVisitor::VisitBindingBlock_(block);
    ICHECK(!block_stack_.empty());
    ICHECK(block_stack_.back() == block);
    block_stack_.pop_back();
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    ExprVisitor::VisitBinding_(binding);
    // The binding var has the same tokens as the binding value.
    SetTokens(binding->var.get(), token_map_[binding->value.get()]);
  }

  void VisitExpr_(const TupleNode* tuple) final {
    Array<Tokens> tokens;
    tokens.reserve(tuple->fields.size());
    for (const Expr& field : tuple->fields) {
      Tokens field_tokens = GetTokens(field);
      tokens.push_back(field_tokens);
    }
    SetTokens(tuple, Tokens(tokens));
  }

  void VisitExpr_(const TupleGetItemNode* tuple_item) final {
    Tokens tokens = GetTokens(tuple_item->tuple);
    // If the tuple has no token, every of its field has no token as well.
    if (tokens.IsNull()) {
      token_map_[tuple_item] = Tokens();
      return;
    }
    ICHECK(tokens.IsNested());
    Array<Tokens> field_tokens = tokens.NestedArray();
    ICHECK_GT(static_cast<int>(field_tokens.size()), tuple_item->index);
    ICHECK_GE(tuple_item->index, 0);
    SetTokens(tuple_item, field_tokens[tuple_item->index]);
  }

  /******************** Utilities ********************/

  Tokens GetTokens(const Expr& expr) {
    this->VisitExpr(expr);
    return token_map_[expr.get()];
  }

  virtual void SetTokens(const ExprNode* expr, Tokens tokens) { token_map_[expr] = tokens; }

  /*! \brief The mapping from each Expr to its corresponding storage tokens. */
  std::unordered_map<const ExprNode*, Tokens> token_map_;
  /*! \brief The binding block stack. */
  std::vector<const BindingBlockNode*> block_stack_;
};

/*!
 * \brief The visitor class for storage token initialization.
 * \details It goes through the entire function to get the storage tokens
 * used by each Expr. After the initialization, we
 * - know the tokens that each Expr is using,
 * - know the number of references for each token,
 * - rule out the builtin alloc_tensors to which the planning does not apply.
 */
class StorageAllocatorInit : public StorageAllocatorBaseVisitor {
 public:
  /*!
   * \brief The entry of the initialization.
   * \param mod The IRModule to be planned
   * \return The mapping from each Expr to the token it uses.
   */
  static std::unordered_map<const ExprNode*, Tokens> Initialize(const IRModule& mod) {
    StorageAllocatorInit initializer(mod);

    for (auto it : mod->functions) {
      const auto* func = it.second.as<FunctionNode>();
      if (func == nullptr) {
        continue;
      }
      initializer(GetRef<Function>(func));
    }
    return initializer.token_map_;
  }

 private:
  using ExprVisitor::VisitExpr_;

  explicit StorageAllocatorInit(const IRModule& ctx_mod) : ctx_mod_(ctx_mod) {}

  void VisitExpr_(const FunctionNode* func) final {
    // Use the attribute-annotated TIR var upper bounds as the TIR var values for
    // memory planning.
    // NOTE: we only apply the annotated upper bounds to the TIR variables that
    // appear in the **function signature**.
    Map<String, IntImm> var_upper_bound_attr =
        func->GetAttr<Map<String, IntImm>>("tir_var_upper_bound").value_or(Map<String, IntImm>());
    Array<tir::Var> var_in_signature = TIRVarsInStructInfo(GetStructInfo(GetRef<Function>(func)));
    var_upper_bound_.clear();
    for (const tir::Var& tir_var : var_in_signature) {
      auto it = var_upper_bound_attr.find(tir_var->name_hint);
      if (it != var_upper_bound_attr.end()) {
        ana_.Bind(tir_var, tvm::Range::FromMinExtent(
                               tvm::IntImm(DataType::Int(64), 0),
                               tvm::IntImm(DataType::Int(64), (*it).second->value + 1)));
      }
    }

    // Recurse into the function to get its tokens.
    Tokens body_tokens = GetTokens(func->body);
    // Discard the tokens used by the function return value, as they are external referenced.
    DiscardTokensIn(body_tokens);
  }

  void VisitExpr_(const CallNode* call) final {
    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    static const Op& call_tir_dyn_op = Op::Get("relax.vm.call_tir_dyn");

    if (call->op == alloc_tensor_op) {
      // Create a storage token for builtin alloc_tensor.
      this->CreateToken(call);
      return;
    } else if (IsReshape(call->op)) {
      // Reuse the input's token for builtin reshape.
      SetTokens(call, GetTokens(call->args[0]));
      return;
    }

    // - Increase the reference counters of the arguments when the callee is
    // a PrimFunc of the context module or an external function via 'call_packed'.
    // It assumes external function calls via 'call_packed' do not retain memory
    // from the arguments.
    // - Otherwise, discard the tokens used by the arguments, as there might be
    // potential external reference.
    if (IsPrimFuncGlobalVar(call->op) || call->op->IsInstance<ExternFuncNode>() ||
        call->op == call_tir_dyn_op) {
      Array<Expr> args =
          call->op == call_tir_dyn_op ? Downcast<Tuple>(call->args[1])->fields : call->args;
      ICHECK(!block_stack_.empty());
      for (const Expr& arg : call->args) {
        Tokens tokens = GetTokensWithAllocSiteCheck(arg, block_stack_.back());
        ForEachLeaf(tokens, [](StorageToken token) { token->ref_counter += 1; });
      }
    } else {
      for (const Expr& arg : call->args) {
        DiscardTokensIn(GetTokens(arg));
      }
    }
  }

  void VisitExpr_(const IfNode* if_node) final {
    Tokens cond_tokens = GetTokens(if_node->cond);
    Tokens then_tokens = GetTokens(if_node->true_branch);
    Tokens else_tokens = GetTokens(if_node->false_branch);
    // Discard the tokens used by the condition, then-body and else-body,
    // as the planning works on block level.
    DiscardTokensIn(cond_tokens);
    DiscardTokensIn(then_tokens);
    DiscardTokensIn(else_tokens);
  }

  void VisitExpr_(const SeqExprNode* seq) final {
    for (const BindingBlock& binding_block : seq->blocks) {
      this->VisitBindingBlock(binding_block);
    }
    Tokens body_tokens = GetTokens(seq->body);
    // Discard the tokens used by the body, as the planning works on block level.
    DiscardTokensIn(body_tokens);
  }

  /******************** Utilities ********************/

  /*!
   * \brief Check if the input op is GlobalVar corresponding to a PrimFunc inside the ctx module.
   * \param op The op to be checked
   * \return A boolean indicating if the input op corresponds to a PrimFunc.
   */
  bool IsPrimFuncGlobalVar(const Expr& op) {
    const auto* global_var = op.as<GlobalVarNode>();
    if (global_var == nullptr) {
      return false;
    }
    auto func_it = ctx_mod_->functions.find(GetRef<GlobalVar>(global_var));
    if (func_it == ctx_mod_->functions.end()) {
      return false;
    }
    return (*func_it).second->IsInstance<tir::PrimFuncNode>();
  }

  /*!
   * \brief Create a storage token for the builtin alloc_tensor call.
   * \param call The call to be processed.
   * \return The created token.
   */
  Tokens CreateToken(const CallNode* call) {
    // Sanity checks about
    // - the call return value is a Tensor;
    // - the shape of the tensor is known, in the form of ShapeExpr;
    // - the tensor has known dtype;
    // - no storage token was created for this call before.
    const auto* sinfo = call->struct_info_.as<TensorStructInfoNode>();
    const auto* shape = sinfo->shape.as<ShapeExprNode>();
    ICHECK_NOTNULL(sinfo);
    ICHECK_NOTNULL(shape);
    ICHECK(!sinfo->IsUnknownDtype());
    ICHECK(sinfo->dtype == Downcast<DataTypeImm>(call->args[1])->value);
    ICHECK(!token_map_.count(call));

    // Use the upper bounds of TIR vars as their values.
    Array<PrimExpr> upper_bounded_shape;
    upper_bounded_shape.reserve(shape->values.size());
    for (const PrimExpr& dim_len : shape->values) {
      int64_t max_bound = ana_.const_int_bound(dim_len)->max_value;
      if (max_bound == std::numeric_limits<int64_t>::max()) {
        upper_bounded_shape.push_back(dim_len);
      } else {
        upper_bounded_shape.push_back(tvm::IntImm(DataType::Int(64), max_bound));
      }
    }

    // No support for TIR vars that are not bounded.
    for (const PrimExpr& dim_len : upper_bounded_shape) {
      const auto* int_len = dim_len.as<IntImmNode>();
      if (!int_len) {
        token_map_[call] = Tokens();
        return Tokens();
      }
    }

    // Create and set token.
    StorageToken token(upper_bounded_shape, sinfo->dtype);

    Tokens tokens(token);
    SetTokens(call, tokens);
    ICHECK(!block_stack_.empty());
    token2block_[token.get()] = block_stack_.back();
    return tokens;
  }

  /*!
   * \brief Override the token setter in the base visitor.
   * For each token, we keep record of all Expr that are using that token.
   * When we want to discard one token, we use the records to remove the token
   * from the Expr that are using it.
   */
  void SetTokens(const ExprNode* expr, Tokens tokens) final {
    StorageAllocatorBaseVisitor::SetTokens(expr, tokens);
    ForEachLeaf(tokens, [this, expr](StorageToken token) {
      this->token2exprs_[token.get()].push_back(expr);
    });
  }

  /*!
   * \brief Token getter with allocation site check.
   * We first get the tokens used by the input Expr, and check if the allocation
   * site of each token is the input current block.
   * Since the planning works on block level, if some token's allocation site
   * is not the current block, we discard the token so that it will not be planned.
   * \param expr The Expr whose tokens is to be got.
   * \param cur_block The pointer to the current block.
   * \return The tokens used by the input Expr.
   */
  Tokens GetTokensWithAllocSiteCheck(const Expr& expr, const BindingBlockNode* cur_block) {
    Tokens tokens = GetTokens(expr);
    ForEachLeaf(tokens, [this, cur_block](StorageToken token) {
      auto it = this->token2block_.find(token.get());
      ICHECK(it != this->token2block_.end());
      if (it->second != cur_block) {
        this->DiscardToken(token);
      }
    });
    return token_map_[expr.get()];
  }

  /*! \brief Discard the input tokens. */
  void DiscardTokensIn(Tokens tokens) {
    ForEachLeaf(tokens, [this](StorageToken token) { this->DiscardToken(token); });
  }

  /*!
   * \brief Discard the input token.
   * For each Expr that is using the input token, remove the token from the Expr's token set.
   * \param token_to_discard The token to be discarded.
   */
  void DiscardToken(StorageToken token_to_discard) {
    const std::vector<const ExprNode*>& exprs = token2exprs_[token_to_discard.get()];
    for (const ExprNode* expr : exprs) {
      token_map_[expr] = MapNestedMsg(token_map_[expr], [token_to_discard](StorageToken token) {
        return token.same_as(token_to_discard) ? Tokens() : Tokens(token);
      });
    }
    token2exprs_.erase(token_to_discard.get());
    token2block_.erase(token_to_discard.get());
  }

  /*! \brief The arithmetic analyzer. */
  arith::Analyzer ana_;
  /*!
   * \brief The context IRModule, used for checking if a callee function is
   * a PrimFunc inside the IRModule.
   */
  const IRModule& ctx_mod_;
  /*! \brief The mapping from TIR variables to their respective upper bound values. */
  std::unordered_map<tir::Var, IntImm, ObjectPtrHash, ObjectPtrEqual> var_upper_bound_;
  /*! \brief The mapping from each token to the binding block where it is created. */
  std::unordered_map<const StorageTokenNode*, const BindingBlockNode*> token2block_;
  /*! \brief The mapping from each token to the Exprs that are using this token. */
  std::unordered_map<const StorageTokenNode*, std::vector<const ExprNode*>> token2exprs_;
};

/*!
 * \brief The visitor class for storage token allocation planning.
 * \details
 * - For each builtin alloc_tensor whose token is not discarded in the
 * initialization stage, we request a storage reuse or decide to allocate
 * storage for this token, depending on if there is appropriate available
 * token in the token pool we maintain.
 * - For each VM builtin reshape, we reuse the input's tokens.
 *
 * After the allocation planning, we
 * - know the token that each builtin alloc_tensor plans to use. Compared
 * with the initialization, here the token is possibly a reuse of some
 * previous token, rather than we having one token for each alloc_tensor.
 * - know the last referenced site for each builtin alloc_tensor. This
 * information is used for inserting kill_tensor in the rewrite stage.
 * - know the tokens allocated in each binding block. This information
 * is used for inserting kill_storage in the rewrite stage.
 */
class StorageAllocator : public StorageAllocatorBaseVisitor {
 public:
  explicit StorageAllocator(std::unordered_map<const ExprNode*, Tokens> token_map) {
    this->token_map_ = std::move(token_map);
  }

  void Allocate(const IRModule& mod) {
    for (auto it : mod->functions) {
      const auto* func = it.second.as<FunctionNode>();
      if (func == nullptr) {
        continue;
      }
      this->VisitExpr_(func);
    }
  }

  /*!
   * \brief The mapping from each `builtin.alloc_tensor` to its corresponding
   * underlying storage token that it is using.
   */
  std::unordered_map<const ExprNode*, StorageToken> alloc_tensor2token;
  /*! \brief The mapping from each Expr to the tensors that need to be killed after it. */
  std::unordered_map<const ExprNode*, std::vector<Var>> expr2killed_tensors;
  /*! \brief The mapping from each binding block to the storage tokens that are create inside. */
  std::unordered_map<const BindingBlockNode*, std::vector<const StorageTokenNode*>> block2tokens;

 private:
  using ExprVisitor::VisitBinding_;
  using ExprVisitor::VisitExpr_;

  void VisitBindingBlock_(const BindingBlockNode* block) final {
    StorageAllocatorBaseVisitor::VisitBindingBlock_(block);
    // Sanity check: each token allocated inside the block should not be
    // referenced by anyone at the end of the block.
    for (const StorageTokenNode* token : block2tokens[block]) {
      ICHECK_EQ(token->ref_counter, 0);
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call) final {
    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    if (call->op == alloc_tensor_op) {
      auto it = token_map_.find(call);
      ICHECK(it != token_map_.end());

      if (it->second.IsNull()) {
        // IsNull being true means the token was discarded, and this alloc_tensor
        // is not considered by the planning.
        return;
      }
      ICHECK(it->second.IsLeaf());
      StorageToken new_token = this->RequestReuseOrAlloc(it->second.LeafValue());

      // Record that this alloc_tensor is using the token.
      alloc_tensor2token.insert({call, new_token});
      token2cur_tensor_[new_token.get()].push_back(binding->var);
      SetTokens(call, Tokens(new_token));
      // Record that the token is allocated in the current block.
      ICHECK(!block_stack_.empty());
      std::vector<const StorageTokenNode*>& block_tokens = block2tokens[block_stack_.back()];
      if (std::find(block_tokens.begin(), block_tokens.end(), new_token.get()) ==
          block_tokens.end()) {
        block_tokens.push_back(new_token.get());
      }
      return;
    } else if (IsReshape(call->op)) {
      Tokens tokens = GetTokens(call->args[0]);
      ICHECK(!tokens.IsNested());
      if (tokens.IsLeaf()) {
        // If the input is using a token, record that the reshape uses the token as well.
        token2cur_tensor_[tokens.LeafValue().get()].push_back(binding->var);
        SetTokens(call, tokens);
      } else {
        ICHECK(token_map_[call].IsNull());
      }
      return;
    }

    // Decrease the reference counter by one for each token that the arguments use.
    // Check if a token can be released (i.e., has no reference) after decrease.
    // And release it if so.
    for (const Expr& arg : call->args) {
      Tokens tokens = GetTokens(arg);
      ForEachLeaf(tokens, [this, call](StorageToken token) {
        ICHECK_GT(token->ref_counter, 0);
        token->ref_counter -= 1;
        this->CheckForRelease(token, call);
      });
    }
  }

  /*! \brief Request a storage reuse, or allocate storage if no appropriate storage is reusable. */
  StorageToken RequestReuseOrAlloc(StorageToken prototype) {
    Optional<StorageToken> token = allocator_.RequestReuse(prototype);
    if (!token.defined()) {
      return allocator_.Alloc(prototype, this->n_storage_++);
    } else {
      return token.value();
    }
  }

  /*!
   * \brief Check if a token has no reference and thus can be released. And release it if so.
   * \param token The token to be checked.
   * \param release_site The CallNode where the the input token is send for release.
   * If the token is checked to release here, we keep record of the release site so that
   * kill_tensor can be inserted here at the rewrite stage.
   */
  void CheckForRelease(StorageToken token, const CallNode* release_site) {
    // Sanity check: the token was allocated before and has non-negative reference.
    ICHECK_GE(token->storage_id, 0);
    ICHECK_GE(token->ref_counter, 0);

    if (token->ref_counter == 0) {
      allocator_.Release(token);
      auto it = token2cur_tensor_.find(token.get());
      ICHECK(it != token2cur_tensor_.end());
      // Record that the tensors that are using this token will be killed
      // immediately after the release site.
      std::vector<Var>& killed_tensors = expr2killed_tensors[release_site];
      killed_tensors.insert(killed_tensors.end(), it->second.begin(), it->second.end());
      token2cur_tensor_.erase(it);
    }
  }

  /*! \brief Number of allocated storages. */
  int n_storage_{0};
  /*! \brief The 1D memory allocator. */
  TokenAllocator1D allocator_;
  /*! \brief The mapping from each token to the tensors that are currently using it. */
  std::unordered_map<const StorageTokenNode*, std::vector<Var>> token2cur_tensor_;
};

/*!
 * \brief The rewriter class based on the token allocation planning.
 * \details
 * - For each builtin alloc_tensor that was planned, substitute it with a memory
 * alloc_tensor. If no memory alloc_storage was created for it before, create one.
 * - Insert memory kill_tensor at the release site of each tensor.
 * - Insert memory kill_storage at the end of each binding block, for the tokens allocated in it.
 */
class StorageAllocationRewriter : public ExprMutator {
 public:
  explicit StorageAllocationRewriter(
      IRModule mod, std::unordered_map<const ExprNode*, StorageToken> alloc_tensor2token,
      std::unordered_map<const ExprNode*, std::vector<Var>> expr2killed_tensors,
      std::unordered_map<const BindingBlockNode*, std::vector<const StorageTokenNode*>>
          block2tokens)
      : ExprMutator(std::move(mod)),
        alloc_tensor2token_(std::move(alloc_tensor2token)),
        expr2killed_tensors_(std::move(expr2killed_tensors)),
        block2tokens_(std::move(block2tokens)) {}

  IRModule Rewrite() {
    const IRModule& mod = builder_->GetContextIRModule();
    for (const auto& [gv, base_func] : mod->functions) {
      const auto* func_ = base_func.as<FunctionNode>();
      if (func_ == nullptr) {
        continue;
      }
      token2storage_var_.clear();
      Function func = Downcast<Function>(this->VisitExpr_(func_));
      builder_->UpdateFunction(gv, func);
    }
    return builder_->GetContextIRModule();
  }

 private:
  using ExprMutator::VisitExpr_;

  BindingBlock VisitBindingBlock_(const BindingBlockNode* block) final {
    builder_->BeginBindingBlock();
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }

    // Insert `memory.kill_storage` for the storage tokens allocated inside this block.
    for (const StorageTokenNode* token : block2tokens_[block]) {
      auto it_token = token2storage_var_.find(token);
      ICHECK(it_token != token2storage_var_.end());
      static const Op& mem_kill_storage = Op::Get("relax.memory.kill_storage");
      this->builder_->Emit(Call(mem_kill_storage, {it_token->second}), /*name_hint=*/"_");
    }

    BindingBlock new_block = builder_->EndBlock();
    return new_block;
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    ExprMutator::VisitBinding_(binding);

    // Insert `memory.kill_tensor` for the tensors that need to be killed after this binding.
    auto it = expr2killed_tensors_.find(binding->value.get());
    if (it != expr2killed_tensors_.end()) {
      for (const Var& var : it->second) {
        static const Op& mem_kill_tensor = Op::Get("relax.memory.kill_tensor");
        this->builder_->Emit(Call(mem_kill_tensor, {Downcast<Var>(this->VisitExpr(var))}),
                             /*name_hint=*/"_");
      }
    }
  }

  Expr VisitExpr_(const CallNode* call) final {
    auto it = alloc_tensor2token_.find(call);
    if (it != alloc_tensor2token_.end()) {
      const auto* sinfo = call->struct_info_.as<TensorStructInfoNode>();
      ICHECK_NOTNULL(sinfo);
      ICHECK_NOTNULL(sinfo->shape.as<ShapeExprNode>());
      PrimValue runtime_device_index = Downcast<PrimValue>(call->args[2]);

      // If the token is visited for the first time, create a storage variable using
      // `memory.alloc_storage` for it.
      StorageToken token = it->second;
      Var storage_var{nullptr};
      auto it_token = token2storage_var_.find(token.get());
      if (it_token == token2storage_var_.end()) {
        static const Op& mem_alloc_storage = Op::Get("relax.memory.alloc_storage");
        ShapeExpr size({tir::make_const(DataType::Int(64), token->bytes)});
        PrimValue virtual_device_index = runtime_device_index;
        std::string storage_scope = "global";
        DataType dtype = token->dtype;
        Call alloc_storage(
            mem_alloc_storage,
            {std::move(size), virtual_device_index, StringImm(storage_scope), DataTypeImm(dtype)},
            Attrs());
        storage_var = builder_->Emit(alloc_storage, "storage");
        token2storage_var_[token.get()] = storage_var;
      } else {
        storage_var = it_token->second;
      }

      // And always create a `memory.alloc_tensor` for the old `builtin.alloc_tensor`.
      static const Op& mem_alloc_tensor = Op::Get("relax.memory.alloc_tensor");
      PrimValue offset = PrimValue::Int64(0);
      DataType dtype = sinfo->dtype;
      return Call(mem_alloc_tensor, {storage_var, offset, sinfo->shape.value(), DataTypeImm(dtype)},
                  Attrs());
    }

    return ExprMutator::VisitExpr_(call);
  }

  /*!
   * \brief The mapping from each memory-reusable `builtin.alloc_tensor` to
   its corresponding underlying storage token that it is using.
   */
  std::unordered_map<const ExprNode*, StorageToken> alloc_tensor2token_;
  /*! \brief The mapping from each Expr to the tensors that need to be killed after it. */
  std::unordered_map<const ExprNode*, std::vector<Var>> expr2killed_tensors_;
  /*! \brief The mapping from each binding block to the storage tokens that are create inside. */
  std::unordered_map<const BindingBlockNode*, std::vector<const StorageTokenNode*>> block2tokens_;
  /*! \brief The mapping from each token to its corresponding storage var in each function. */
  std::unordered_map<const StorageTokenNode*, Var> token2storage_var_;
};

IRModule StaticPlanBlockMemory(IRModule mod) {
  // Step 1. Initialize.
  std::unordered_map<const ExprNode*, Tokens> token_map = StorageAllocatorInit::Initialize(mod);
  // Step 2. Collect the memory allocation info.
  StorageAllocator allocator(std::move(token_map));
  allocator.Allocate(mod);
  // Step 3. Rewrite the function.
  StorageAllocationRewriter rewriter(std::move(mod),  //
                                     std::move(allocator.alloc_tensor2token),
                                     std::move(allocator.expr2killed_tensors),
                                     std::move(allocator.block2tokens));
  return rewriter.Rewrite();
}

namespace transform {

Pass StaticPlanBlockMemory() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::StaticPlanBlockMemory(std::move(m)); };
  return CreateModulePass(pass_func, /*opt_level=*/0, "StaticPlanBlockMemory", {});
}

TVM_REGISTER_GLOBAL("relax.transform.StaticPlanBlockMemory").set_body_typed(StaticPlanBlockMemory);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
