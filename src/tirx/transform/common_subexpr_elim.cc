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
 * \file common_subexpr_elim.cc
 * \brief Two-phase Common Subexpression Elimination (CSE) for TIR.
 *
 * Architecture overview
 * ---------------------
 * The pass is structured as two cooperating phases (single plan, single rewrite):
 *
 *   Phase 1 — **CSEPlanner** (analysis, no mutation)
 *     Walks the TIR tree bottom-up and builds:
 *       - A *scope tree* that mirrors the nesting structure of For/If/While/AttrStmt.
 *       - An *expression DAG* mapping each structurally-unique eligible expression
 *         to its occurrence count, LCA scope, first-use location, and direct
 *         children (which shallower expressions it contains).
 *     From this it produces a *plan* in a single pass (shallower expressions
 *     first): two tables describing what to insert where (InsertBeforeTable)
 *     and what to replace (ExprRemapTable). Shallower-first processing with
 *     repr propagation resolves all CSE opportunities without a cascade loop.
 *
 *   Phase 2 — **CSERewriter** (mechanical mutation)
 *     Consumes the plan and performs two kinds of edits:
 *       - Inserts `Bind(cse_var, expr)` statements at the planned insertion points.
 *       - Replaces every occurrence of a CSE'd expression with its variable.
 *     Insertions are handled by overriding VisitStmt and wrapping in SeqStmt;
 *     SeqStmt flattening handles correct nesting.
 *
 * Eligibility rules
 * -----------------
 * An expression is eligible for CSE if:
 *   - It is not a leaf (Var, IntImm, FloatImm, StringImm).
 *   - It does not contain Call or BufferLoad (side-effects / memory dependence).
 *   - It is not Ramp or Broadcast (hardware-specific vector ops).
 *
 * Scope tree
 * ----------
 * Each For, IfThenElse (each branch), While, and AttrStmt body creates a new
 * scope. The scope tree enables computing the Lowest Common Ancestor (LCA) of
 * all scopes where an expression occurs, which determines the correct insertion
 * point — the narrowest scope that dominates all uses.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../analysis/check_contains.h"

namespace tvm {
namespace tirx {

// ============================================================================
// Plan interface types (internal, C++ only)
// ============================================================================

/*!
 * \brief Map from expression to CSE variable, keyed by structural equality.
 *
 * Used by CSERewriter to look up whether a visited expression should be
 * replaced by a previously-introduced CSE variable.
 */
using ExprRemapTable = std::unordered_map<PrimExpr, Var, ffi::StructuralHash, ExprDeepEqual>;

/*!
 * \brief Map from statement (by pointer identity) to a list of Bind
 *        statements that should be inserted immediately before it.
 *
 * Pointer identity (ffi::ObjectPtrHash/Equal) is used because the insertion
 * point is a specific child of a SeqStmt, not a structurally-equivalent
 * statement elsewhere in the tree.
 */
using InsertBeforeTable =
    std::unordered_map<Stmt, std::vector<Stmt>, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

// ============================================================================
// CSEPlanner: Phase 1 — scan tree, build scope tree + expression table
// ============================================================================

/*!
 * \brief Phase 1 of the two-phase CSE pass.
 *
 * CSEPlanner is a read-only visitor that scans the TIR tree bottom-up and builds:
 *   1. A **scope tree** (vector of ScopeEntry) reflecting For/If/While/AttrStmt nesting.
 *   2. An **expression DAG** (ExprTable) where each node is an eligible expression
 *      with occurrence count, expr_depth, LCA scope, first-use location, and
 *      direct children (other table entries reachable without passing through
 *      another table entry). Children and expr_depth are computed incrementally
 *      during the bottom-up scan — no separate traversal needed.
 *
 * After scanning, ComputePlan() converts the internal state into two output tables:
 *   - InsertBeforeTable: where to insert `Bind(cse_var, expr)` statements.
 *   - ExprRemapTable: which expressions to replace with their CSE variable.
 *
 * Usage:
 * \code
 *   auto [insert_before, expr_remap] = CSEPlanner::Plan(body, params);
 * \endcode
 */
class CSEPlanner : public StmtExprVisitor {
 public:
  /*!
   * \brief Run the planner on a function body (static entry point).
   *
   * Creates a planner instance, initializes the root scope, scans the body,
   * and returns the computed plan.
   *
   * \param body The TIR function body to analyze.
   * \return A pair of (InsertBeforeTable, ExprRemapTable) describing the
   *         planned CSE transformations.
   */
  static std::pair<InsertBeforeTable, ExprRemapTable> Plan(const Stmt& body) {
    CSEPlanner planner;
    // Root scope (no parent, depth 0, no creator statement)
    planner.scopes_.push_back({-1, 0, Stmt()});
    planner.current_scope_ = 0;
    // Scan the tree (VisitStmt sets current_stmt_ automatically)
    planner.VisitStmt(body);
    // Convert scan results into the plan
    return planner.ComputePlan();
  }

 private:
  /*!
   * \brief One node in the scope tree.
   *
   * The scope tree mirrors the nesting structure of the TIR program.
   * Each scope-creating statement (For, IfThenElse branch, While, AttrStmt)
   * gets its own ScopeEntry. The root scope (depth 0) represents the function
   * body itself.
   */
  struct ScopeEntry {
    /*! \brief Parent scope ID (-1 for root). */
    int parent;
    /*! \brief Distance from root (root = 0). */
    int depth;
    /*!
     * \brief The statement that created this scope (e.g. ForNode).
     *
     * Null for the root scope. Used as the insertion point when a CSE
     * binding must be placed before the scope.
     */
    Stmt creator_stmt;
  };

  /*!
   * \brief Node in the expression DAG built during the bottom-up scan.
   *
   * The planner maintains one ExprEntry per structurally-unique eligible
   * expression (keyed by ExprDeepEqual). Since expressions are recorded
   * bottom-up (children before parents), the DAG children are naturally
   * discovered when a node is first added. Fields like expr_depth are
   * computed incrementally from children — no separate traversal needed.
   */
  struct ExprEntry {
    /*! \brief Total number of occurrences across all scopes. */
    int count{0};
    /*!
     * \brief Nesting depth of eligible sub-expressions (leaf eligible = 1).
     *
     * Computed from children: `1 + max(child.expr_depth)`, or 1 if no children.
     * Used to sort entries so that shallower expressions are processed first.
     */
    int expr_depth{0};
    /*! \brief The expression itself (first occurrence). */
    PrimExpr repr;
    /*!
     * \brief Scope ID of the Lowest Common Ancestor of all scopes containing an occurrence.
     *
     * Determines the outermost valid insertion point.
     */
    int lca_scope{-1};
    /*!
     * \brief Scope ID where the first occurrence was found.
     *
     * When lca_scope == first_use_scope, the binding is inserted before first_use_stmt.
     */
    int first_use_scope{-1};
    /*!
     * \brief The statement containing the first occurrence.
     *
     * Used as the insertion point when the LCA equals the first-use scope.
     */
    Stmt first_use_stmt;
    /*!
     * \brief Direct children in the expression DAG: (child_expr, multiplicity).
     *
     * A "direct child" is an eligible table entry reachable from this expression
     * without passing through another table entry. Multiplicity counts how many
     * times the child appears (e.g., 2 for `(x+y) * (x+y)` with child `x+y`).
     * Populated during RecordExpr (bottom-up: children already in table).
     */
    std::vector<std::pair<PrimExpr, int>> children;
    /*!
     * \brief Number of occurrences consumed by parent expressions' CSE bindings.
     *
     * Computed after the DAG is fully built, before plan generation.
     * Independent count = count - consumed; only entries with independent >= 2
     * are CSE candidates.
     */
    int consumed{0};
  };

  /*! \brief Expression table keyed by structural equality (ExprDeepEqual). */
  using ExprTable = std::unordered_map<PrimExpr, ExprEntry, ffi::StructuralHash, ExprDeepEqual>;

  // ------------------------------------------------------------------
  // Eligibility predicates
  // ------------------------------------------------------------------

  /*!
   * \brief Check if an expression node type is forbidden for CSE.
   *
   * Call nodes may have side effects. BufferLoad nodes depend on memory
   * state and cannot be safely hoisted or deduplicated.
   *
   * \param expr The expression to check.
   * \return true if the expression is a Call or BufferLoad.
   */
  static bool IsForbiddenNode(const PrimExpr& expr) {
    return (expr.as<CallNode>() != nullptr || expr.as<BufferLoadNode>() != nullptr);
  }

  /*!
   * \brief Check if an expression is eligible for common subexpression elimination.
   *
   * An expression is eligible if it represents a non-trivial pure computation:
   *   - Not a leaf (Var, IntImm, FloatImm, StringImm — no computation to save).
   *   - Not a Call or BufferLoad (side effects / memory dependence).
   *   - Not Ramp or Broadcast (hardware-specific vector construction).
   *   - Does not transitively contain any forbidden node.
   *
   * \param expr The expression to check.
   * \return true if the expression can participate in CSE.
   */
  static bool IsEligible(const PrimExpr& expr) {
    if (expr.as<IntImmNode>() || expr.as<FloatImmNode>() || expr.as<StringImmNode>() ||
        expr.as<VarNode>()) {
      return false;
    }
    if (IsForbiddenNode(expr)) return false;
    if (expr.as<RampNode>() || expr.as<BroadcastNode>()) return false;
    if (CheckContains::ExprContains(expr, IsForbiddenNode)) return false;
    return true;
  }

  // ------------------------------------------------------------------
  // Expression substitution
  // ------------------------------------------------------------------

  /*!
   * \brief Replace all occurrences of `target` in `body` with `replacement`.
   *
   * Uses structural equality (ExprDeepEqual) to find matches. Stops recursing
   * into a sub-tree once a match is found (the replacement is a leaf Var).
   *
   * \param body The expression to transform.
   * \param target The sub-expression to find.
   * \param replacement The expression to substitute in (typically a CSE Var).
   * \return The transformed expression.
   */
  static PrimExpr SubstituteSubexpr(const PrimExpr& body, const PrimExpr& target,
                                    const PrimExpr& replacement) {
    struct Replacer : public ExprMutator {
      ExprDeepEqual eq;
      PrimExpr target, replacement;
      PrimExpr VisitExpr(const PrimExpr& e) final {
        if (eq(e, target)) return replacement;
        return ExprMutator::VisitExpr(e);
      }
    };
    Replacer r;
    r.target = target;
    r.replacement = replacement;
    return r.VisitExpr(body);
  }

  // ------------------------------------------------------------------
  // Scope tree operations
  // ------------------------------------------------------------------

  /*!
   * \brief Allocate a new child scope in the scope tree.
   *
   * \param parent The parent scope ID.
   * \param creator_stmt The statement that creates this scope (e.g. ForNode).
   *                     Stored for later use as an insertion point.
   * \return The ID of the newly allocated scope.
   */
  int AllocScope(int parent, Stmt creator_stmt) {
    int id = static_cast<int>(scopes_.size());
    scopes_.push_back({parent, scopes_[parent].depth + 1, std::move(creator_stmt)});
    return id;
  }

  /*!
   * \brief Compute the Lowest Common Ancestor of two scope IDs.
   *
   * Walks both scopes upward to the same depth, then walks both upward
   * in lockstep until they meet. This is the standard LCA algorithm for
   * trees with parent pointers.
   *
   * \param a First scope ID.
   * \param b Second scope ID.
   * \return The scope ID of the LCA.
   */
  int LCA(int a, int b) const {
    while (scopes_[a].depth > scopes_[b].depth) a = scopes_[a].parent;
    while (scopes_[b].depth > scopes_[a].depth) b = scopes_[b].parent;
    while (a != b) {
      a = scopes_[a].parent;
      b = scopes_[b].parent;
    }
    return a;
  }

  /*!
   * \brief Find the statement to insert a CSE binding before.
   *
   * Two cases:
   *   - LCA == first-use scope: insert before the first_use_stmt directly.
   *   - LCA is an ancestor: walk from first_use_scope upward to find the
   *     scope-creating statement that is a direct child of the LCA scope,
   *     and insert before that statement.
   *
   * \param entry The expression entry containing scope and first-use metadata.
   * \return The statement before which the CSE Bind should be inserted.
   */
  Stmt FindInsertionStmt(const ExprEntry& entry) const {
    if (entry.first_use_scope == entry.lca_scope) {
      return entry.first_use_stmt;
    }
    int s = entry.first_use_scope;
    while (scopes_[s].parent != entry.lca_scope) s = scopes_[s].parent;
    return scopes_[s].creator_stmt;
  }

  // ------------------------------------------------------------------
  // Expression recording
  // ------------------------------------------------------------------

  /*!
   * \brief Record an occurrence of an expression in the expression table.
   *
   * On first occurrence: initializes the entry, records direct children
   * (AST children that are in the table), and computes expr_depth from
   * children. On subsequent occurrences: updates the LCA scope.
   *
   * \param e The expression to record.
   * \param ast_children The direct AST children of e (passed by the caller
   *                     who knows the node structure: op->a, op->b, etc.).
   */
  void RecordExpr(const PrimExpr& e, std::initializer_list<PrimExpr> ast_children) {
    if (!IsEligible(e)) return;
    // Inside a Let body, expressions may reference Let-bound variables that
    // are not visible at the statement level. Skip recording to prevent
    // extracting them before the containing statement.
    if (let_depth_ > 0) return;
    ExprEntry& entry = table_[e];
    bool is_first_occurrence = (entry.count == 0);
    if (is_first_occurrence) {
      entry.lca_scope = current_scope_;
      entry.first_use_scope = current_scope_;
      entry.first_use_stmt = current_stmt_;
      entry.repr = e;
      // Build DAG edges: check which AST children are eligible table entries.
      // Since we visit bottom-up, children are already in the table.
      CollectChildren(entry, ast_children);
    } else {
      // Widen the insertion scope to cover all occurrences.
      entry.lca_scope = LCA(entry.lca_scope, current_scope_);
    }
    entry.count += 1;
  }

  /*!
   * \brief Populate children and expr_depth for a newly created entry.
   *
   * Each AST child (e.g. op->a, op->b) that exists in the table becomes
   * a DAG child. Multiplicity tracks duplicates (e.g. `(x+y)*(x+y)` has
   * child `x+y` with multiplicity 2). expr_depth is 1 + max child depth.
   */
  void CollectChildren(ExprEntry& entry, std::initializer_list<PrimExpr> ast_children) {
    ExprDeepEqual eq;
    int max_child_depth = 0;
    for (const PrimExpr& child : ast_children) {
      auto it = table_.find(child);
      if (it == table_.end()) continue;
      max_child_depth = std::max(max_child_depth, it->second.expr_depth);
      // Check if this child was already seen (handles multiplicity).
      bool already_recorded = false;
      for (auto& [existing_child, multiplicity] : entry.children) {
        if (eq(existing_child, child)) {
          multiplicity++;
          already_recorded = true;
          break;
        }
      }
      if (!already_recorded) entry.children.push_back({child, 1});
    }
    entry.expr_depth = 1 + max_child_depth;
  }

  // ------------------------------------------------------------------
  // Visitor overrides — expressions
  // ------------------------------------------------------------------
  // Each arithmetic/comparison/logical/cast/select node visitor calls the
  // base class to recurse into children first, then records the full
  // expression. This bottom-up order ensures that sub-expressions are
  // recorded before their parents.
  // ------------------------------------------------------------------

  using StmtExprVisitor::VisitExpr_;

  // Binary arithmetic operators (op->a, op->b)
#define CSE_VISIT_BINARY(NodeType)                         \
  void VisitExpr_(const NodeType* op) override {           \
    StmtExprVisitor::VisitExpr_(op);                       \
    RecordExpr(ffi::GetRef<PrimExpr>(op), {op->a, op->b}); \
  }
  CSE_VISIT_BINARY(AddNode)
  CSE_VISIT_BINARY(SubNode)
  CSE_VISIT_BINARY(MulNode)
  CSE_VISIT_BINARY(DivNode)
  CSE_VISIT_BINARY(ModNode)
  CSE_VISIT_BINARY(FloorDivNode)
  CSE_VISIT_BINARY(FloorModNode)
  CSE_VISIT_BINARY(MinNode)
  CSE_VISIT_BINARY(MaxNode)
  CSE_VISIT_BINARY(EQNode)
  CSE_VISIT_BINARY(NENode)
  CSE_VISIT_BINARY(LTNode)
  CSE_VISIT_BINARY(LENode)
  CSE_VISIT_BINARY(GTNode)
  CSE_VISIT_BINARY(GENode)
  CSE_VISIT_BINARY(AndNode)
  CSE_VISIT_BINARY(OrNode)
#undef CSE_VISIT_BINARY

  void VisitExpr_(const NotNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op), {op->a});
  }
  void VisitExpr_(const CastNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op), {op->value});
  }
  void VisitExpr_(const SelectNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op), {op->condition, op->true_value, op->false_value});
  }

  /*!
   * \brief Let expressions bind variables visible only in their body.
   *
   * The value is visited normally (Let-bound var not yet in scope).
   * The body is visited with let_depth_ incremented to prevent CSE from
   * extracting expressions that may reference the Let-bound variable
   * to a position before the containing statement where it is undefined.
   */
  void VisitExpr_(const LetNode* op) override {
    VisitExpr(op->value);
    ++let_depth_;
    VisitExpr(op->body);
    --let_depth_;
  }

  // ------------------------------------------------------------------
  // Visitor overrides — statements
  // ------------------------------------------------------------------

  /*!
   * \brief Override VisitStmt to track current_stmt_ for insertion-point determination.
   *
   * Every VisitStmt call updates current_stmt_ before dispatching. This ensures
   * that RecordExpr always sees the innermost statement containing the expression,
   * whether it's a SeqStmt child, a for-loop body, or any other statement.
   */
  void VisitStmt(const Stmt& stmt) override {
    current_stmt_ = stmt;
    StmtExprVisitor::VisitStmt(stmt);
  }

  /*! \brief For loops: bounds in parent scope, body in child scope. */
  void VisitStmt_(const ForNode* op) override {
    VisitExpr(op->min);
    VisitExpr(op->extent);
    int saved = current_scope_;
    current_scope_ = AllocScope(saved, ffi::GetRef<Stmt>(op));
    VisitStmt(op->body);
    current_scope_ = saved;
  }

  /*!
   * \brief IfThenElse creates separate scopes for then/else branches.
   *
   * The condition is visited in the parent scope (so expressions shared
   * between the condition and a branch can be hoisted above the If).
   * Each branch gets its own scope so that expressions appearing in only
   * one branch are not hoisted above the If.
   */
  void VisitStmt_(const IfThenElseNode* op) override {
    VisitExpr(op->condition);
    int saved = current_scope_;
    Stmt stmt = ffi::GetRef<Stmt>(op);
    current_scope_ = AllocScope(saved, stmt);
    VisitStmt(op->then_case);
    if (op->else_case) {
      current_scope_ = AllocScope(saved, stmt);
      VisitStmt(op->else_case.value());
    }
    current_scope_ = saved;
  }

  /*! \brief While loops: condition in parent scope, body in child scope. */
  void VisitStmt_(const WhileNode* op) override {
    VisitExpr(op->condition);
    int saved = current_scope_;
    current_scope_ = AllocScope(saved, ffi::GetRef<Stmt>(op));
    VisitStmt(op->body);
    current_scope_ = saved;
  }

  /*! \brief AttrStmt: value in parent scope, body in child scope. */
  void VisitStmt_(const AttrStmtNode* op) override {
    VisitExpr(op->value);
    int saved = current_scope_;
    current_scope_ = AllocScope(saved, ffi::GetRef<Stmt>(op));
    VisitStmt(op->body);
    current_scope_ = saved;
  }

  /*! \brief AllocBuffer is flat (no body). Visit buffer shape expressions. */
  void VisitStmt_(const AllocBufferNode* op) override { VisitBufferDef(op->buffer, true); }

  /*! \brief DeclBuffer is flat (no body). Visit buffer shape expressions. */
  void VisitStmt_(const DeclBufferNode* op) override { VisitBufferDef(op->buffer, false); }

  // ------------------------------------------------------------------
  // ComputePlan: convert scan results into the output plan
  // ------------------------------------------------------------------

  /*!
   * \brief Convert the accumulated expression table into InsertBefore + ExprRemap tables.
   *
   * Algorithm (shallower-first with repr propagation):
   *   1. Collect all entries and sort by expr_depth ascending (shallower first),
   *      with structural hash as tie-breaker for determinism.
   *   2. Compute independent occurrence counts from the DAG children.
   *      For each parent P with count >= 2, its children's consumed counts
   *      are incremented by `(P.count - 1) * multiplicity` (the Bind value
   *      retains one copy). An entry with fewer than 2 independent occurrences
   *      is skipped (avoids unnecessary single-use bindings).
   *   3. For each entry with independent_count >= 2:
   *      a. Determine the insertion point.
   *      b. Create a CSE variable and Bind statement (using the entry's repr,
   *         which may already reference CSE vars from shallower entries).
   *      c. Add to insert_before and expr_remap.
   *      d. Propagate: replace this expression in all deeper entries' repr
   *         with the new CSE variable.
   *
   * \return A pair of (InsertBeforeTable, ExprRemapTable).
   */
  std::pair<InsertBeforeTable, ExprRemapTable> ComputePlan() {
    // Step 1: Sort entries by depth ascending (shallower first), hash for determinism
    std::vector<std::pair<PrimExpr, ExprEntry*>> all_entries;
    for (auto& kv : table_) {
      all_entries.push_back({kv.first, &kv.second});
    }

    std::stable_sort(
        all_entries.begin(), all_entries.end(),
        [](const std::pair<PrimExpr, ExprEntry*>& a, const std::pair<PrimExpr, ExprEntry*>& b) {
          if (a.second->expr_depth != b.second->expr_depth)
            return a.second->expr_depth < b.second->expr_depth;
          ffi::StructuralHash hasher;
          return hasher(a.first) < hasher(b.first);
        });

    // Step 2: Compute consumed counts in ExprEntry from the DAG.
    // For each parent P with count >= 2, each child's consumed count is
    // incremented by (P.count - 1) * multiplicity. Only direct children
    // are affected — no double-counting through grandparents.
    for (auto& [expr, entry] : all_entries) {
      if (entry->count < 2) continue;
      for (const auto& [child_expr, mult] : entry->children) {
        auto cit = table_.find(child_expr);
        if (cit != table_.end()) {
          cit->second.consumed += (entry->count - 1) * mult;
        }
      }
    }

    InsertBeforeTable insert_before;
    ExprRemapTable expr_remap;
    int counter = 0;

    // Step 3: Process each candidate (shallower first)
    for (auto& [expr, entry] : all_entries) {
      if (entry->count - entry->consumed < 2) continue;

      // Step 3a: Determine where to insert the Bind
      Stmt insert_at = FindInsertionStmt(*entry);

      // Step 3b: Create CSE variable and Bind statement.
      // entry->repr may already contain CSE vars from shallower entries.
      ++counter;
      std::string name = "cse_v" + std::to_string(counter);
      Var cse_var(name, entry->repr.dtype());
      Stmt bind = Bind(cse_var, entry->repr);

      // Step 3c: Record in output tables.
      // expr_remap maps the ORIGINAL expression (for tree matching by the rewriter).
      insert_before[insert_at].push_back(bind);
      expr_remap[expr] = cse_var;

      // Step 3d: Propagate into deeper entries' repr.
      // Replace occurrences of this entry's repr with cse_var so that
      // deeper Bind values reference the CSE variable instead of
      // recomputing the sub-expression.
      for (auto& [other_expr, other_entry] : all_entries) {
        if (other_entry->expr_depth <= entry->expr_depth) continue;
        other_entry->repr = SubstituteSubexpr(other_entry->repr, entry->repr, cse_var);
      }
    }

    return {insert_before, expr_remap};
  }

  // ------------------------------------------------------------------
  // State
  // ------------------------------------------------------------------
  /*! \brief The scope tree (indexed by scope ID). */
  std::vector<ScopeEntry> scopes_;
  /*! \brief Expression → metadata table. */
  ExprTable table_;
  /*! \brief Scope ID of the currently visited node. */
  int current_scope_ = 0;
  /*! \brief Current statement for insertion-point tracking. Set by VisitStmt. */
  Stmt current_stmt_;
  /*! \brief Nesting depth of Let expression bodies. When > 0, recording is suppressed. */
  int let_depth_ = 0;
};

// ============================================================================
// CSERewriter: Phase 2 — mechanical insertion + substitution
// ============================================================================

/*!
 * \brief Phase 2 of the two-phase CSE pass.
 *
 * CSERewriter is a StmtExprMutator that consumes the plan produced by
 * CSEPlanner and performs two kinds of edits:
 *   - **Insertion**: Before each statement listed in InsertBeforeTable,
 *     insert the planned `Bind(cse_var, expr)` statements.
 *   - **Substitution**: Replace every expression listed in ExprRemapTable
 *     with the corresponding CSE variable.
 *
 * Insertions are handled uniformly by overriding VisitStmt: when a
 * statement has insert_before entries, the visited statement is wrapped
 * in a SeqStmt with the Bind stmts prepended. SeqStmt's constructor
 * flattens nested SeqStmts, so this works correctly for both SeqStmt
 * children and direct bodies of scope-creating statements (For, If, etc.).
 */
class CSERewriter : public StmtExprMutator {
 public:
  /*!
   * \brief Construct a rewriter from the plan tables.
   * \param insert_before Map from stmt → list of Bind stmts to insert before it.
   * \param expr_remap Map from expression → CSE variable to substitute.
   */
  CSERewriter(InsertBeforeTable insert_before, ExprRemapTable expr_remap)
      : insert_before_(std::move(insert_before)), expr_remap_(std::move(expr_remap)) {}

  /*!
   * \brief Apply the rewrite to a function body.
   * \param body The original function body.
   * \return The rewritten body with CSE bindings inserted and expressions replaced.
   */
  Stmt Rewrite(const Stmt& body) { return VisitStmt(body); }

 protected:
  using StmtExprMutator::VisitExpr;
  using StmtExprMutator::VisitExpr_;

  /*!
   * \brief Visit an expression, replacing it with its CSE variable if planned.
   *
   * Checks the remap table before recursing — if the full expression matches,
   * it is replaced without visiting children.
   */
  PrimExpr VisitExpr(const PrimExpr& e) override {
    auto it = expr_remap_.find(e);
    if (it != expr_remap_.end()) return it->second;
    return StmtExprMutator::VisitExpr(e);
  }

  /*!
   * \brief Visit a statement, prepending planned Bind insertions.
   *
   * Looks up the original statement (by pointer identity) in insert_before_
   * before recursing. If insertions are planned, wraps the visited result
   * in a SeqStmt with the Bind statements prepended. SeqStmt flattening
   * ensures correct structure regardless of context.
   */
  Stmt VisitStmt(const Stmt& stmt) override {
    auto it = insert_before_.find(stmt);
    Stmt visited = StmtExprMutator::VisitStmt(stmt);
    if (it != insert_before_.end()) {
      ffi::Array<Stmt> new_stmts(it->second.begin(), it->second.end());
      new_stmts.push_back(visited);
      return SeqStmt(new_stmts);
    }
    return visited;
  }

 private:
  /*! \brief Plan: stmts to insert before each target. */
  InsertBeforeTable insert_before_;
  /*! \brief Plan: expressions to replace with CSE vars. */
  ExprRemapTable expr_remap_;
};

// ============================================================================
// Pass registration
// ============================================================================

namespace transform {

/*!
 * \brief Create the CommonSubexprElim pass.
 *
 * Plans all CSE opportunities in a single pass (shallower-first with repr
 * propagation), then rewrites the tree once.
 *
 * \return The pass.
 */
Pass CommonSubexprElim() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto [insert_before, expr_remap] = CSEPlanner::Plan(f->body);
    if (!insert_before.empty()) {
      auto* n = f.CopyOnWrite();
      n->body = CSERewriter(std::move(insert_before), std::move(expr_remap)).Rewrite(f->body);
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.CommonSubexprElim", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.CommonSubexprElim", CommonSubexprElim);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
