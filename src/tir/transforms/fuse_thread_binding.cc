#include <tvm/arith/analyzer.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <unordered_map>

namespace tvm {
namespace tir {

/* 阶段1：分析线程绑定的最大范围 */
class ThreadBindingAnalyzer : public StmtVisitor {
 public:
  // 存储线程标签到最大范围的映射
  std::unordered_map<std::string, PrimExpr> max_extent_;
  bool multiple_repeat_thread_ = false;

  void VisitStmt_(const AttrStmtNode* op) override {
    if (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread) {
      HandleThreadBinding(Downcast<IterVar>(op->node), op->value);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode* op) override {
    if (op->kind == ForKind::kThreadBinding) {
      HandleThreadBinding(op->thread_binding.value(), op->extent);
    }
    StmtVisitor::VisitStmt_(op);
  }

 private:
  void HandleThreadBinding(const IterVar& iter_var, const PrimExpr& extent) {
    const std::string& tag = iter_var->thread_tag;
    arith::Analyzer ana;

    // 更新最大范围
    auto it = max_extent_.find(tag);
    if (it == max_extent_.end()) {
      max_extent_[tag] = extent;
    } else {
      multiple_repeat_thread_ = true;
      if (ana.CanProve(extent > it->second)) {
        max_extent_[tag] = extent;
      }
    }
  }
};

/* 阶段2：统一线程绑定并插入条件 */
class ThreadBindingUnifier : public StmtExprMutator {
 public:
  explicit ThreadBindingUnifier(const std::unordered_map<std::string, PrimExpr>& max_extent)
      : max_extent_(max_extent) {}

  static Stmt Unify(Stmt stmt, const std::unordered_map<std::string, PrimExpr>& max_extent) {
    ThreadBindingUnifier unifier(max_extent);
    Stmt new_stmt = unifier(std::move(stmt));
    return unifier.EmitLaunchThreads(new_stmt);
  }

 private:
  // 存储分析阶段的结果
  const std::unordered_map<std::string, PrimExpr>& max_extent_;
  // 变量替换映射（旧变量 -> 新变量）
  Map<Var, PrimExpr> var_substitution_map_;
  // 条件栈（处理嵌套条件）
  std::vector<PrimExpr> cond_stack_;
  /*!
   * \brief A mapping from a thread tag to its corresponding IterVar that is shared by all
   * occurrences of the thread tag
   */
  Map<String, IterVar> thread_tag2iter_var_map_;
  /*!
   * \brief A list of IterVar corresponding to threads in current kernel. This will be used to
   * generate for-loops to launch threads.
   */
  Array<IterVar> launch_threads_;

  Stmt VisitStmt_(const AttrStmtNode* op) override {
    if (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread) {
      IterVar old_iter_var = Downcast<IterVar>(op->node);
      return HandleBinding(op, old_iter_var->var, old_iter_var,
                           Range::FromMinExtent(IntImm(op->value->dtype, 0), op->value));
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode* op) override {
    if (op->kind == ForKind::kThreadBinding) {
      return HandleBinding(op, op->loop_var, op->thread_binding.value(),
                           Range::FromMinExtent(op->min, op->extent));
    }
    return StmtMutator::VisitStmt_(op);
  }

  template <typename Node>
  Stmt HandleBinding(const Node* op, const Var& old_var, const IterVar& old_iter_var,
                     const Range& dom) {
    // Step 1. Fetch the thread tag.
    IterVar new_iter_var{nullptr};
    const std::string& thread_tag = old_iter_var->thread_tag;
    // 获取统一后的范围
    auto it_extent = max_extent_.find(thread_tag);
    CHECK(it_extent != max_extent_.end()) << "Missing analysis for thread tag: " << thread_tag;
    const PrimExpr& unified_extent = it_extent->second;

    Map<String, IterVar>::iterator it = thread_tag2iter_var_map_.find(thread_tag);
    if (it != thread_tag2iter_var_map_.end()) {
      new_iter_var = (*it).second;
    } else {
      new_iter_var = IterVar(dom, Var(thread_tag, dom->extent.dtype()), old_iter_var->iter_type,
                             old_iter_var->thread_tag);
      thread_tag2iter_var_map_.Set(thread_tag, new_iter_var);
      launch_threads_.push_back(new_iter_var);
    }

    // 创建新变量
    var_substitution_map_.Set(old_var, cast(old_var.dtype(), new_iter_var->var));

    // 生成条件判断
    arith::Analyzer ana;
    PrimExpr condition;
    if (!ana.CanProveEqual(dom->extent, unified_extent)) {
      condition = (new_iter_var->var < dom->extent);
      cond_stack_.push_back(condition);
    }
    // 处理子节点
    Stmt body = StmtMutator::VisitStmt(op->body);

    // 包裹条件语句
    if (condition.defined()) {
      body = IfThenElse(CombineConditions(), body, Stmt());
    }

    // 创建统一后的循环
    return body;
  }

  PrimExpr VisitExpr_(const VarNode* var) final {
    // If this variable appears as a key in `var_substitution_map_`, we substitute it with its
    // corresponding value in the mapping.
    Map<Var, PrimExpr>::iterator it = var_substitution_map_.find(GetRef<Var>(var));
    return it != var_substitution_map_.end() ? (*it).second : GetRef<Var>(var);
  }

  PrimExpr CombineConditions() {
    if (cond_stack_.empty()) return Bool(true);
    PrimExpr res = cond_stack_[0];
    for (size_t i = 1; i < cond_stack_.size(); ++i) {
      res = res && cond_stack_[i];
    }
    return res;
  }

  /*!
   * \brief Emit loop nests representing all thread bindings of the kernel
   * \param body The body of the innermost loop of the thread bindings.
   * \return The loop nests of the thread bindings.
   */
  Stmt EmitLaunchThreads(const Stmt& body) {
    Stmt result = body;
    while (!launch_threads_.empty()) {
      const IterVar& thread_binding = launch_threads_.back();
      const std::string& thread_tag = thread_binding->thread_tag;
      // 获取统一后的范围
      auto it_extent = max_extent_.find(thread_tag);
      CHECK(it_extent != max_extent_.end()) << "Missing analysis for thread tag: " << thread_tag;
      const PrimExpr& unified_extent = it_extent->second;
      // Recreate the IterVar as we don't duplicate `dom` in both For and IterVar. This is
      // necessary for unit tests.
      result = For(thread_binding->var, thread_binding->dom->min, unified_extent,
                   ForKind::kThreadBinding, result,
                   IterVar(NullValue<Range>(), Var(""), IterVarType::kThreadIndex,
                           thread_binding->thread_tag));
      launch_threads_.pop_back();
    }
    return result;
  }
};

/* 主转换函数 */
PrimFunc FuseThreadBindings(PrimFunc f) {
  // 阶段1：分析最大范围
  ThreadBindingAnalyzer analyzer;
  analyzer(f->body);
  if (!analyzer.multiple_repeat_thread_) {
    return f;
  }
  // 阶段2：应用转换
  PrimFuncNode* fptr = f.CopyOnWrite();
  fptr->body = ThreadBindingUnifier::Unify(std::move(f->body), analyzer.max_extent_);
  return f;
}

namespace transform {
Pass FuseThreadBindings() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return FuseThreadBindings(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.FuseThreadBindings", {});
}

TVM_REGISTER_GLOBAL("tir.transform.FuseThreadBindings").set_body_typed(FuseThreadBindings);

}  // namespace transform
}  // namespace tir
}  // namespace tvm