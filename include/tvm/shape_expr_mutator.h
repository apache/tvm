#include <tvm/expr.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>

namespace tvm {
namespace ir {

class IndexVarFinder final : public IRVisitor {
 public:
  void Visit_(const Variable* op) final {
    if (!var_const_map_.count(op)) {
      var_const_map_[op] = 42;
    }
  }
  const std::unordered_map<const Variable*, int64_t>& var_map() {
    return var_const_map_;
  }
 private:
  std::unordered_map<const Variable*, int64_t> var_const_map_;
};

class IndexVarReplacer final : public IRMutator {
 public:
  Expr Mutate_(const Variable* op, const Expr& e) final {
    if (var_const_map_.count(op)) {
      return Expr(IntImm::make(Int(32), var_const_map_[op]));
    }
    return e;
  }
  void Init(const std::unordered_map<const Variable*, int64_t>& var_map) {
    var_const_map_ = var_map;
  }
 private:
  std::unordered_map<const Variable*, int64_t> var_const_map_;
};

} // namespace ir
} // namespace tvm
