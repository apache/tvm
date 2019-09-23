#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>

namespace tvm {
namespace ir {

class BoundedAnalyzer final : public IRVisitor {
 public:
  void Visit_(const For* op) {
    analyzer.Bind(op->loop_var,
                   Range::make_by_min_extent(op->min, op->extent));
    return IRVisitor::Visit_(op);
  }

  void Visit_(const AttrStmt* op) {
    if (op->attr_key == attr::thread_extent ||
        op->attr_key == attr::virtual_thread) {
      IterVar iv(op->node.node_);
      CHECK_NE(iv->thread_tag.length(), 0U);
      analyzer.Bind(iv->var,
                      Range::make_by_min_extent(0, op->value));
      IRVisitor::Visit_(op);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Reduce* op) {
    // Setup the domain information before simplification.
    for (const IterVar& iv : op->axis) {
      analyzer.Bind(iv->var, iv->dom);
    }
    // Recursively call simplification when necessary.
    IRVisitor::Visit_(op);
  }

  /*! \brief internal analyzer field. */
  arith::Analyzer analyzer;
};

}  // namespace ir
}  // namespace tvm
