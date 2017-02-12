/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir_visitor.h
 * \brief Visitor to quickly visit IR trees
 */
#ifndef TVM_IR_VISITOR_H_
#define TVM_IR_VISITOR_H_

#include "./ir.h"

namespace tvm {
namespace ir {

/*!
 * \brief a base class for visitor to iterative traverse the IR
 *
 *  This IRVisitor is implemented via IRFunctor
 *  This enables extensions of possible new Node.
 *
 * \sa IRFunctor, PostOrderVisit
 */
class IRVisitor {
 public:
  /*!
   * \brief recursively visit an IR node
   */
  virtual void Visit(const NodeRef& node) {
    static const FVisit& f = vtable();
    if (node.defined()) f(node, this);
  }
  /*! \brief destructor */
  virtual ~IRVisitor() {}
  /*! \brief functor type of visitor */
  using FVisit = IRFunctor<void(const NodeRef&, IRVisitor*)>;
  /*! \return internal vtable*/
  static FVisit& vtable();
  // overloadable visit function.
  virtual void Visit_(const Variable* op);
  virtual void Visit_(const AttrStmt* op);
  virtual void Visit_(const LetStmt* op);
  virtual void Visit_(const For* op);
  virtual void Visit_(const Allocate* op);
  virtual void Visit_(const IfThenElse* op);
  virtual void Visit_(const Load* op);
  virtual void Visit_(const Store* op);
  virtual void Visit_(const Let* op);
  virtual void Visit_(const Free* op);
  virtual void Visit_(const Call* op);
  virtual void Visit_(const Add* op);
  virtual void Visit_(const Sub* op);
  virtual void Visit_(const Mul* op);
  virtual void Visit_(const Div* op);
  virtual void Visit_(const Mod* op);
  virtual void Visit_(const Min* op);
  virtual void Visit_(const Max* op);
  virtual void Visit_(const EQ* op);
  virtual void Visit_(const NE* op);
  virtual void Visit_(const LT* op);
  virtual void Visit_(const LE* op);
  virtual void Visit_(const GT* op);
  virtual void Visit_(const GE* op);
  virtual void Visit_(const And* op);
  virtual void Visit_(const Or* op);
  virtual void Visit_(const Reduce* op);
  virtual void Visit_(const Cast* op);
  virtual void Visit_(const Not* op);
  virtual void Visit_(const Select* op);
  virtual void Visit_(const Ramp* op);
  virtual void Visit_(const Broadcast* op);
  virtual void Visit_(const AssertStmt* op);
  virtual void Visit_(const ProducerConsumer* op);
  virtual void Visit_(const Provide* op);
  virtual void Visit_(const Realize* op);
  virtual void Visit_(const Block* op);
  virtual void Visit_(const Evaluate* op);
  virtual void Visit_(const IntImm* op);
  virtual void Visit_(const UIntImm* op);
  virtual void Visit_(const FloatImm* op);
  virtual void Visit_(const StringImm* op);
};

/*!
 * \brief recursively visit the ir in post DFS order node, apply fvisit
 * Each node is ganranteed to be visited only once.
 * \param node The ir to be visited.
 * \param fvisit The visitor function to be applied.
 */
void PostOrderVisit(const NodeRef& node, std::function<void(const NodeRef&)> fvisit);

}  // namespace ir
}  // namespace tvm

#endif  // TVM_IR_VISITOR_H_
