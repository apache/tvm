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
 *  This enables extensions of possible new IRNode.
 *
 * \sa IRFunctor, PostOrderVisit
 */
class IRVisitor {
 public:
  /*!
   * \brief recursively visit an IR node
   */
  virtual void Visit(const IRNodeRef& node) {
    static const FVisit& f = vtable();
    if (node.defined()) f(node, this);
  }
  /*! \brief destructor */
  virtual ~IRVisitor() {}
  /*! \brief functor type of visitor */
  using FVisit = IRFunctor<void(const IRNodeRef&, IRVisitor*)>;
  /*! \return internal vtable*/
  static FVisit& vtable();
};

/*!
 * \brief recursively visit the ir in post DFS order node, apply fvisit
 * Each node is ganranteed to be visited only once.
 * \param node The ir to be visited.
 * \param fvisit The visitor function to be applied.
 */
void PostOrderVisit(const IRNodeRef& node, std::function<void(const IRNodeRef&)> fvisit);

}  // namespace ir
}  // namespace tvm

#endif  // TVM_IR_VISITOR_H_
