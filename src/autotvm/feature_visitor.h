/*!
 *  Copyright (c) 2018 by Contributors
 * \file feature_visitor.h
 * \brief Base class for feature extractor.
 *        These features are used for machine learning cost model
 */

#ifndef TVM_AUTOTVM_FEATURE_VISITOR_H_
#define TVM_AUTOTVM_FEATURE_VISITOR_H_

#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <string>

namespace tvm {
namespace autotvm {

using namespace tvm::ir;

/*!
 * \brief Type of for loop, used as one-hot encoding in features
 */
enum AnnotationType {
  kBlockX, kBlockY, kBlockZ, kThreadX, kThreadY, kThreadZ,
  kUnrolled, kVectorized, kParallel, kSerial, kVirtualThread,
  kNum,
};

/*!
 * \brief A base class for feature extractor, used for processing
 * for loop and memory access in the IR
 */
class FeatureVisitor : public IRVisitor {
 public:
  // for loop
  void Visit_(const For *op);
  void Visit_(const AttrStmt *op);

  // memory access
  void Visit_(const Load *op);
  void Visit_(const Store *op);

 protected:
  /*!
 * \brief Enter a for loop node
 * \param var The expression to be printed.
 * \param length The output stream
 * \param ann_type The type for the for loop
 * \return skip Whether skip this node
 */
  virtual bool EnterItervar_(tvm::VarExpr var, int64_t length, AnnotationType ann_type) = 0;
  /*! \brief Exit a for loop subtree */
  virtual void ExitItervar_() = 0;
  /*!
   * \brief Enter a memory access node
   * \param buffer_var The buffer to access.
   * \param index Index expression
   */
  virtual void EnterMem_(tvm::VarExpr buffer_var, tvm::Expr index) = 0;
  /*! \brief Exit a memory access node */
  virtual void ExitMem_() = 0;
};

}  // namespace autotvm
}  // namespace tvm

#endif  // TVM_AUTOTVM_FEATURE_VISITOR_H_
