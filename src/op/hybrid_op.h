#ifndef TVM_HYBRID_OP_H
#define TVM_HYBRID_OP_H

#include <tvm/expr.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/schedule.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../pass/ir_util.h"
#include "../pass/arg_binder.h"
#include "../schedule/message_passing.h"


namespace tvm {
namespace op {

/*!
 * \brief Find all the iteration variables in the given statement body.
 * \param stmt The body to be inspected.
 */
std::vector<IterVar> GatherLoopVars(Stmt stmt);

/*!
 * \brief Replace the tensor reference (especially in Provide's) in stmt by the replace map.
 * \param stmt The statement to be processed.
 * \param replace The replacement rule.
 */
Stmt ReplaceProvideTensor(Stmt stmt,
                   const std::unordered_map<Tensor, Tensor>& replace);

} // namespace op
} // namespace tvm

#endif // TVM_HYBRID_OP_H
