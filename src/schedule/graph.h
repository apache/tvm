/*!
 *  Copyright (c) 2016 by Contributors
 * \file graph.h
 * \brief Utilities to get information about schedule graph.
 */
#ifndef TVM_SCHEDULE_GRAPH_H_
#define TVM_SCHEDULE_GRAPH_H_

#include <tvm/expr.h>
#include <tvm/schedule.h>
#include <tvm/operation.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace schedule {

/*!
 * \brief data structure of Operation->Tensors it reads
 */
using ReadGraph = Map<Operation, Array<Tensor> >;

/*!
 * \brief AttachPath maps op-> a list of IterVar
 */
using AttachPath = Map<Operation, Array<IterVar> >;

/*!
 * \brief Get read graph of each operation to all the
 *  Tensors that it directly depends on.
 *
 *  The result map contains Operations needed to finish root Operation.
 * \param roots The root operation.
 * \return The result map.
 */
ReadGraph CreateReadGraph(const Array<Operation>& roots);

/*!
 * \brief Get a post DFS ordered of operations in the graph.
 * \param roots The root of the graph.
 * \param g The read graph.
 * \return vector order of Operations in PostDFS order.
 *
 * \note PostDFSOrder is a special case of Topoligical order,
 *   and can be used when topoligical order is needed.
 */
Array<Operation> PostDFSOrder(
    const Array<Operation>& roots, const ReadGraph& g);

/*!
 * \brief Create feedgraph for given Schedule
 * \param  g The read graph.
 * \return The created feedgraph.
 */
FeedGraph CreateFeedGraph(const ReadGraph& g);

/*!
 * \brief Create AttachPath that  maps op-> a list of IterVar
 *  That represents the loop nest op sits in from inner most to outermost
 *  Also inserts attach_stage for scan updates when needed.
 *
 * \param sch The schedule.
 * \return The attach path.
 */
AttachPath CreateAttachPath(Schedule sch);

/*!
 * \brief Get all operations inside the recursion of scan.
 * \param scan The scan node.
 * \param feed_graph The feed graph to help analysis.
 * \return The body operations, in read dependency order.
 */
Array<Operation> ScanGetBody_(
    const ScanOpNode* scan, const FeedGraph& feed_graph);
// same as ScanGetBody_, but create FeedGraph internally.
Array<Operation> ScanGetBody(const Operation& scan);

/*!
 * \brief Analyze each spatial dimension of scan's result.
 *  Give check on whether each dimension is fix point,
 *  An axis is a fixed point if it only refers back to itself in recursion
 *  and it is not used in axis of other recursion field.
 *
 *  next_state[t, ..., axis, ...] = f(prev_state[t-1, ...,axis,...]
 *
 * \param scan The scan node.
 * \param body The body of scan, sorted in reverse PostDFSOrder.
 * \return Map of spatial_axis -> IntImm
 */
Map<IterVar, Expr> ScanFixPointAnalysis(
    const Operation& scan, const Array<Operation>& body);

}  // namespace schedule
}  // namespace tvm

#endif  // TVM_SCHEDULE_GRAPH_H_
