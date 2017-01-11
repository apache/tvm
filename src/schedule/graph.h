/*!
 *  Copyright (c) 2016 by Contributors
 * \file graph.h
 * \brief Utilities to get information about schedule graph.
 */
#ifndef TVM_SCHEDULE_GRAPH_H_
#define TVM_SCHEDULE_GRAPH_H_

#include <tvm/expr.h>
#include <tvm/schedule.h>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace schedule {

/*!
 * \brief data structure of Operation->Tensors it reads
 */
using ReadGraph = Map<Operation, Array<Tensor> >;

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

}  // namespace schedule
}  // namespace tvm

#endif  // TVM_SCHEDULE_GRAPH_H_
