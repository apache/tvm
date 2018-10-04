/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph_runtime_debug.cc
 */
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/ndarray.h>
#include <chrono>
#include "../graph_runtime.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Graph runtime with debug .
 *
 *  This is the extension of GraphRuntime class used for debugging
 *  TVM runtime PackedFunc API.
 */
class GraphRuntimeDebug : public GraphRuntime {
 public:
  /*!
   * \brief Run each operation and get the output.
   * \param index The index of op which needs to be run.
   * \return the elapsed time.
   */
  double DebugRun(size_t index) {
    CHECK(index < op_execs().size());
    TVMContext ctx = data_entry()[GetEntryId(index, 0)].operator->()->ctx;
    auto tbegin = std::chrono::high_resolution_clock::now();
    if (op_execs()[index]) {
      op_execs()[index]();
    }
    TVMSynchronize(ctx.device_type, ctx.device_id, nullptr);
    auto tend = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double> >(
        tend - tbegin).count();
    return time;
  }

  /*!
   * \brief Run each operation and get the output.
   * \param index The index of op which needs to be returned.
   * \param eid The Entry id of the op.
   */
  NDArray GetOutputByLayer(int index, int eid) {
    return data_entry()[GetEntryId(index, eid)];
  }

  /*!
   * \brief GetFunction Get the function based on input.
   * \param name The function which needs to be invoked.
   * \param sptr_to_self Packed function pointer.
   */
  PackedFunc GetFunction(const std::string& name,
                         const std::shared_ptr<ModuleNode>& sptr_to_self);

  /*!
   * \brief Get the node index given the name of node.
   * \param name The name of the node.
   * \return The index of node.
   */
  int GetNodeIndex(const std::string& name) const {
    for (size_t nid = 0; nid < GetNumOfNodes(); ++nid) {
      if (GetNodeName(nid) == name) {
        return static_cast<int>(nid);
      }
    }
    LOG(FATAL) << "cannot find " << name << " among nodex";
    return -1;
}

/*!
 * \brief Copy index-th node to data_out.
 *
 * This method will do a partial run of the the graph
 * from begining upto the index-th node and return output of index-th node.
 * This is costly operation and suggest to use only for debug porpose.
 *
 * \param index: The  index of the node.
 * \param data_out the node data.
 */
void DebugGetNodeOutput(int index, DLTensor* data_out) {
  CHECK_LT(static_cast<size_t>(index), op_execs().size());
  uint32_t eid = index;

  for (size_t i = 0; i < op_execs().size(); ++i) {
    if (op_execs()[i]) op_execs()[i]();
    if (static_cast<int>(i) == index) break;
  }

  data_entry()[eid].CopyTo(data_out);
}
};


/*!
 * \brief GetFunction Get the function based on input.
 * \param name The function which needs to be invoked.
 * \param sptr_to_self Packed function pointer.
 */
PackedFunc GraphRuntimeDebug::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  // return member functions during query.
  if (name == "debug_run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->DebugRun(static_cast<size_t>(args[0].operator int64_t()));
      });
  } else if (name == "get_output_by_layer") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->GetOutputByLayer(args[0], args[1]);
      });
  } else if (name == "debug_get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        if (args[0].type_code() == kStr) {
          this->DebugGetNodeOutput(this->GetNodeIndex(args[0]), args[1]);
        } else {
          this->DebugGetNodeOutput(args[0], args[1]);
        }
      });
  } else {
    return GraphRuntime::GetFunction(name, sptr_to_self);
  }
}

/*!
 * \brief GraphRuntimeDebugCreate Get the function based on input.
 * \param sym_json The graph symbol in json format.
 * \param m Compiled module which will be loaded.
 * \param ctxs All devices contexts.
 */
Module GraphRuntimeDebugCreate(const std::string& sym_json,
                               const tvm::runtime::Module& m,
                               const std::vector<TVMContext>& ctxs) {
  std::shared_ptr<GraphRuntimeDebug> exec = std::make_shared<GraphRuntimeDebug>();
  exec->Init(sym_json, m, ctxs);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.graph_runtime_debug.create")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    CHECK_GE(args.num_args, 4)
        << "The expected number of arguments for graph_runtime.create is "
           "at least 4, but it has "
        << args.num_args;
    *rv = GraphRuntimeDebugCreate(args[0], args[1], GetAllContext(args));
  });
}  // namespace runtime
}  // namespace tvm
