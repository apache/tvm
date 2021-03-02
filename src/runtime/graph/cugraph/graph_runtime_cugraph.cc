#include <tvm/runtime/container.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>


#include <chrono>

#include "../graph_runtime.h"

namespace tvm {
namespace runtime {

class GraphRuntimeCuGraph : public GraphRuntime {
  public:
    int StartCapture() {
      const TVMContext& ctx = data_entry_[entry_id(0, 0)]->ctx;

      TVMStreamCreate(ctx.device_type, ctx.device_id, &capture_stream_);

      TVMSetStream(ctx.device_type, ctx.device_id, capture_stream_);
      TVMStreamBeginCapture(ctx.device_type, ctx.device_id, capture_stream_);
      return 0;
    }

    int RunCudaGraph() {
      const TVMContext& ctx = data_entry_[entry_id(0, 0)]->ctx;
      TVMStreamRunCapture(ctx.device_type, ctx.device_id, capture_stream_, cu_graph_);
      return 0;
    }

    int EndCapture() {
      const TVMContext& ctx = data_entry_[entry_id(0, 0)]->ctx;
      // void * (cudaGraph_t *)
      TVMStreamEndCapture(ctx.device_type, ctx.device_id, capture_stream_, &cu_graph_);
      return 0;
    }

    /*!
     * \brief GetFunction Get the function based on input.
     * \param name The function which needs to be invoked.
     * \param sptr_to_self Packed function pointer.
    */
    PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);

  private:
    TVMObjectHandle cu_graph_;
    TVMStreamHandle capture_stream_;
};

PackedFunc GraphRuntimeCuGraph::GetFunction(const std::string& name,
                                            const ObjectPtr<Object>& sptr_to_self) {

  if (name == "run_cuda_graph") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->RunCudaGraph();
    });
  } else if (name == "start_capture") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->StartCapture();
    });
  } else if (name == "end_capture") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->EndCapture();
    });
  } else {
    return GraphRuntime::GetFunction(name, sptr_to_self);
  }
}

Module GraphRuntimeCuGraphCreate(const std::string& sym_json, const tvm::runtime::Module& m,
                                 const std::vector<TVMContext>& ctxs,
                                 PackedFunc lookup_linked_param_func) {
  auto exec = make_object<GraphRuntimeCuGraph>();
  exec->Init(sym_json, m, ctxs, lookup_linked_param_func);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.graph_runtime_cugraph.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  ICHECK_GE(args.num_args, 4) << "The expected number of arguments for graph_runtime.create is "
                                 "at least 4, but it has "
                              << args.num_args;
  PackedFunc lookup_linked_param_func;
  int ctx_start_arg = 2;
  if (args[2].type_code() == kTVMPackedFuncHandle) {
    lookup_linked_param_func = args[2];
    ctx_start_arg++;
  }

  *rv = GraphRuntimeCuGraphCreate(args[0], args[1], GetAllContext(args, ctx_start_arg),
                                  lookup_linked_param_func);
});

}
}
