/*!
 *  Copyright (c) 2017 by Contributors
 * \file packed_func_ext.cc
 * \brief Registeration of extension type.
 */
#include <tvm/expr.h>
#include <tvm/packed_func_ext.h>
#include <nnvm/op.h>
#include <nnvm/compiler/packed_func_ext.h>
#include <nnvm/compiler/op_attr_types.h>
#include "./node_attr.h"
#include "compile_engine.h"

namespace tvm {
namespace runtime {

TVM_REGISTER_EXT_TYPE(nnvm::Graph);
TVM_REGISTER_EXT_TYPE(nnvm::Symbol);
TVM_REGISTER_EXT_TYPE(nnvm::compiler::AttrDict);

}  // namespace runtime
}  // namespace tvm

namespace nnvm {
DMLC_JSON_ENABLE_ANY(int, int);
}  // namespace nnvm

namespace nnvm {
namespace compiler {

using tvm::Tensor;
using tvm::Array;
using tvm::Node;
using tvm::runtime::TVMArgs;
using tvm::runtime::TVMRetValue;

TVM_REGISTER_GLOBAL("nnvm.compiler._dict_get")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    const AttrDict& dict = args[0].AsExtension<AttrDict>();
    std::string key = args[1];
    auto it = dict.find(key);
    if (it != dict.end()) {
      *rv = it->second;
    } else {
      *rv = nullptr;
    }
  });

TVM_REGISTER_GLOBAL("nnvm.compiler._dict_size")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    const AttrDict& dict = args[0].AsExtension<AttrDict>();
    *rv = static_cast<int64_t>(dict.size());
  });

TVM_REGISTER_GLOBAL("nnvm.compiler._dict_keys")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    const AttrDict& dict = args[0].AsExtension<AttrDict>();
    tvm::Array<tvm::Expr> keys;
    for (const auto& kv : dict) {
      keys.push_back(kv.first);
    }
    *rv = keys;
  });

// custom version of TVM compute
TVM_REGISTER_GLOBAL("nnvm._register_compute")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    // Intentionally copy and not de-allocate it, to avoid free pyobject during shutdown
    PackedFunc* f = new PackedFunc(args[1].operator PackedFunc());
    Op& op = ::dmlc::Registry<nnvm::Op>::Get()->__REGISTER_OR_GET__(args[0]);
    auto fcompute = [f](const NodeAttrs& attrs,
                        const Array<Tensor>& inputs,
                        const Array<Tensor>& out_info)
        -> Array<Tensor> {
      TVMRetValue ret = (*f)(GetAttrDict(attrs), inputs, out_info);
      if ((*ret.ptr<std::shared_ptr<tvm::Node> >())->derived_from<tvm::TensorNode>()) {
        return {ret.operator Tensor()};
      } else {
        return ret;
      }
    };
    op.set_attr<FTVMCompute>("FTVMCompute", fcompute, args[2]);
  });

TVM_REGISTER_GLOBAL("nnvm._register_schedule")
.set_body([](TVMArgs args, TVMRetValue *rv) {
        // Intentionally copy and not de-allocate it, to avoid free pyobject during shutdown
    PackedFunc* f = new PackedFunc(args[1].operator PackedFunc());
    Op& op = ::dmlc::Registry<nnvm::Op>::Get()->__REGISTER_OR_GET__(args[0]);
    auto fschedule = [f](const NodeAttrs& attrs,
                         const Array<Tensor>& outs,
                         const std::string& target) {
      return (*f)(GetAttrDict(attrs), outs, target).operator Schedule();
    };
    op.set_attr<FTVMSchedule>("FTVMSchedule", fschedule, args[2]);
  });

TVM_REGISTER_GLOBAL("nnvm._register_pattern")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    Op& op = ::dmlc::Registry<nnvm::Op>::Get()->__REGISTER_OR_GET__(args[0]);
    op.set_attr<TOpPattern>("TOpPattern", args[1].operator int(), args[2]);
  });

TVM_REGISTER_GLOBAL("nnvm.graph._move_module")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    const nnvm::Graph& g = args[0].AsExtension<Graph>();
    *rv = const_cast<nnvm::Graph*>(&g)->
        MoveCopyAttr<tvm::runtime::Module>(args[1]);
  });

TVM_REGISTER_GLOBAL("nnvm.graph._move_graph")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    const nnvm::Graph& g = args[0].AsExtension<Graph>();
    std::string key = args[1];
    if (g.attrs.count(key)) {
      *rv = const_cast<nnvm::Graph*>(&g)->
          MoveCopyAttr<nnvm::Graph>(key);
    } else {
      *rv = nullptr;
    }
  });
}  // namespace compiler
}  // namespace nnvm
