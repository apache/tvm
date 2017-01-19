/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions
 * \file c_api_impl.cc
 */
#include <tvm/expr.h>
#include <tvm/tensor.h>
#include "./c_api_registry.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::tvm::APIFunctionReg);
}  // namespace dmlc

namespace tvm {

using ArgStack = const std::vector<APIVariantValue>;
using RetValue = APIVariantValue;

TVM_REGISTER_API(_format_str)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK(args.at(0).type_id == kNodeHandle);
    std::ostringstream os;
    os << args.at(0).operator NodeRef();
    *ret = os.str();
  })
.add_argument("expr", "Node", "expression to be printed");

TVM_REGISTER_API(_raw_ptr)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK(args.at(0).type_id == kNodeHandle);
    *ret = reinterpret_cast<int64_t>(args.at(0).sptr.get());
  })
.add_argument("src", "NodeBase", "the node base");

TVM_REGISTER_API(_save_json)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = SaveJSON(args.at(0));
  })
.add_argument("src", "json_str", "the node ");

TVM_REGISTER_API(_load_json)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = NodeRef(LoadJSON_(args.at(0)));
  })
.add_argument("src", "NodeBase", "the node");

}  // namespace tvm
