/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_subst.cc
 * \brief Function for substituting a concrete type in place of a type ID
 */
#include "./type_visitor.h"

namespace tvm {
namespace relay {

struct Bad : TypeVisitor<std::shared_ptr<int> &&> {
  void VisitType_(const TensorTypeNode* t, std::shared_ptr<int> && s) final {
    std::shared_ptr<int> sptr(std::forward<std::shared_ptr<int> &&>(s));
    std::cout << sptr.use_count() << ":" << *sptr << std::endl;
  }
};

TVM_REGISTER_API("relay._ir_pass.bad")
  .set_body([](TVMArgs args, TVMRetValue *ret) {
      Type t = args[0];
      Bad()(t, std::shared_ptr<int>(new int(123)));
      CHECK(false);
    });


}  // namespace relay
}  // namespace tvm
