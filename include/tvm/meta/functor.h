#ifndef TVM_META_FUNCTOR_H_
#define TVM_META_FUNCTOR_H_
#include <tvm/meta/expr.h>

namespace tvm {
namespace meta {

template <typename FType>
class ExprFunctor;

#define META_EXPR_FUNCTOR_DISPATCH(NODE)                                                        \
  vtable.template set_dispatch<NODE>([](const ObjectRef& n, TSelf* self, Args... args) {        \
    return self->VisitMetaIR_(static_cast<const NODE*>(n.get()), std::forward<Args>(args)...);  \
  });

template <typename R, typename... Args>
class ExprFunctor<R(const MetaIR& n, Args...)> {
 private:
  using TSelf = ExprFunctor<R(const MetaIR& n, Args...)>;
  using FType = tvm::NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  virtual ~ExprFunctor() {}

  R operator()(const MetaIR& n, Args... args) {
    return VisitMetaIR(n, std::forward<Args>(args)...);
  }

  virtual R VisitMetaIR(const MetaIR& n, Args... args) {
    CHECK(n.defined());
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }

  virtual R VisitMetaIR_(const VarDefNode* n, Args... args) = 0;
  virtual R VisitMetaIR_(const ObjectDefNode* n, Args... args) = 0;

 private:
  static FType InitVTable() {
    FType vtable;
    META_EXPR_FUNCTOR_DISPATCH(VarDefNode);
    META_EXPR_FUNCTOR_DISPATCH(ObjectDefNode);
    return vtable;
  }
};

}  // namespace meta
}  // namespace tvm
#endif  // TVM_META_FUNCTOR_H_
