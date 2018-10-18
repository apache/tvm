/*!
 *  Copyright (c) 2018 by Contributors
 * \file attr_functor.h
 * \brief A way to define arbitrary function signature
 *        with dispatch on common attributes.
 *
 * Common attributes include:
 *  - int, float, str constants
 *  - array of attributes
 *  - map of attributes
 */
#ifndef TVM_LANG_ATTR_FUNCTOR_H_
#define TVM_LANG_ATTR_FUNCTOR_H_

namespace tvm {

template <typename FType>
class AttrFunctor;

#define ATTR_FUNCTOR_DEFAULT                                        \
  { return VisitAttrDefault_(op, std::forward<Args>(args)...); }


#define ATTR_FUNCTOR_DISPATCH(OP)                                       \
  vtable.template set_dispatch<OP>(                                     \
      [](const NodeRef& n, TSelf* self, Args... args) {                 \
        return self->VisitAttr_(static_cast<const OP*>(n.node_.get()),  \
                                std::forward<Args>(args)...);           \
      });                                                               \

// A functor for common attribute information.
template <typename R, typename... Args>
class AttrFunctor<R(const NodeRef& n, Args...)> {
 private:
  using TSelf = AttrFunctor<R(const NodeRef& n, Args...)>;
  using FType = tvm::IRFunctor<R(const NodeRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitAttr(const NodeRef& n, Args... args) {
    static FType vtable = InitVTable();
    if (vtable.can_dispatch(n)) {
      return vtable(n, this, std::forward<Args>(args)...);
    } else {
      return VisitAttrDefault_(n.get(), std::forward<Args>(args)...);
    }
  }
  virtual R VisitAttr_(const ArrayNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const StrMapNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::IntImm* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::UIntImm* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::FloatImm* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::StringImm* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttrDefault_(const Node* node, Args... args) = 0;

 private:
  // initialize the vtable.
  static FType InitVTable() {
    using namespace ir;
    FType vtable;
    // Set dispatch
    ATTR_FUNCTOR_DISPATCH(StrMapNode);
    ATTR_FUNCTOR_DISPATCH(ArrayNode);
    ATTR_FUNCTOR_DISPATCH(IntImm);
    ATTR_FUNCTOR_DISPATCH(UIntImm);
    ATTR_FUNCTOR_DISPATCH(FloatImm);
    ATTR_FUNCTOR_DISPATCH(StringImm);
    return vtable;
  }
};

}  // namespace tvm
#endif  // TVM_LANG_ATTR_FUNCTOR_H_
