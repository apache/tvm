#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>

// TODO(gus) how to do these imports correctly?
#include "../codegen/datatype/datatype_registry.h"

namespace tvm {
namespace ir {

/*!
 * \brief Helper mutator to implement lowering of custom datatypes.
 *
 * Lowering datatypes works as follows: for every expression containing a custom
 * datatype, we search for a global (registered by the implementer of the custom
 * datatype) for lowering this type of expression, and uses it to lower the
 * expression.
 */
class DatatypesLowerer : public IRMutator {
 public:
  DatatypesLowerer(const std::string& target) : target_(target) {}

  inline Expr Mutate_(const Cast* op, const Expr& e) final {
    auto type_code = op->type.code();
    auto src_type_code = op->value.type().code();
    // If either datatype is a registered custom datatype, we must lower.
    if (DatatypeRegistry::Global()->DatatypeRegistered(type_code) ||
        DatatypeRegistry::Global()->DatatypeRegistered(src_type_code)) {
      auto lower = GetCastLowerFunc(target_, type_code, src_type_code);
      internal_assert(lower);
      // TODO(gus) they use this->Mutate; why?
      Expr r = (*lower)(e);
      return Mutate(r);
    }
    return e;
  }

#define DEFINE_MUTATE__(OP)                                          \
  inline Expr Mutate_(const OP* op, const Expr& e) final {           \
    auto type_code = op->type.code();                                \
    if (DatatypeRegistry::Global()->DatatypeRegistered(type_code)) { \
      auto lower = Get##OP##LowerFunc(target_, type_code);           \
      internal_assert(lower);                                        \
      Expr r = (*lower)(e);                                          \
      return Mutate(r);                                              \
    }                                                                \
    return e;                                                        \
  }

  // TODO(gus) this list should be the same as the list of
  // DEFINE_GET_LOWER_FUNC_ in datatypes_registry.h. We should avoid the
  // duplication.
  DEFINE_MUTATE__(Add)
  DEFINE_MUTATE__(Sub)
  DEFINE_MUTATE__(Mul)
  DEFINE_MUTATE__(Div)
  DEFINE_MUTATE__(Mod)
  DEFINE_MUTATE__(Min)
  DEFINE_MUTATE__(Max)
  DEFINE_MUTATE__(EQ)
  DEFINE_MUTATE__(NE)
  DEFINE_MUTATE__(LT)
  DEFINE_MUTATE__(LE)
  DEFINE_MUTATE__(GT)
  DEFINE_MUTATE__(GE)
  DEFINE_MUTATE__(Select)
  DEFINE_MUTATE__(Load)
  DEFINE_MUTATE__(Ramp)
  DEFINE_MUTATE__(Broadcast)
  DEFINE_MUTATE__(Let)
  DEFINE_MUTATE__(Call)
  DEFINE_MUTATE__(Variable)
  DEFINE_MUTATE__(Shuffle)

 private:
  std::string target_;
};

LoweredFunc LowerDatatypes(LoweredFunc f, const std::string& target) {
  auto n = make_node<LoweredFuncNode>(*f.operator->());
  n->body = DatatypesLowerer(target).Mutate(n->body);
  return LoweredFunc(n);
}

}  // namespace ir
}  // namespace tvm
