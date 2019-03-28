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
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Cast>();
    auto type_code = op->type.code();
    auto src_type_code = op->value.type().code();
    // If either datatype is a registered custom datatype, we must lower.
    if (DatatypeRegistry::Global()->DatatypeRegistered(type_code) ||
        DatatypeRegistry::Global()->DatatypeRegistered(src_type_code)) {
      auto lower = GetCastLowerFunc(target_, type_code, src_type_code);
      CHECK(lower) << "Cast lowering function for target " << target_
                   << " destination type " << static_cast<unsigned>(type_code)
                   << " source type " << static_cast<unsigned>(src_type_code)
                   << "not found";
      return (*lower)(expr);
    }
    return expr;
  }

#define DEFINE_MUTATE__(OP)                                                    \
  inline Expr Mutate_(const OP *op, const Expr &e) final {                     \
    Expr expr = IRMutator::Mutate_(op, e);                                     \
    op = expr.as<OP>();                                                        \
    auto type_code = op->type.code();                                          \
    if (DatatypeRegistry::Global()->DatatypeRegistered(type_code)) {           \
      auto lower = Get##OP##LowerFunc(target_, type_code);                     \
      CHECK(lower) << #OP " lowering function for target " << target_          \
                   << " type " << static_cast<unsigned>(type_code)             \
                   << "not found";                                             \
      return (*lower)(expr);                                                   \
    }                                                                          \
    return expr;                                                               \
  }

  // TODO(gus) this list should be the same as the list of
  // DEFINE_GET_LOWER_FUNC_ in datatypes_registry.h. We should avoid the
  // duplication.
  // TODO(gus) what should be included in this list? See the commentary below
  // on the Load case. Some of these things may not actually need to be lowered,
  // or perhaps they all need to be lowered but special cases need to be added.
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
  // TODO(gus) currently running into an error where, in a simple program
  // consisting of casting two placeholder floats to bfloats, adding them, and
  // casting back to float, the lowering is encountering a Load of a bfloat.
  // This makes sense; when we use the cast values, this is technically a Load
  // in TVM.
  // One way to make this go away for now is to remove Load from this list. But
  // is this correct? Somehow, it works, but I'm not sure how; the codegen stage
  // is somehow gracefully able to encounter a Load of a custom datatype and not
  // fail.
  //DEFINE_MUTATE__(Load)
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
