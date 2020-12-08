#ifndef IR_MRELAY_H_
#define IR_MRELAY_H_

#include <mlir/Dialect/Traits.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace mlir {
namespace mrelay {

class RelayDialect : public Dialect {
 public:
  explicit RelayDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "mrelay"; }

  // Parses a type registered to this dialect.
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  // Prints a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &printer) const override;
};

#define GET_OP_CLASSES
#include "src/ir/relay_ops.h.inc"

} // namespace mrelay
} // namespace mlir

#endif // IR_MRELAY_H_
