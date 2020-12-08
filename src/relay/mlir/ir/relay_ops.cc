#include "relay_ops.h"
#include <assert.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Traits.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Transforms/InliningUtils.h>

namespace mlir {

namespace mrelay {

#define GET_OP_CLASSES
#include "src/ir/relay_ops.cc.inc"

//===----------------------------------------------------------------------===//
// Relay Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct RelayInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }

  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// relay Dialect Constructor
//===----------------------------------------------------------------------===//

RelayDialect::RelayDialect(MLIRContext *context)
    : mlir::Dialect(getDialectNamespace(), context,
                    TypeID::get<RelayDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "src/ir/relay_ops.cc.inc"
      >();
  addInterfaces<RelayInlinerInterface>();
  // addTypes<RelayCustomType>(); Add custom types if needed
}

Type RelayDialect::parseType(DialectAsmParser &parser) const {
  StringRef data_type;
  if (parser.parseKeyword(&data_type))
    return Type();
  parser.emitError(parser.getNameLoc())
      << "unknown mlir relay type: " << data_type;
  return nullptr;
}

void RelayDialect::printType(Type type, DialectAsmPrinter &printer) const {
  // todo (yongwwww): add print for added types
  printer << "<unknown mlir relay type>";
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

template <typename Op, typename ElementType = Type, typename ValType,
          typename Convert>
static Attribute BinaryFolder(Op *op, ArrayRef<Attribute> attrs) {
  if (!attrs[0] || !attrs[1])
    return {};

  DenseElementsAttr lhs = attrs[0].dyn_cast<DenseElementsAttr>();
  DenseElementsAttr rhs = attrs[1].dyn_cast<DenseElementsAttr>();
  if (!lhs || !rhs)
    return {};

  ShapedType type = op->getType().template cast<ShapedType>();
  if (!type.hasStaticShape()) {
    return {};
  }

  Type etype = type.getElementType();

  // Evaluate for integer values.
  if (!etype.isa<ElementType>()) {
    return {};
  }

  SmallVector<ValType, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip :
       llvm::zip(lhs.getValues<ValType>(), rhs.getValues<ValType>())) {
    values.push_back(Convert()(std::get<0>(zip), std::get<1>(zip)));
  }

  return DenseElementsAttr::get(type, values);
}

#define BINARY_FOLDER(Op, Func)                                                \
  OpFoldResult Op::fold(ArrayRef<Attribute> attrs) {                           \
    if (getElementTypeOrSelf(getType()).isa<FloatType>())                      \
      return BinaryFolder<Op, FloatType, APFloat, Func<APFloat>>(this, attrs); \
    if (getElementTypeOrSelf(getType()).isa<IntegerType>())                    \
      return BinaryFolder<Op, IntegerType, APInt, Func<APInt>>(this, attrs);   \
    return {};                                                                 \
  }

BINARY_FOLDER(mrelay::AddOp, std::plus);

#undef BINARY_FOLDER

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

OpFoldResult mrelay::ConstOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");

  // Return the held attribute value.
  return value();
}

// Builds a constant op with the specified attribute `value`.
void mrelay::ConstOp::build(OpBuilder &builder, OperationState &result,
                            Attribute value) {
  Type type;
  if (auto elemAttr = value.dyn_cast<ElementsAttr>()) {
    type = elemAttr.getType();
  } else if (value.isa<BoolAttr>() || value.isa<FloatAttr>() ||
             value.isa<IntegerAttr>()) {
    type = RankedTensorType::get(/*shape=*/{}, value.getType());
    value = DenseElementsAttr::get(type.cast<TensorType>(), value);
  }

  assert(type && "unsupported attribute type for building mlir relay const op");
  result.types.push_back(type);
  result.addAttribute("value", value);
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(mrelay::ReshapeOp op) {
  // If the operand type is dynamically shaped there is nothing to verify.
  auto operand_ty = op.data().getType().dyn_cast<RankedTensorType>();
  if (!operand_ty || !operand_ty.hasStaticShape())
    return success();

  // If the operand type is statically shaped (not required) the number of
  // elements must match that of the result type.
  auto result_ty = op.getType().cast<RankedTensorType>();
  assert(result_ty && result_ty.hasStaticShape() &&
         "result type must be statically shaped");
  int64_t num_result_elements = result_ty.getNumElements();
  int64_t num_operand_elements = operand_ty.getNumElements();
  if (num_result_elements != num_operand_elements)
    return op.emitOpError()
           << "number of output elements (" << num_result_elements
           << ") doesn't match expected number of elements ("
           << num_operand_elements << ")";

  return success();
}

OpFoldResult mrelay::ReshapeOp::fold(ArrayRef<Attribute> operands) {
  // todo (yongwww): add logic for new_shape
  if (getOperand(0).getType() == getType()) {
    return getOperand(0);
  }

  if (auto prev_op =
          dyn_cast_or_null<ReshapeOp>(getOperand(0).getDefiningOp())) {
    setOperand(0, prev_op.getOperand(0));
    return getResult();
  }

  if (auto elements = operands.front().dyn_cast_or_null<DenseElementsAttr>()) {
    return elements.reshape(getResult().getType().cast<ShapedType>());
  }

  return {};
}

//===----------------------------------------------------------------------===//
// StridedSliceOp
//===----------------------------------------------------------------------===//

OpFoldResult mrelay::StridedSliceOp::fold(ArrayRef<Attribute> operands) {
  return {};
}

} // namespace mrelay
} // namespace mlir
