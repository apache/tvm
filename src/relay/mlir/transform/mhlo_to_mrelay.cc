#include "mhlo_to_mrelay.h"
#include "src/ir/relay_ops.h"
#include "src/pass/passes.h"
#include "src/transform/rewriter.h"
#include <mlir-hlo/Dialect/mhlo/IR/hlo_ops.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Shape/Transforms/Passes.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/StandardOps/Transforms/FuncConversions.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/Bufferize.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
namespace mrelay {
namespace {

template <typename T>
using BaseOpConversion = OpConversionPattern<T>;

template <typename MhloOpTy>
class MhloToMrelayOpConverter : public BaseOpConversion<MhloOpTy> {
 public:
  using BaseOpConversion<MhloOpTy>::BaseOpConversion;
  LogicalResult matchAndRewrite(
      MhloOpTy mhloOp, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    Operation* op = mhloOp.getOperation();
    // op->getName().dump();
    SmallVector<Value, 4> buffer_args(operands.begin(), operands.end());
    auto relay_op = rewriter.create<MhloToMrelayOp<MhloOpTy>>(op->getLoc(), op->getResultTypes(),
                                                              buffer_args, op->getAttrs());
    rewriter.replaceOp(op, relay_op.getResult());
    return success();
  }
};

struct MhloToMrelayDynamicReshapeConverter
    : public BaseOpConversion<mhlo::DynamicReshapeOp> {
 public:
  using BaseOpConversion<mhlo::DynamicReshapeOp>::BaseOpConversion;

  LogicalResult matchAndRewrite(
    mhlo::DynamicReshapeOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const final {
    Type result_type;
    if (auto ranked_type = op.getType().dyn_cast<RankedTensorType>()) {
      result_type =
        MemRefType::get(ranked_type.getShape(), ranked_type.getElementType());
    } else if (auto unranked_type =
      op.getType().dyn_cast<UnrankedTensorType>()) {
      result_type = UnrankedMemRefType::get(unranked_type.getElementType(), 0);
    } else {
      return failure();
    }
    mhlo::DynamicReshapeOp::Adaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<MemRefReshapeOp>(
      op, result_type, adaptor.operand(), adaptor.output_shape());
    return success();
  }
};

// Legalize mhlo.return
struct MhloToMrelayReturnOpConverter : public BaseOpConversion<mhlo::ReturnOp> {
 public:
  using BaseOpConversion<mhlo::ReturnOp>::BaseOpConversion;

  LogicalResult matchAndRewrite(
    mhlo::ReturnOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto& entry_block = op.getParentRegion()->front();
    auto num_arguments = entry_block.getNumArguments();
    if (operands.size() > num_arguments) {
      return op.emitError(
        "The number of operands that need Copy operations is more "
        "than the number of target function arguments.");
    }

    // The index of the first output block argument.
    auto dest_arg_idx = num_arguments - operands.size();

    return success();
  }
};

class MhloToMrelayTensorLoadOpConverter
    : public BaseOpConversion<mlir::TensorLoadOp> {
 public:
  using BaseOpConversion<mlir::TensorLoadOp>::BaseOpConversion;
  LogicalResult matchAndRewrite(
    mlir::TensorLoadOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOp(op, operands);
    return success();
  }
};

struct MhloLegalizeToMrelay
    : public PassWrapper<MhloLegalizeToMrelay, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<RelayDialect>();
  }

 public:
  MhloLegalizeToMrelay() = default;
  MhloLegalizeToMrelay(const MhloLegalizeToMrelay& o) {}

  void runOnOperation() override {
    OwningRewritePatternList patterns;
    auto& context = getContext();
    ConversionTarget target(context);
    target.addLegalDialect<RelayDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addIllegalOp<mlir::TensorLoadOp>();
    target.addIllegalOp<mlir::TensorStoreOp>();
    target.addIllegalDialect<mhlo::MhloDialect>();

    BufferizeTypeConverter converter;
    auto isMemRefType = [](Type type) { return type.isa<BaseMemRefType>(); };
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      auto inputs = op.getType().getInputs();
      return llvm::all_of(inputs, isMemRefType) &&
      converter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<CallOp>([&](CallOp op) {
      return std::all_of(op.operand_type_begin(), op.operand_type_end(),
                         isMemRefType) &&
                         std::all_of(op.result_type_begin(), op.result_type_end(),
                                     isMemRefType);
    });
    target.addDynamicallyLegalOp<mlir::ReturnOp>([&](mlir::ReturnOp op) {
      return std::all_of(op.operand_type_begin(), op.operand_type_end(),
                         isMemRefType);
    });

    populateMhloToMrelayConversionPattern(&context, &patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

void populateMhloToMrelayConversionPattern(MLIRContext* context,
                                           OwningRewritePatternList* patterns) {
  // clang-format off
  patterns->insert<
      // MhloToMrelayDynamicReshapeConverter,
      MhloToMrelayOpConverter<mhlo::AddOp>,
      MhloToMrelayOpConverter<mhlo::ConstOp>,
      // MhloToMrelayOpConverter<mhlo::ExpOp>,
      // MhloToMrelayOpConverter<mhlo::MaxOp>,
      MhloToMrelayOpConverter<mhlo::ReshapeOp>
      // MhloToMrelayOpConverter<mhlo::SliceOp>
  >(context);
  // clang-format on
}

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToMrelayPass() {
  return std::make_unique<MhloLegalizeToMrelay>();
}

} // namespace mrelay
}  // namespace mlir
