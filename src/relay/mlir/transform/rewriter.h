#ifndef RELAY_MLIR_REWRITER_H
#define RELAY_MLIR_REWRITER_H

#include <memory>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/Bufferize.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
class OwningRewritePatternList;

namespace mrelay {

// Collection of rewrite patterns for lowering of mhlo to mlir relay dialect.
void populateMhloToMrelayConversionPattern(MLIRContext *context,
                                           OwningRewritePatternList *patterns);

} // namespace mrelay
}  // namespace mlir

#endif //RELAY_MLIR_REWRITER_H






