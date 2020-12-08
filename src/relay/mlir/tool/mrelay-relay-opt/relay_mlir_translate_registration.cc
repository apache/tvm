#include "src/ir/relay_ops.h"
#include "src/translate/relay_mlir_translate.h"
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/MlirOptMain.h>
#include <mlir/Translation.h>

namespace mlir {

inline void RegisterAllRelayDialects(DialectRegistry &registry) {
  registry.insert<mlir::StandardOpsDialect, mrelay::RelayDialect>();
}

static TranslateToMLIRRegistration
    RelayToMlirTranslate("relay-to-mlir", mrelay::RelayToMlirTranslateFunction);

static TranslateFromMLIRRegistration
    mlir_to_relay_translate("mlir-to-relay",
                            mrelay::MlirToRelayTranslateFunction,
                            [](DialectRegistry &registry) {
                              mlir::RegisterAllRelayDialects(registry);
                            });

} // namespace mlir
