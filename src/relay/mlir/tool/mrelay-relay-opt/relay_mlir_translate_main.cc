#include "src/translate/relay_mlir_translate.h"
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/ToolUtilities.h>
#include <mlir/Translation.h>

static llvm::cl::opt<std::string> input_filename(llvm::cl::Positional,
                                                 llvm::cl::desc("<input file>"),
                                                 llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    output_filename("o", llvm::cl::desc("Output filename"),
                    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each chunk "
                   "independently"),
    llvm::cl::init(false));

int main(int argc, char **argv) {

  llvm::InitLLVM y(argc, argv);

  // Add flags for all the registered translations.
  llvm::cl::opt<const mlir::TranslateFunction *, false, mlir::TranslationParser>
      requested_translation("", llvm::cl::desc("Translation to perform"));

  mlir::registerAsmPrinterCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Relay MLIR translation driver\n");

  std::string error_message;
  auto output = mlir::openOutputFile(output_filename, &error_message);
  if (!output) {
    llvm::errs() << error_message << "\n";
    return 1;
  }

  auto input = mlir::openInputFile(input_filename, &error_message);

  if (!input) {
    llvm::errs() << error_message << "\n";
    return 1;
  }

  // Processes the memory buffer with a new MLIRContext.
  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
                           llvm::raw_ostream &os) {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());
    mlir::MLIRContext context;
    mlir::SourceMgrDiagnosticHandler diagnostic_handler(sourceMgr, &context);
    return (*requested_translation)(sourceMgr, os, &context);
  };

  if (splitInputFile) {
    if (failed(mlir::splitAndProcessBuffer(std::move(input), processBuffer,
                                           output->os())))
      return 1;
  } else {
    if (failed(processBuffer(std::move(input), output->os())))
      return 1;
  }

  output->keep();
  return 0;
}
