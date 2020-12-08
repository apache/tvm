# Path to LLVM source folder.
if(DEFINED ENV{LLVM_PROJ_SRC})
  set(LLVM_PROJ_SRC $ENV{LLVM_PROJ_SRC})
  if(EXISTS ${LLVM_PROJ_SRC})
    message(STATUS "LLVM_PROJ_SRC " ${LLVM_PROJ_SRC})
  else()
    message(FATAL_ERROR "The path specified by LLVM_PROJ_SRC does not exist: "
            ${LLVM_PROJ_SRC})
  endif()
else()
  message(FATAL_ERROR "env variable LLVM_PROJ_SRC not set")
endif()

# Path to LLVM build folder
if(DEFINED ENV{LLVM_PROJ_BUILD})
  set(LLVM_PROJ_BUILD $ENV{LLVM_PROJ_BUILD})
  if(EXISTS ${LLVM_PROJ_BUILD})
    message(STATUS "LLVM_PROJ_BUILD " ${LLVM_PROJ_BUILD})
  else()
    message(FATAL_ERROR "The path specified by LLVM_PROJ_BUILD does not exist: "
            ${LLVM_PROJ_BUILD})
  endif()
else()
  message(FATAL_ERROR "env variable LLVM_PROJ_BUILD not set")
endif()

# LLVM project lib folder
set(LLVM_PROJECT_LIB ${LLVM_PROJ_BUILD}/lib)
set(LLVM_PROJ_BIN ${LLVM_PROJ_BUILD}/bin)

# Include paths for MLIR
set(LLVM_SRC_INCLUDE_PATH ${LLVM_PROJ_SRC}/llvm/include)
set(LLVM_BIN_INCLUDE_PATH ${LLVM_PROJ_BUILD}/include)
set(MLIR_SRC_INCLUDE_PATH ${LLVM_PROJ_SRC}/mlir/include)
set(MLIR_BIN_INCLUDE_PATH ${LLVM_PROJ_BUILD}/tools/mlir/include)
set(MLIR_TOOLS_DIR ${LLVM_PROJ_BIN})

set(RELAY_MLIR_TOOLS_DIR ${RELAY_MLIR_BIN_ROOT}/bin)
set(RELAY_MLIR_LIT_TEST_SRC_DIR ${RELAY_MLIR_SRC_ROOT}/test/mlir)
set(RELAY_MLIR_LIT_TEST_BUILD_DIR ${CMAKE_BINARY_DIR}/test/mlir)

set(
        MLIR_INCLUDE_PATHS
        ${LLVM_SRC_INCLUDE_PATH};${LLVM_BIN_INCLUDE_PATH};${MLIR_SRC_INCLUDE_PATH};${MLIR_BIN_INCLUDE_PATH}
)
include_directories(${MLIR_INCLUDE_PATHS})

# Force CMAKE_INSTALL_PREFIX and BUILD_SHARED_LIBS to be the same as LLVM build
file(STRINGS ${LLVM_PROJ_BUILD}/CMakeCache.txt prefix REGEX CMAKE_INSTALL_PREFIX)
string(REGEX REPLACE "CMAKE_INSTALL_PREFIX:PATH=" "" prefix ${prefix})
set(CMAKE_INSTALL_PREFIX ${prefix} CACHE PATH "" FORCE)
message(STATUS "CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})

file(STRINGS ${LLVM_PROJ_BUILD}/CMakeCache.txt shared REGEX BUILD_SHARED_LIBS)
string(REGEX REPLACE "BUILD_SHARED_LIBS:BOOL=" "" shared ${shared})
set(BUILD_SHARED_LIBS ${shared} CACHE BOOL "" FORCE)
message(STATUS "BUILD_SHARED_LIBS: " ${BUILD_SHARED_LIBS})

# Threading libraries required due to parallel pass execution.
find_package(Threads REQUIRED)
find_package(Curses REQUIRED)
find_package(ZLIB REQUIRED)

# Set output library path
set(RELAY_MLIR_LIB_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

function(find_mlir_lib lib)
  find_library(${lib}
          NAMES ${lib}
          PATHS ${LLVM_PROJECT_LIB}
          NO_DEFAULT_PATH)
  if(${${lib}} STREQUAL ${lib}-NOTFOUND)
    message(FATAL_ERROR "${lib} not found, did you forget to build llvm-project?")
  endif()
endfunction(find_mlir_lib)


find_mlir_lib(MLIRAffine)
find_mlir_lib(MLIRAffineUtils)
find_mlir_lib(MLIRAffineToStandard)
find_mlir_lib(MLIRAffineTransforms)
find_mlir_lib(MLIRAnalysis)
find_mlir_lib(MLIRCallInterfaces)
find_mlir_lib(MLIRControlFlowInterfaces)
find_mlir_lib(MLIRCopyOpInterface)
find_mlir_lib(MLIRDialect)
find_mlir_lib(MLIREDSC)
find_mlir_lib(MLIRExecutionEngine)
find_mlir_lib(MLIRInferTypeOpInterface)
find_mlir_lib(MLIRIR)
find_mlir_lib(MLIRLLVMIR)
find_mlir_lib(MLIRLoopAnalysis)
find_mlir_lib(MLIRSCFToStandard)
find_mlir_lib(MLIRLoopLikeInterface)
find_mlir_lib(MLIRLinalg)
find_mlir_lib(MLIRLinalgEDSC)
find_mlir_lib(MLIRLinalgAnalysis)
find_mlir_lib(MLIRLinalgTransforms)
find_mlir_lib(MLIRLinalgUtils)
find_mlir_lib(MLIRSCF)
find_mlir_lib(MLIRSCFTransforms)
find_mlir_lib(MLIRLLVMIRTransforms)
# find_mlir_lib(MLIRMlirOptMain)
find_mlir_lib(MLIRParser)
find_mlir_lib(MLIRPass)
find_mlir_lib(MLIRRewrite)
find_mlir_lib(MLIRStandard)
find_mlir_lib(MLIRStandardOpsTransforms)
find_mlir_lib(MLIRStandardToLLVM)
find_mlir_lib(MLIRSideEffectInterfaces)
find_mlir_lib(MLIRTargetLLVMIR)
find_mlir_lib(MLIRTransforms)
find_mlir_lib(MLIRTransformUtils)
find_mlir_lib(MLIRSupport)
find_mlir_lib(MLIRShape)
find_mlir_lib(MLIRShapeToStandard)
find_mlir_lib(MLIRSideEffectInterfaces)
find_mlir_lib(MLIROpenMP)
find_mlir_lib(MLIROptLib)
find_mlir_lib(MLIRTableGen)
find_mlir_lib(MLIRTargetLLVMIRModuleTranslation)
find_mlir_lib(MLIRTargetLLVMIR)
find_mlir_lib(MLIRTransformUtils)
find_mlir_lib(MLIRTranslation)
find_mlir_lib(MLIRVector)
find_mlir_lib(MLIRVectorInterfaces)
find_mlir_lib(MLIRVectorToLLVM)
find_mlir_lib(MLIRVectorToSCF)
# find_mlir_lib(MLIRMlirOptMain)
find_mlir_lib(MLIRAffineEDSC)
find_mlir_lib(MLIRLinalgEDSC)
find_mlir_lib(MLIRViewLikeInterface)
find_mlir_lib(MLIRPresburger)

find_mlir_lib(LLVMCore)
find_mlir_lib(LLVMSupport)
find_mlir_lib(LLVMAsmParser)
find_mlir_lib(LLVMBinaryFormat)
find_mlir_lib(LLVMRemarks)
find_mlir_lib(LLVMIRReader)
find_mlir_lib(LLVMTransformUtils)
find_mlir_lib(LLVMBitstreamReader)
find_mlir_lib(LLVMAnalysis)
find_mlir_lib(LLVMBitWriter)
find_mlir_lib(LLVMBitReader)
find_mlir_lib(LLVMMC)
find_mlir_lib(LLVMMCParser)
find_mlir_lib(LLVMObject)
find_mlir_lib(LLVMProfileData)
find_mlir_lib(LLVMDemangle)
find_mlir_lib(LLVMFrontendOpenMP)

set(MLIRLibs
        ${MLIRAffineToStandard}
        ${MLIRAffine}
        ${MLIRAffineUtils}
        ${MLIRCopyOpInterface}
        ${MLIRLLVMIR}
        ${MLIRStandard}
        ${MLIRStandardOpsTransforms}
        ${MLIRStandardToLLVM}
        ${MLIRTransforms}
        ${MLIRSCFToStandard}
        ${MLIRVector}
        ${MLIRVectorInterfaces}
        ${MLIRVectorToLLVM}
        ${MLIRVectorToSCF}
        ${MLIRSCF}
        ${MLIRIR}
        ${MLIRLLVMIR}
        ${MLIROptLib}
        ${MLIRParser}
        ${MLIRPass}
        ${MLIRTargetLLVMIR}
        ${MLIRTargetLLVMIRModuleTranslation}
        ${MLIRTransforms}
        ${MLIRTransformUtils}
        ${MLIRAffine}
        ${MLIRAffineToStandard}
        ${MLIRAffineTransforms}
        ${MLIRAnalysis}
        ${MLIRCallInterfaces}
        ${MLIRControlFlowInterfaces}
        ${MLIRDialect}
        ${MLIREDSC}
        ${MLIRExecutionEngine}
        ${MLIRIR}
        ${MLIRLLVMIRTransforms}
        ${MLIRSCFToStandard}
        ${MLIRSCF}
        ${MLIRSCFTransforms}
        ${MLIRLoopAnalysis}
        ${MLIRLoopLikeInterface}
        ${MLIROpenMP}
        # ${MLIRMlirOptMain}
        ${MLIRSideEffectInterfaces}
        ${MLIRStandard}
        ${MLIRStandardToLLVM}
        ${MLIRTranslation}
        ${MLIRSupport}
        ${MLIRLinalg}
        ${MLIRLinalgEDSC}
        ${MLIRLinalgAnalysis}
        ${MLIRLinalgTransforms}
        ${MLIRLinalgUtils}
        ${MLIRAffineEDSC}
        ${MLIRLinalgEDSC}
        ${MLIRViewLikeInterface}
        ${MLIRPresburger}
        ${MLIRShape}
        ${MLIRShapeToStandard}
        ${MLIRInferTypeOpInterface}
        ${MLIRRewrite}
        # strict order verified
        ${LLVMBitWriter}
        ${LLVMObject}
        ${LLVMBitReader}
        # strict order verified
        ${LLVMFrontendOpenMP}
        ${LLVMTransformUtils}
        ${LLVMAnalysis}
        # strict order verified
        ${LLVMAsmParser}
        ${LLVMCore}
        # strict order not yet verified
        ${LLVMRemarks}
        ${LLVMMCParser}
        ${LLVMMC}
        ${LLVMProfileData}
        ${LLVMBinaryFormat}
        ${LLVMBitstreamReader}
        ${LLVMIRReader}
        ${LLVMMLIRTableGen}
        ${LLVMSupport}
        ${LLVMDemangle}
        ${CMAKE_THREAD_LIBS_INIT}
	      ${CURSES_LIBRARIES}
	      ${ZLIB_LIBRARIES})

set(LLVM_CMAKE_DIR
        "${LLVM_PROJ_BUILD}/lib/cmake/llvm"
        CACHE PATH "Path to LLVM cmake modules")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)

function(relay_mlir_tablegen ofn)
  tablegen(MLIR
          ${ARGV}
          "-I${MLIR_SRC_INCLUDE_PATH}"
          "-I${MLIR_BIN_INCLUDE_PATH}"
          "-I${RELAY_MLIR_SRC_ROOT}")
  set(TABLEGEN_OUTPUT
          ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
          PARENT_SCOPE)
endfunction()

# Import the pre-built mlir TableGen as an imported exetuable. It is required by
# the LLVM TableGen command to have the TableGen target so that changes to the
# table gen utility itself can be detected and cause re-compilation of .td file.
add_executable(mlir-tblgen IMPORTED)
set_property(TARGET mlir-tblgen
        PROPERTY IMPORTED_LOCATION ${LLVM_PROJ_BIN}/mlir-tblgen)
set(MLIR_TABLEGEN_EXE mlir-tblgen)

# Add a dialect used by Relay MLIR and copy the generated operation
# documentation to the desired places.
# c.f. https://github.com/llvm/llvm-project/blob/e298e216501abf38b44e690d2b28fc788ffc96cf/mlir/CMakeLists.txt#L11
function(add_relay_mlir_dialect_doc dialect dialect_tablegen_file)
  # Generate Dialect Documentation
  set(LLVM_TARGET_DEFINITIONS ${dialect_tablegen_file})
  relay_mlir_tablegen(${dialect}.md -gen-op-doc)
  set(GEN_DOC_FILE ${RELAY_MLIR_BIN_ROOT}/docs/Dialects/${dialect}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_BINARY_DIR}/${dialect}.md
          ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${dialect}.md)
  add_custom_target(${dialect}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(relay-mlir-doc ${dialect}DocGen)
endfunction()

add_custom_target(relay-mlir-doc)

