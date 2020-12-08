# Path to mlir-hlo source folder.

if(DEFINED ENV{MLIR_HLO_SRC})
  set(MLIR_HLO_SRC $ENV{MLIR_HLO_SRC})
    if(EXISTS ${MLIR_HLO_SRC})
        message(STATUS "MLIR_HLO_SRC" ${MLIR_HLO_SRC})
  else()
      message(FATAL_ERROR "The path specified by MLIR_HLO_SRC does not exist: "
          ${MLIR_HLO_SRC})
  endif()
else()
    message(FATAL_ERROR "ENV variable MLIR_HLO_SRC not set")
endif()

# Path to mlir-hlo build folder
set(MLIR_HLO_BUILD "${MLIR_HLO_SRC}/build")
if(EXISTS ${MLIR_HLO_BUILD})
    message(STATUS "MLIR_HLO_BUILD " ${MLIR_HLO_BUILD})
else()
    message(FATAL_ERROR "Expecting MLIR_HLO_BUILD to exist: " ${MLIR_HLO_BUILD})
endif()

include_directories("${MLIR_HLO_SRC}/include")
include_directories("${MLIR_HLO_SRC}/lib")
include_directories("${MLIR_HLO_BUILD}/include")
include_directories("${MLIR_HLO_BUILD}/lib")

set(MLIR_HLO_LINK_FLAGS -lhlo -L${MLIR_HLO_SRC}/build)