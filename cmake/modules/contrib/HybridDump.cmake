message(STATUS "Build with contrib.hybriddump")
file(GLOB HYBRID_CONTRIB_SRC src/contrib/hybrid/*.cc)
list(APPEND COMPILER_SRCS ${HYBRID_CONTRIB_SRC})
