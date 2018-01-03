ROCBLAS_CONTRIB_SRC = $(wildcard src/contrib/rocblas/*.cc)
ROCBLAS_CONTRIB_OBJ = $(patsubst src/%.cc, build/%.o, $(ROCBLAS_CONTRIB_SRC))

ifeq ($(USE_ROCBLAS), 1)
CFLAGS += -DTVM_USE_ROCBLAS=1
ADD_LDFLAGS += -lrocblas
RUNTIME_DEP += $(ROCBLAS_CONTRIB_OBJ)
endif
