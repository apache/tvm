CUBLAS_CONTRIB_SRC = $(wildcard src/contrib/cublas/*.cc)
CUBLAS_CONTRIB_OBJ = $(patsubst src/%.cc, build/%.o, $(CUBLAS_CONTRIB_SRC))

ifeq ($(USE_CUBLAS), 1)
CFLAGS += -DTVM_USE_CUBLAS=1
ADD_LDFLAGS += -lcublas
RUNTIME_DEP += $(CUBLAS_CONTRIB_OBJ)
endif
