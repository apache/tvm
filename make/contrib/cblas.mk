CBLAS_CONTRIB_SRC = $(wildcard src/contrib/cblas/*.cc)
CBLAS_CONTRIB_OBJ = $(patsubst src/%.cc, build/%.o, $(CBLAS_CONTRIB_SRC))

ifeq ($(USE_BLAS), openblas)
	ADD_LDFLAGS += -lopenblas
	RUNTIME_DEP += $(CBLAS_CONTRIB_OBJ)
else ifeq ($(USE_BLAS), atlas)
	ADD_LDFLAGS += -lcblas
	RUNTIME_DEP += $(CBLAS_CONTRIB_OBJ)
else ifeq ($(USE_BLAS), blas)
	ADD_LDFLAGS += -lblas
	RUNTIME_DEP += $(CBLAS_CONTRIB_OBJ)
else ifeq ($(USE_BLAS), apple)
	ADD_CFLAGS += -I/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/
	FRAMEWORKS += -framework Accelerate
	RUNTIME_DEP += $(CBLAS_CONTRIB_OBJ)
endif
