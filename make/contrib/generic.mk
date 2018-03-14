GENERIC_CONTRIB_SRC = $(wildcard src/contrib/generic/*.cc)
GENERIC_CONTRIB_OBJ = $(patsubst src/%.cc, build/%.o, $(GENERIC_CONTRIB_SRC))

ifeq ($(USE_BLAS), openblas)
	ADD_LDFLAGS += -lopenblas
	RUNTIME_DEP += $(GENERIC_CONTRIB_OBJ)
else ifeq ($(USE_BLAS), atlas)
	ADD_LDFLAGS += -lcblas
	RUNTIME_DEP += $(GENERIC_CONTRIB_OBJ)
else ifeq ($(USE_BLAS), blas)
	ADD_LDFLAGS += -lblas
	RUNTIME_DEP += $(GENERIC_CONTRIB_OBJ)
else ifeq ($(USE_BLAS), apple)
	ADD_CFLAGS += -I/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/
	FRAMEWORKS += -framework Accelerate
	RUNTIME_DEP += $(GENERIC_CONTRIB_OBJ)
endif
