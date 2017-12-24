MIOPEN_CONTRIB_SRC = $(wildcard src/contrib/miopen/*.cc)
MIOPEN_CONTRIB_OBJ = $(patsubst src/%.cc, build/%.o, $(MIOPEN_CONTRIB_SRC))

ifeq ($(USE_MIOPEN), 1)
CFLAGS += -DTVM_USE_MIOPEN=1
ADD_LDFLAGS += -lMIOpen
RUNTIME_DEP += $(MIOPEN_CONTRIB_OBJ)
endif
