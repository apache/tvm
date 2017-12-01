RANDOM_CONTRIB_SRC = $(wildcard src/contrib/random/*.cc)
RANDOM_CONTRIB_OBJ = $(patsubst src/%.cc, build/%.o, $(RANDOM_CONTRIB_SRC))

ifeq ($(USE_RANDOM), 1)
	RUNTIME_DEP += $(RANDOM_CONTRIB_OBJ)
endif
