SORT_CONTRIB_SRC = $(wildcard src/contrib/sort/*.cc)
SORT_CONTRIB_OBJ = $(patsubst src/%.cc, build/%.o, $(SORT_CONTRIB_SRC))

ifeq ($(USE_SORT), 1)
    RUNTIME_DEP += $(SORT_CONTRIB_OBJ)
endif
