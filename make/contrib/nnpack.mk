NNPACK_CONTRIB_SRC = $(wildcard src/contrib/nnpack/*.cc)
NNPACK_CONTRIB_OBJ = $(patsubst src/%.cc, build/%.o, $(NNPACK_CONTRIB_SRC))

ifeq ($(USE_NNPACK), 1)
ifndef NNPACK_PATH
	NNPACK_PATH = $(ROOTDIR)/NNPACK
endif
	PTHREAD_POOL_PATH = $(NNPACK_PATH)/deps/pthreadpool
	CFLAGS += -DTVM_USE_NNPACK=1 -I$(NNPACK_PATH)/include -I$(PTHREAD_POOL_PATH)/include
	LDFLAGS += -L$(NNPACK_PATH)/lib -lnnpack -lpthreadpool -lpthread
	RUNTIME_DEP += $(NNPACK_CONTRIB_OBJ)
endif
