CUDNN_CONTRIB_SRC = $(wildcard src/contrib/cudnn/*.cc)
CUDNN_CONTRIB_OBJ = $(patsubst src/%.cc, build/%.o, $(CUDNN_CONTRIB_SRC))

ifeq ($(USE_CUDNN), 1)
CFLAGS += -DTVM_USE_CUDNN=1 -I$(CUDA_PATH)/include
ADD_LDFLAGS += -lcudnn
RUNTIME_DEP += $(CUDNN_CONTRIB_OBJ)
endif
