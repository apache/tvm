ifeq ($(USE_IMAGE), 1)

IMAGE_CONTRIB_SRC = $(wildcard src/contrib/image/*.cc)
IMAGE_CONTRIB_OBJ = $(patsubst src/%.cc, build/%.o, $(IMAGE_CONTRIB_SRC))
RUNTIME_DEP += $(IMAGE_CONTRIB_OBJ)

ifeq ($(USE_JPEG), 1)
	ADD_LDFLAGS += -ljpeg
	ADD_CFLAGS  += -DTVM_JPEG_OPS
endif
ifeq ($(USE_PNG), 1)
	ADD_LDFLAGS += -lpng
	ADD_CFLAGS  += -DTVM_PNG_OPS
endif
ifeq ($(USE_GIF), 1)
	ADD_LDFLAGS += -lgif
	ADD_CFLAGS  += -DTVM_GIF_OPS
endif
ifeq ($(USE_BMP), 1)
	ADD_LDFLAGS += -lbmp
	ADD_CFLAGS  += -DTVM_BMP_OPS
endif

endif
