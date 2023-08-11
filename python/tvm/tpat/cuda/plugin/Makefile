#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Variables need to be defined by Users
# CUDA_PATH   = /path/to/cuda
# CUDNN_PATH = /path/to/cudnn
# TRT_PATH = /path/to/TensorRT
CUDA_PATH   = /home/huangzhe1/anaconda3/envs/tvm_tunning
CUDNN_PATH = /home/huangzhe1/husen/cudnn-linux-x86_64-8.9.3.28_cuda11-archive
TRT_PATH = /home/huangzhe1/husen/TensorRT-8.6.1.6
ARCH = sm_86
########################################

CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_LIB_PATH  = $(CUDA_PATH)/lib
CUDA_COM_PATH = $(CUDA_PATH)/samples/common/inc

CUDNN_INC_PATH = $(CUDNN_PATH)/include
CUDNN_LIB_PATH = $(CUDNN_PATH)/lib

TRT_INC_PATH   = $(TRT_PATH)/include
TRT_LIB_PATH = $(TRT_PATH)/lib

GCC = g++
NVCC = $(CUDA_PATH)/bin/nvcc
CCFLAGS = -w -std=c++11
INCLUDES := -I. -I$(CUDA_COM_PATH) -I$(CUDA_INC_PATH) -I$(CUDNN_INC_PATH) -I$(TRT_INC_PATH) -I/usr/include

LDFLAGS := -L$(CUDA_LIB_PATH) -L$(CUDNN_LIB_PATH) -L$(TRT_LIB_PATH)
LDFLAGS += -lnvinfer -lcudart -lcuda

LDFLAGS += -Wl,-rpath=$(CUDA_LIB_PATH)
LDFLAGS += -Wl,-rpath=$(CUDNN_LIB_PATH)
LDFLAGS += -Wl,-rpath=$(TRT_LIB_PATH)

SO = $(plugin_name).so
OBJ = $(shell find . -name '*.o')
DEP = $(OBJ:.o=.d)

SRCDIR := ./src
OBJDIR := ./obj
LIBDIR := ./lib

all: $(SO)

$(plugin_name).so: $(plugin_name).o

-include $(DEP)

clean:
	rm -rf $(LIBDIR)/$(SO) $(OBJDIR)/*

%.o: $(SRCDIR)/%.cpp
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(GCC) $(CCFLAGS) -fPIC -MD -MP $(INCLUDES) -o $@ -c $<

%.o: $(SRCDIR)/%.cu
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(NVCC) $(CCFLAGS) -M -MT $@ $(INCLUDES) -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAGS) $(INCLUDES) -Xcompiler -fPIC -arch=$(ARCH) -o $@ -c $<

$(SO):
	$(GCC) $(CCFLAGS) -shared -o $@ $+ $(LDFLAGS)
	$(AT)if [ ! -d  $(LIBDIR) ]; then mkdir -p $(LIBDIR); fi
	$(AT) mv *.o   $(OBJDIR)/
	$(AT) mv *.d   $(OBJDIR)/
	$(AT) mv *.so $(LIBDIR)/
