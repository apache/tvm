# Makefile for example to deploy TVM modules in SGX.

PYTHON ?= python

NNVM_ROOT := $(shell cd ../../; pwd)
TVM_ROOT := $(NNVM_ROOT)/tvm
DMLC_CORE_ROOT := $(NNVM_ROOT)/dmlc-core

SGX_SDK ?= /opt/sgxsdk
SGX_MODE ?= SIM
SGX_ARCH ?= x64
SGX_DEBUG ?= 1

sgx_edger8r := $(SGX_SDK)/bin/x64/sgx_edger8r
sgx_enclave_signer := $(SGX_SDK)/bin/x64/sgx_sign

ifneq ($(SGX_MODE), HW)
	sgx_sim := _sim
endif
urts_library_name := sgx_urts$(sgx_sim)
trts_library_name := sgx_trts$(sgx_sim)
tservice_library_name := sgx_tservice$(sgx_sim)
uservice_library_name := sgx_uae_service$(sgx_sim)

pkg_cflags := -std=c++11 -O2 -fPIC\
	-I$(NNVM_ROOT)/include\
	-I$(NNVM_ROOT)\
	-I$(TVM_ROOT)/include\
	-I$(TVM_ROOT)/dlpack/include\
	-I$(DMLC_CORE_ROOT)/include\
	-DDMLC_LOG_STACK_TRACE=0\

pkg_ldflags := -L$(TVM_ROOT)/lib

enclave_include_paths := -I$(SGX_SDK)/include\
	-I$(SGX_SDK)/include/tlibc\
	-I$(SGX_SDK)/include/libcxx\
	-I$(SGX_SDK)/include/stdc++\

enclave_cflags := -static -nostdinc\
	-fvisibility=hidden -fpie -fstack-protector-strong\
	-ffunction-sections -fdata-sections\
	-DDMLC_CXX11_THREAD_LOCAL=0\
	$(enclave_include_paths)\

enclave_cxxflags := -nostdinc++ $(enclave_cflags)

enclave_ldflags :=\
	-Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles -L$(SGX_SDK)/lib64\
	-Wl,--whole-archive -l$(trts_library_name) -Wl,--no-whole-archive\
	-Wl,--start-group\
	-lsgx_tstdc -lsgx_tstdcxx -lsgx_tcxx -lsgx_tcrypto -lsgx_tkey_exchange -l$(tservice_library_name)\
	-Wl,--end-group\
	-Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined\
	-Wl,-pie,-eenclave_entry -Wl,--export-dynamic\
	-Wl,--defsym,__ImageBase=0 -Wl,--gc-sections

app_cflags := -I$(SGX_SDK)/include -Ilib

app_ldflags := -L$(SGX_SDK)/lib64\
	-l$(urts_library_name) -l$(uservice_library_name) -lpthread\

.PHONY: clean all

all: lib/model.signed.so bin/run_model

# The code library built by TVM
lib/deploy_%.o: build_model.py
	@mkdir -p $(@D)
	$(PYTHON) build_model.py

# EDL files
lib/model_%.c: model.edl $(sgx_edger8r)
	@mkdir -p $(@D)
	$(sgx_edger8r) $< --trusted-dir $(@D) --untrusted-dir $(@D) --search-path $(SGX_SDK)/include

lib/model_%.o: lib/model_%.c
	$(CC) $(enclave_cflags) -c $< -o $@

# The enclave library
lib/model.so: enclave.cc $(TVM_ROOT)/sgx/sgx_runtime.cc lib/model_t.o lib/deploy_lib.o
	$(CXX) $^ -o $@ $(pkg_cflags) $(pkg_ldflags) $(enclave_cxxflags) $(enclave_ldflags)\
		-Wl,--format=binary -Wl,lib/deploy_graph.json -Wl,lib/deploy_params.bin -Wl,--format=default

# The signed enclave
lib/model.signed.so: lib/model.so enclave_config.xml
	$(sgx_enclave_signer) sign -key enclave_private.pem -enclave $< -out $@ -config enclave_config.xml

# An app that runs the enclave
bin/run_model: app.cc lib/model_u.o
	@mkdir -p $(@D)
	$(CXX) $^ -o $@ $(app_cflags) $(app_ldflags)

# Debugging binary that runs TVM without SGX
bin/run_model_nosgx: enclave.cc $(TVM_ROOT)/sgx/sgx_runtime.cc lib/deploy_lib.o
	@mkdir -p $(@D)
	$(CXX) $^ -o $@ $(pkg_cflags) $(pkg_ldflags)\
		-Wl,--format=binary -Wl,lib/deploy_graph.json -Wl,lib/deploy_params.bin -Wl,--format=default


clean:
	rm -rf lib bin
