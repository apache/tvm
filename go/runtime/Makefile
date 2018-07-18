.PHONY: clean all

TVM_BASE   = $(PWD)/../../
SWIGEXE    = swig
TARGET     = tvmgo
INTERFACE  = tvmgo.i
LIBS       = -lm -ldl
NATIVE_SRC = tvm.cc

GOPATH=$(PWD)/gopath
GOPATHDIR=${GOPATH}/src/${TARGET}/
CGO_CPPFLAGS="-I. -I${TVM_BASE}/ -I${TVM_BASE}/dmlc-core/include -I${TVM_BASE}/include -I${TVM_BASE}/dlpack/include/"
CGO_CFLAGS="-I${TVM_BASE}"
CGO_LDFLAGS="-ldl -lm"


all:
	@${SWIGEXE} -DSWIGWORDSIZE64 -go -c++ -cgo -intgosize 64 -I../../include -I../../dlpack/include -o $(TARGET)_wrap.cxx $(INTERFACE)
	@mkdir gopath 2>/dev/null || true
	@mkdir gopath/src 2>/dev/null || true
	@mkdir gopath/src/$(INTERFACE:.i=) 2>/dev/null || true
	@cp $(TARGET)_wrap.cxx gopath/src/$(TARGET)
	@cp $(NATIVE_SRC) gopath/src/$(TARGET)
	@cp $(TARGET).go gopath/src/$(TARGET)
	@export GOPATH=$(GOPATH); \
	export CGO_CPPFLAGS=$(CGO_CPPFLAGS); \
	export CGO_CFLAGS=$(CGO_CFLAGS); \
	export CGO_LDFLAGS=$(CGO_LDFLAGS); \
	(cd $(GOPATHDIR) && go clean && go build -o $(TARGET).a && go install)

clean:
	@if [ -d $(GOPATHDIR) ] ; then \
	export GOPATH=$(GOPATH); \
	export CGO_CPPFLAGS=$(CGO_CPPFLAGS); \
	export CGO_CFLAGS=$(CGO_CFLAGS); \
	export CGO_LDFLAGS=$(CGO_LDFLAGS); \
	(cd $(GOPATHDIR) && go clean -cache); fi
	rm -rf gopath
	rm -f *_wrap* *.a $(TARGET).go *.o
