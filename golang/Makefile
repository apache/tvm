.PHONY: clean all

TVM_BASE   = $(CURDIR)/../
TARGET     = gotvm
LIBS       = -lm -ldl
NATIVE_SRC = tvm_runtime_pack.cc

GOPATH=$(CURDIR)/gopath
GOPATHDIR=${GOPATH}/src/${TARGET}/
CGO_CPPFLAGS="-I. -I${TVM_BASE}/ -I${TVM_BASE}/3rdparty/dmlc-core/include -I${TVM_BASE}/include -I${TVM_BASE}/3rdparty/dlpack/include/"
CGO_CXXFLAGS="-std=c++11"
CGO_CFLAGS="-I${TVM_BASE}"
CGO_LDFLAGS="-ldl -lm"

all:
	@mkdir gopath 2>/dev/null || true
	@mkdir gopath/src 2>/dev/null || true
	@mkdir gopath/src/$(TARGET) 2>/dev/null || true
	@cp src/$(TARGET).cc gopath/src/$(TARGET)
	@cp src/$(TARGET).h gopath/src/$(TARGET)
	@cp src/$(NATIVE_SRC) gopath/src/$(TARGET)
	@cp src/*.go gopath/src/$(TARGET)
	@export GOPATH=$(GOPATH); \
	export CGO_CPPFLAGS=$(CGO_CPPFLAGS); \
	export CGO_CXXFLAGS=$(CGO_CXXFLAGS); \
	export CGO_CFLAGS=$(CGO_CFLAGS); \
	export CGO_LDFLAGS=$(CGO_LDFLAGS); \
	(cd $(GOPATHDIR) && go clean -cache \
	&& golint && go build -o $(TARGET).a \
	&& go install)
	@find . -name gotvm.a
	@#mkdir gopath/doc 2>/dev/null || true
	@#godoc -html -goroot gopath/ gotvm | grep -v "for documentation on the gotvm command" > gopath/doc/gotvm.html
	@#echo "Run 'godoc -http=:6060  -goroot=./gopath' for documentation"

samples: all
	cp gopath/pkg/linux_amd64/gotvm.a sample/ -rfa
	make -C sample

tests: all
	@(cd sample; python3 deploy.py)
	@export GOPATH=$(GOPATH); \
	export CGO_CPPFLAGS=$(CGO_CPPFLAGS); \
	export CGO_CXXFLAGS=$(CGO_CXXFLAGS); \
	export CGO_CFLAGS=$(CGO_CFLAGS); \
	export CGO_LDFLAGS=$(CGO_LDFLAGS); \
	(cd $(GOPATHDIR) \
	&& cp ../../../sample/deploy.so . \
	&& go test -v)

clean:
	@if [ -d $(GOPATHDIR) ] ; then \
	export GOPATH=$(GOPATH); \
	export CGO_CPPFLAGS=$(CGO_CPPFLAGS); \
	export CGO_CFLAGS=$(CGO_CFLAGS); \
	export CGO_LDFLAGS=$(CGO_LDFLAGS); \
	(cd $(GOPATHDIR) && go clean -cache); fi
	@rm -rf gopath
	@make -C sample clean

lint:
	@(cd src; golint)
	@python3 ${TVM_BASE}/dmlc-core/scripts/lint.py gotvm cpp src/*.cc
	@python3 ${TVM_BASE}/dmlc-core/scripts/lint.py gotvm cpp src/*.h
