.PHONY: clean all

TVM_BASE   = $(PWD)/../../
TARGET     = gotvm
LIBS       = -lm -ldl
NATIVE_SRC = tvm.cc

GOPATH=$(PWD)/gopath
GOPATHDIR=${GOPATH}/src/${TARGET}/
CGO_CPPFLAGS="-I. -I${TVM_BASE}/ -I${TVM_BASE}/dmlc-core/include -I${TVM_BASE}/include -I${TVM_BASE}/dlpack/include/"
CGO_CFLAGS="-I${TVM_BASE}"
CGO_LDFLAGS="-ldl -lm"


all:
	@mkdir gopath 2>/dev/null || true
	@mkdir gopath/src 2>/dev/null || true
	@mkdir gopath/src/$(TARGET) 2>/dev/null || true
	@cp $(TARGET).cc gopath/src/$(TARGET)
	@cp $(TARGET).h gopath/src/$(TARGET)
	@cp $(NATIVE_SRC) gopath/src/$(TARGET)
	@cp *.go gopath/src/$(TARGET)
	@export GOPATH=$(GOPATH); \
	export CGO_CPPFLAGS=$(CGO_CPPFLAGS); \
	export CGO_CFLAGS=$(CGO_CFLAGS); \
	export CGO_LDFLAGS=$(CGO_LDFLAGS); \
	(cd $(GOPATHDIR) && go clean \
	&& golint && go build -o $(TARGET).a \
	&& go install \
	&& python3 ../../../../sample/deploy.py \
	&& go test -v)
	@mkdir gopath/doc 2>/dev/null || true
	@godoc -html -goroot gopath/ gotvm | grep -v "for documentation on the gotvm command" > gopath/doc/gotvm.html
	@echo "Run 'godoc -http=:6060  -goroot=./gopath' for documentation"

clean:
	@if [ -d $(GOPATHDIR) ] ; then \
	export GOPATH=$(GOPATH); \
	export CGO_CPPFLAGS=$(CGO_CPPFLAGS); \
	export CGO_CFLAGS=$(CGO_CFLAGS); \
	export CGO_LDFLAGS=$(CGO_LDFLAGS); \
	(cd $(GOPATHDIR) && go clean -cache); fi
	rm -rf gopath
	rm -f *.a *.o

lint:
	@golint
	@python3 ${TVM_BASE}/dmlc-core/scripts/lint.py gotvm cpp *.cc
	@python3 ${TVM_BASE}/dmlc-core/scripts/lint.py gotvm cpp *.h

test:
	@cp *_test.go gopath/src/$(TARGET)
	@export GOPATH=$(GOPATH); \
	export CGO_CPPFLAGS=$(CGO_CPPFLAGS); \
	export CGO_CFLAGS=$(CGO_CFLAGS); \
	export CGO_LDFLAGS=$(CGO_LDFLAGS); \
	(cd $(GOPATHDIR) \
	&& go test -v)
