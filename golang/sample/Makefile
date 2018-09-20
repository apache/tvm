.PHONY: clean all

SOURCES=$(wildcard *.go)
EXECUTABLE=$(patsubst %.go, %, $(SOURCES))

all: $(EXECUTABLE)
	@golint
	@python3 deploy.py

%: %.o
	@go tool link -linkmode external -extld "g++" -extldflags "-ldl" -o $@ $<

%.o: %.go
	@go tool compile -pack -o $@ $<

clean:
	@rm -f $(EXECUTABLE) *.so *.o *.a
