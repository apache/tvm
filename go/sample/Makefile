.PHONY: clean all

all:
	@golint
	@go tool compile -pack -o simple.o simple.go
	@go tool link -linkmode external -extld "g++" -extldflags "-ldl" -o simple simple.o
	@python3 deploy.py
	@go tool compile -pack -o complex.o complex.go
	@go tool link -linkmode external -extld "g++" -extldflags "-ldl" -o complex complex.o
	@go tool compile -pack -o funccb.o funccb.go
	@go tool link -linkmode external -extld "g++" -extldflags "-ldl" -o funccb funccb.o

clean:
	rm -f simple complex funccb *.so *.o
