.PHONY: clean all

all:
	@go tool compile -pack -o simple.o simple.go
	@go tool link -linkmode external -extld "g++" -extldflags "-ldl" -o simple simple.o
	@python3 deploy.py
	@go tool compile -pack -o complex.o complex.go
	@go tool link -linkmode external -extld "g++" -extldflags "-ldl" -o complex complex.o

clean:
	rm -f simple complex *.so *.o
