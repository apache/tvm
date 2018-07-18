.PHONY: clean all

all:
	@go tool compile -pack -o simple.o simple.go
	@go tool link -linkmode external -extld "g++" -extldflags "-ldl" -o simple simple.o
	@go tool compile -pack -o complex.o complex.go
	@go tool link -linkmode external -extld "g++" -extldflags "-ldl" -o complex complex.o
	@python3 deploy.py

clean:
	rm -f simple complex *.so *.o
