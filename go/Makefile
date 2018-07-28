.PHONY: clean all

all:
	(cd runtime; make)
	cp runtime/gopath/pkg/linux_amd64/gotvm.a sample/ -rfa
	make -C sample

clean:
	@make -C runtime clean
	@rm sample/gotvm.a
	@make -C sample clean
