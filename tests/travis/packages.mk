# rules for gtest
.PHONY: iverilog

iverilog: | ${CACHE_PREFIX}/bin/vvp

${CACHE_PREFIX}/bin/vvp:
	rm -rf verilog-10.1.tar.gz verilog-10.1
	wget ftp://icarus.com/pub/eda/verilog/v10/verilog-10.1.tar.gz
	tar xf verilog-10.1.tar.gz
	cd verilog-10.1;./configure --prefix=${CACHE_PREFIX}; make install
