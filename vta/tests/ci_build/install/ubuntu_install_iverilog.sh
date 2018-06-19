apt-get install -y --no-install-recommends --force-yes make bison flex
wget ftp://icarus.com/pub/eda/verilog/v10/verilog-10.1.tar.gz
tar xf verilog-10.1.tar.gz
cd verilog-10.1
./configure --prefix=/usr
make install -j8
cd ..
rm -rf verilog-10.1 verilog-10.1.tar.gz
