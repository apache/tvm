mv build/config.cmake ./;
rm -rf build;
mkdir build;
mv config.cmake build/;
cd build;
cmake ..;
make -j40

