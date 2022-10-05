git clone https://github.com/llvm/llvm-project.git
git checkout -b release/13.x origin/release/13.x
cd llvm-project
mkdir build
cd build
cmake -G Ninja -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_TARGETS_TO_BUILD="X86;ARM;NVPTX;AArch64;Mips;Hexagon" -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_BUILD_32_BITS=OFF -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON ../llvm/