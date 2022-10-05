wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y clang-8 clang++-8 libc++-8-dev libc++abi-8-dev
