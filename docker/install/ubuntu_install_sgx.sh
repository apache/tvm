apt-get update -qq && apt-get install -qq \
    # for SGX SDK
    build-essential git cmake \
    wget python pkg-config software-properties-common \
    autoconf automake libtool ocaml ocamlbuild \
    # for SGX PSW
    libssl-dev libcurl4-openssl-dev

git clone https://github.com/intel/linux-sgx.git
cd linux-sgx
git checkout sgx_2.2
curl 'https://gist.github.com/nhynes/c770b0e91610f8c020a8d1a803a1e7cb' | git am
./download_prebuilt.sh
make -j sdk && make -j sdk_install_pkg
./linux/installer/bin/sgx_linux_x64_sdk_2.2.100.45311.bin --prefix /opt
cd -

git clone https://github.com/baidu/rust-sgx-sdk.git /opt/rust-sgx-sdk
cd /opt/rust-sgx-sdk
git checkout bdd75ca05f66d1f5df637182ec335970f769b03a
cd -

curl -sSo rustup.sh 'https://sh.rustup.rs'
# rustc nightly-2018-08-25 is the version supported by the above version of rust-sgx-sdk
bash rustup.sh -y --no-modify-path --default-toolchain nightly-2018-08-25
rustup component add rust-src
cargo install rustfmt-nightly --force
cargo install xargo
