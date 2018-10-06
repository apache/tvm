apt-get update && apt-get install -y --no-install-recommends --force-yes curl

curl -sSo rustup.sh 'https://sh.rustup.rs'
# rustc nightly-2018-08-25 is the version supported by the above version of rust-sgx-sdk
bash rustup.sh -y --no-modify-path --default-toolchain nightly-2018-08-25
. $HOME/.cargo/env
rustup toolchain add nightly
rustup component add rust-src
cargo +nightly install rustfmt-nightly --version 0.99.5 --force
cargo +nightly install xargo
