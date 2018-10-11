apt-get update && apt-get install -y --no-install-recommends --force-yes curl

export RUSTUP_HOME=/opt/rust
export CARGO_HOME=/opt/rust
# this rustc is one supported by the installed version of rust-sgx-sdk
curl https://sh.rustup.rs -sSf | sh -s -- -y --no-modify-path --default-toolchain nightly-2018-09-25
. $CARGO_HOME/env
rustup toolchain add nightly
rustup component add rust-src
cargo +nightly install rustfmt-nightly --version 0.99.5 --force
cargo +nightly install xargo

# make rust usable by all users
chmod a+w /opt/rust
sudo find /opt/rust -type d -exec chmod a+w {} \;
