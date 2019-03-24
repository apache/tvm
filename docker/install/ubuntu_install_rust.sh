#!/bin/bash

set -e
set -u
set -o pipefail

apt-get update && apt-get install -y --no-install-recommends curl

export RUSTUP_HOME=/opt/rust
export CARGO_HOME=/opt/rust
# this rustc is one supported by the installed version of rust-sgx-sdk
curl -s -S -L https://sh.rustup.rs -sSf | sh -s -- -y --no-modify-path --default-toolchain nightly-2019-03-24
. $CARGO_HOME/env
rustup component add rustfmt
cargo install sccache --no-default-features

# make rust usable by all users
chmod -R a+w /opt/rust
