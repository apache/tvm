## Setup

1. [Install the Fortanix Enclave Development Platform](https://edp.fortanix.com/docs/installation/guide/)
2. `rustup component add llvm-tools-preview` to get `llvm-ar`
3. `cargo run` to start the enclave TCP server
4. Send a 28x28 "image" to the enclave model server using `head -c $((28*28*4)) /dev/urandom | nc 127.0.0.1 4242`
