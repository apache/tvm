#!/bin/bash

set -e

# Install cross-compiler when not building natively.
# Depending on the architecture selected to compile for,
# you may need to install an alternative cross-compiler.
if [ "$architecture_type" != "aarch64" ]; then
  apt-get update && apt-get install -y --no-install-recommends \
    g++-aarch64-linux-gnu \
    gcc-aarch64-linux-gnu
fi

compute_lib_version="v21.05"
compute_lib_base_url="https://github.com/ARM-software/ComputeLibrary/releases/download/${compute_lib_version}"
compute_lib_file_name="arm_compute-${compute_lib_version}-bin-linux.tar.gz"
compute_lib_download_url="${compute_lib_base_url}/${compute_lib_file_name}"

target_lib="linux-arm64-v8a-neon"

# uncomment line below if you need asserts/debug version of the library
# target_lib="${target_lib}-asserts"

extract_dir="arm_compute-${compute_lib_version}-bin-linux"
install_path="/opt/arm/acl"

tmpdir=$(mktemp -d)

cleanup()
{
  rm -rf "$tmpdir"
}

trap cleanup 0

cd "$tmpdir"

curl -sL "${compute_lib_download_url}" -o "${compute_lib_file_name}"
tar xzf "${compute_lib_file_name}"

mkdir -p "${install_path}"
cp -r "${extract_dir}/include" "${install_path}/"
cp -r "${extract_dir}/arm_compute" "${install_path}/include/"
cp -r "${extract_dir}/support" "${install_path}/include/"
cp -r "${extract_dir}/utils" "${install_path}/include/"
cp -r "${extract_dir}/lib/${target_lib}" "${install_path}/lib"
