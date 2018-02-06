"""measure bandwidth and compute peak

e.g.
python3 -m tvm.exec.measure_peak --target cuda --rpc-host 0.0.0.0 --rpc-port 9090
python3 -m tvm.exec.measure_peak --target opencl --target-host "llvm -target=aarch64-linux-gnu" \
        --rpc-host $TVM_OPENCL_DEVICE_HOST --rpc-port 9090
"""

import argparse
import logging

from ..contrib.peak import measure_peak_all

def main():
    """Main funciton"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default="llvm",
                        help='The build target')
    parser.add_argument('--target-host', type=str, default=None,
                        help='The host code compilation target')
    parser.add_argument('--rpc-host', type=str, default="0.0.0.0",
                        help='the hostname of the server')
    parser.add_argument('--rpc-port', type=int, default=9090,
                        help='The port of the PRC')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    measure_peak_all(args.target, args.target_host, args.rpc_host, args.rpc_port)

if __name__ == "__main__":
    main()
