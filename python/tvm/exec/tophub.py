# pylint: disable=invalid-name
"""Download pre-tuned parameters of ops"""

import argparse
import logging

from ..autotvm.tophub import list_packages, download_package

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", type=str, nargs='+',
                        help="Target to download. Use 'all' to download for all targets")
    parser.add_argument("-l", "--list", action='store_true', help="List available packages")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.list:
        info = list_packages()
        print("\n%-20s %-20s" % ("Target", "Size"))
        print("-" * 41)
        for target, info in info:
            print("%-20s %-20s" % (target, "%.2f MB" % (info['size']/1000000)))

    if args.download:
        info = list_packages()
        all_targets = [x[0] for x in info]
        if 'all' in args.download:
            targets = all_targets
        else:
            targets = args.download

        for t in targets:
            if t not in all_targets:
                print("Warning : cannot find tuned parameters of " + t + ". (ignored)")
            download_package(t)
