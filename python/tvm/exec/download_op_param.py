# pylint: disable=invalid-name
"""Download pre-tuned parameters of ops"""

import argparse
import logging

from ..autotvm.record import list_pretuned_op_param, download_pretuned_op_param

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, nargs='+',
                        help="Target to download. Use 'all' to download for all targets")
    parser.add_argument("-l", action='store_true', help="List available packages")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.l:
        info = list_pretuned_op_param()
        print("\n%-20s %-20s" % ("Target", "Size"))
        print("-" * 41)
        for target, info in info:
            print("%-20s %-20s" % (target, "%.2f MB" % (info['size']/1000000)))

    if args.target:
        info = list_pretuned_op_param()
        all_targets = [x[0] for x in info]
        if 'all' in args.target:
            targets = all_targets
        else:
            targets = args.target

        for t in targets:
            if t not in all_targets:
                print("Warning : cannot find tuned parameters of " + t + ". (ignored)")
            download_pretuned_op_param(t)
