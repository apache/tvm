"""VTA config tool"""
import os
import sys
import json
import argparse

def get_pkg_config(cfg):
    """Get the pkg config object."""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    proj_root = os.path.abspath(os.path.join(curr_path, "../"))
    pkg_config_py = os.path.join(proj_root, "python/vta/pkg_config.py")
    libpkg = {"__file__": pkg_config_py}
    exec(compile(open(pkg_config_py, "rb").read(), pkg_config_py, "exec"), libpkg, libpkg)
    PkgConfig = libpkg["PkgConfig"]
    return PkgConfig(cfg, proj_root)


def main():
    """Main funciton"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cflags", action="store_true",
                        help="print the cflags")
    parser.add_argument("--update", action="store_true",
                        help="Print out the json option.")
    parser.add_argument("--ldflags", action="store_true",
                        help="print the cflags")
    parser.add_argument("--cfg-json", action="store_true",
                        help="print all the config json")
    parser.add_argument("--target", action="store_true",
                        help="print the target")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    curr_path = os.path.dirname(
        os.path.abspath(os.path.expanduser(__file__)))
    proj_root = os.path.abspath(os.path.join(curr_path, "../"))
    path_list = [
        os.path.join(proj_root, "config.json"),
        os.path.join(proj_root, "make/config.json")
    ]
    ok_path_list = [p for p in path_list if os.path.exists(p)]
    if not ok_path_list:
        raise RuntimeError("Cannot find config in %s" % str(path_list))
    cfg = json.load(open(ok_path_list[0]))
    cfg["LOG_OUT_WIDTH"] = cfg["LOG_INP_WIDTH"]
    pkg = get_pkg_config(cfg)

    if args.target:
        print(pkg.target)

    if args.cflags:
        print(" ".join(pkg.cflags))

    if args.ldflags:
        print(" ".join(pkg.ldflags))

    if args.cfg_json:
        print(pkg.cfg_json)


if __name__ == "__main__":
    main()
