#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import argparse
import copy
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys


_LOG = logging.getLogger(__name__)


THIS_DIR = os.path.realpath(os.path.dirname(__file__) or ".")


# List of vagrant providers supported by this tool
ALL_PROVIDERS = (
    "parallels",
    "virtualbox",
    "vmware_desktop",
)


def parse_virtualbox_devices():
    output = subprocess.check_output(["VBoxManage", "list", "usbhost"], encoding="utf-8")
    devices = []
    current_dev = {}
    for line in output.split("\n"):
        if not line.strip():
            if current_dev:
                if "VendorId" in current_dev and "ProductId" in current_dev:
                    devices.append(current_dev)
                current_dev = {}

            continue

        key, value = line.split(":", 1)
        value = value.lstrip(" ")
        current_dev[key] = value

    if current_dev:
        devices.append(current_dev)
    return devices


VIRTUALBOX_VID_PID_RE = re.compile(r"0x([0-9A-Fa-f]{4}).*")


def attach_virtualbox(uuid, vid_hex=None, pid_hex=None, serial=None):
    usb_devices = parse_virtualbox_devices()
    for dev in usb_devices:
        m = VIRTUALBOX_VID_PID_RE.match(dev["VendorId"])
        if not m:
            _LOG.warning("Malformed VendorId: %s", dev["VendorId"])
            continue

        dev_vid_hex = m.group(1).lower()

        m = VIRTUALBOX_VID_PID_RE.match(dev["ProductId"])
        if not m:
            _LOG.warning("Malformed ProductId: %s", dev["ProductId"])
            continue

        dev_pid_hex = m.group(1).lower()

        if (
            vid_hex == dev_vid_hex
            and pid_hex == dev_pid_hex
            and (serial is None or serial == dev["SerialNumber"])
        ):
            rule_args = [
                "VBoxManage",
                "usbfilter",
                "add",
                "0",
                "--action",
                "hold",
                "--name",
                "test device",
                "--target",
                uuid,
                "--vendorid",
                vid_hex,
                "--productid",
                pid_hex,
            ]
            if serial is not None:
                rule_args.extend(["--serialnumber", serial])
            subprocess.check_call(rule_args)
            subprocess.check_call(["VBoxManage", "controlvm", uuid, "usbattach", dev["UUID"]])
            return

    raise Exception(
        f"Device with vid={vid_hex}, pid={pid_hex}, serial={serial!r} not found:\n{usb_devices!r}"
    )


def attach_parallels(uuid, vid_hex=None, pid_hex=None, serial=None):
    usb_devices = json.loads(
        subprocess.check_output(["prlsrvctl", "usb", "list", "-j"], encoding="utf-8")
    )
    for dev in usb_devices:
        _, dev_vid_hex, dev_pid_hex, _, _, dev_serial = dev["System name"].split("|")
        dev_vid_hex = dev_vid_hex.lower()
        dev_pid_hex = dev_pid_hex.lower()
        if (
            vid_hex == dev_vid_hex
            and pid_hex == dev_pid_hex
            and (serial is None or serial == dev_serial)
        ):
            subprocess.check_call(["prlsrvctl", "usb", "set", dev["Name"], uuid])
            if "Used-By-Vm-Name" in dev:
                subprocess.check_call(
                    ["prlctl", "set", dev["Used-By-Vm-Name"], "--device-disconnect", dev["Name"]]
                )
            subprocess.check_call(["prlctl", "set", uuid, "--device-connect", dev["Name"]])
            return

    raise Exception(
        f"Device with vid={vid_hex}, pid={pid_hex}, serial={serial!r} not found:\n{usb_devices!r}"
    )


def attach_vmware(uuid, vid_hex=None, pid_hex=None, serial=None):
    print("NOTE: vmware doesn't seem to support automatic attaching of devices :(")
    print("The VMWare VM UUID is {uuid}")
    print("Please attach the following usb device using the VMWare GUI:")
    if vid_hex is not None:
        print(f" - VID: {vid_hex}")
    if pid_hex is not None:
        print(f" - PID: {pid_hex}")
    if serial is not None:
        print(f" - Serial: {serial}")
    if vid_hex is None and pid_hex is None and serial is None:
        print(" - (no specifications given for USB device)")
    print()
    print("Press [Enter] when the USB device is attached")
    input()


ATTACH_USB_DEVICE = {
    "parallels": attach_parallels,
    "virtualbox": attach_virtualbox,
    "vmware_desktop": attach_vmware,
}


def generate_packer_config(file_path, providers):
    builders = []
    for provider_name in providers:
        builders.append(
            {
                "type": "vagrant",
                "box_name": f"microtvm-base-{provider_name}",
                "output_dir": f"output-packer-{provider_name}",
                "communicator": "ssh",
                "source_path": "generic/ubuntu1804",
                "provider": provider_name,
                "template": "Vagrantfile.packer-template",
            }
        )

    with open(file_path, "w") as f:
        json.dump(
            {
                "builders": builders,
            },
            f,
            sort_keys=True,
            indent=2,
        )


def build_command(args):
    generate_packer_config(
        os.path.join(THIS_DIR, args.platform, "base-box", "packer.json"),
        args.provider or ALL_PROVIDERS,
    )
    env = None
    packer_args = ["packer", "build"]
    if args.debug_packer:
        env = copy.copy(os.environ)
        env["PACKER_LOG"] = "1"
        env["PACKER_LOG_PATH"] = "packer.log"
        packer_args += ["-debug"]

    packer_args += ["packer.json"]
    subprocess.check_call(
        packer_args, cwd=os.path.join(THIS_DIR, args.platform, "base-box"), env=env
    )


REQUIRED_TEST_CONFIG_KEYS = {
    "vid_hex": str,
    "pid_hex": str,
    "test_cmd": list,
}


VM_BOX_RE = re.compile(r'(.*\.vm\.box) = "(.*)"')


# Paths, relative to the platform box directory, which will not be copied to release-test dir.
SKIP_COPY_PATHS = [".vagrant", "base-box"]


def do_build_release_test_vm(release_test_dir, user_box_dir, base_box_dir, provider_name):
    if os.path.exists(release_test_dir):
        try:
            subprocess.check_call(["vagrant", "destroy", "-f"], cwd=release_test_dir)
        except subprocess.CalledProcessError:
            _LOG.warning("vagrant destroy failed--removing dirtree anyhow", exc_info=True)

        shutil.rmtree(release_test_dir)

    for dirpath, _, filenames in os.walk(user_box_dir):
        rel_path = os.path.relpath(dirpath, user_box_dir)
        if any(
            rel_path == scp or rel_path.startswith(f"{scp}{os.path.sep}") for scp in SKIP_COPY_PATHS
        ):
            continue

        dest_dir = os.path.join(release_test_dir, rel_path)
        os.makedirs(dest_dir)
        for filename in filenames:
            shutil.copy2(os.path.join(dirpath, filename), os.path.join(dest_dir, filename))

    release_test_vagrantfile = os.path.join(release_test_dir, "Vagrantfile")
    with open(release_test_vagrantfile) as f:
        lines = list(f)

    found_box_line = False
    with open(release_test_vagrantfile, "w") as f:
        for line in lines:
            m = VM_BOX_RE.match(line)
            if not m:
                f.write(line)
                continue

            box_package = os.path.join(
                base_box_dir, f"output-packer-{provider_name}", "package.box"
            )
            box_relpath = os.path.relpath(box_package, release_test_dir)
            f.write(f'{m.group(1)} = "{box_relpath}"\n')
            found_box_line = True

    if not found_box_line:
        _LOG.error(
            "testing provider %s: couldn't find config.box.vm = line in Vagrantfile; unable to test",
            provider_name,
        )
        return False

    # Delete the old box registered with Vagrant, which may lead to a falsely-passing release test.
    remove_args = ["vagrant", "box", "remove", box_relpath]
    return_code = subprocess.call(remove_args, cwd=release_test_dir)
    assert return_code in (0, 1), f'{" ".join(remove_args)} returned exit code {return_code}'
    subprocess.check_call(["vagrant", "up", f"--provider={provider_name}"], cwd=release_test_dir)

    return True


def do_run_release_test(release_test_dir, provider_name, test_config, test_device_serial):
    with open(
        os.path.join(release_test_dir, ".vagrant", "machines", "default", provider_name, "id")
    ) as f:
        machine_uuid = f.read()
    ATTACH_USB_DEVICE[provider_name](
        machine_uuid,
        vid_hex=test_config["vid_hex"],
        pid_hex=test_config["pid_hex"],
        serial=test_device_serial,
    )
    tvm_home = os.path.realpath(os.path.join(THIS_DIR, "..", "..", ".."))

    def _quote_cmd(cmd):
        return " ".join(shlex.quote(a) for a in cmd)

    test_cmd = _quote_cmd(["cd", tvm_home]) + " && " + _quote_cmd(test_config["test_cmd"])
    subprocess.check_call(["vagrant", "ssh", "-c", f"bash -ec '{test_cmd}'"], cwd=release_test_dir)


def test_command(args):
    user_box_dir = os.path.join(THIS_DIR, args.platform)
    base_box_dir = os.path.join(THIS_DIR, args.platform, "base-box")
    test_config_file = os.path.join(base_box_dir, "test-config.json")
    with open(test_config_file) as f:
        test_config = json.load(f)
        for key, expected_type in REQUIRED_TEST_CONFIG_KEYS.items():
            assert key in test_config and isinstance(
                test_config[key], expected_type
            ), f"Expected key {key} of type {expected_type} in {test_config_file}: {test_config!r}"

        test_config["vid_hex"] = test_config["vid_hex"].lower()
        test_config["pid_hex"] = test_config["pid_hex"].lower()

    providers = args.provider
    provider_passed = {p: False for p in providers}

    release_test_dir = os.path.join(THIS_DIR, "release-test")

    if args.skip_build:
        assert len(providers) == 1, "--skip-build was given, but >1 provider specified"

    for provider_name in providers:
        try:
            if not args.skip_build:
                do_build_release_test_vm(
                    release_test_dir, user_box_dir, base_box_dir, provider_name
                )
            do_run_release_test(
                release_test_dir, provider_name, test_config, args.test_device_serial
            )
            provider_passed[provider_name] = True

        finally:
            if not args.skip_build and len(providers) > 1:
                subprocess.check_call(["vagrant", "destroy", "-f"], cwd=release_test_dir)
                shutil.rmtree(release_test_dir)

        if not all(provider_passed[p] for p in provider_passed.keys()):
            sys.exit(
                "some providers failed release test: "
                + ",".join(name for name, passed in provider_passed if not passed)
            )


def release_command(args):
    if not args.skip_creating_release_version:
        subprocess.check_call(
            [
                "vagrant",
                "cloud",
                "version",
                "create",
                f"tlcpack/microtvm-{args.platform}",
                args.release_version,
            ]
        )
    if not args.release_version:
        sys.exit(f"--release-version must be specified")

    for provider_name in args.provider:
        subprocess.check_call(
            [
                "vagrant",
                "cloud",
                "publish",
                "-f",
                f"tlcpack/microtvm-{args.platform}",
                args.release_version,
                provider_name,
                os.path.join(
                    THIS_DIR,
                    args.platform,
                    "base-box",
                    f"output-packer-{provider_name}/package.box",
                ),
            ]
        )


ALL_COMMANDS = {
    "build": build_command,
    "test": test_command,
    "release": release_command,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automates building, testing, and releasing a base box"
    )
    parser.add_argument(
        "command",
        default=",".join(ALL_COMMANDS),
        choices=ALL_COMMANDS,
        help="Action or actions (comma-separated) to perform.",
    )
    parser.add_argument(
        "platform",
        help="Name of the platform VM to act on. Must be a sub-directory of this directory.",
    )
    parser.add_argument(
        "--provider",
        choices=ALL_PROVIDERS,
        action="append",
        default=[],
        help="Name of the provider or providers to act on; if not specified, act on all",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help=(
            "For use with the 'test' command. If given, assume a box has already been built in "
            "the release-test subdirectory. Attach a USB device to this box and execute the "
            "release test script--do not delete it."
        ),
    )
    parser.add_argument(
        "--test-device-serial",
        help=(
            "If given, attach the test device with this USB serial number. Corresponds to the "
            "iSerial field from `lsusb -v` output."
        ),
    )
    parser.add_argument(
        "--release-version",
        help="Version to release, in the form 'x.y.z'. Must be specified with release.",
    )
    parser.add_argument(
        "--skip-creating-release-version",
        action="store_true",
        help="With release, skip creating the version and just upload for this provider.",
    )
    parser.add_argument(
        "--debug-packer",
        action="store_true",
        help=(
            "When the build command is given, run packer in debug mode, and write log to the "
            "base-box directory"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.sep in args.platform or not os.path.isdir(os.path.join(THIS_DIR, args.platform)):
        sys.exit(f"<platform> must be a sub-direcotry of {THIS_DIR}; got {args.platform}")

    if not args.provider:
        args.provider = list(ALL_PROVIDERS)

    todo = []
    for phase in args.command.split(","):
        if phase not in ALL_COMMANDS:
            sys.exit(f"unknown command: {phase}")

        todo.append(ALL_COMMANDS[phase])

    for phase in todo:
        phase(args)


if __name__ == "__main__":
    main()
