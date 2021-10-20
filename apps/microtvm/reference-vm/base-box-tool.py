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

# List of supported electronics platforms. Each must correspond
# to a sub-directory of this directory.
ALL_PLATFORMS = (
    "arduino",
    "zephyr",
)

# Extra scripts required to execute on provisioning
# in [platform]/base-box/base_box_provision.sh
EXTRA_SCRIPTS = {
    "arduino": (),
    "zephyr": ("docker/install/ubuntu_init_zephyr_project.sh",),
}

PACKER_FILE_NAME = "packer.json"


# List of identifying strings for microTVM boards for testing.
# TODO add a way to declare supported boards to ProjectAPI
ALL_MICROTVM_BOARDS = {
    "arduino": (
        "due",
        "feathers2",
        "metrom4",
        "nano33ble",
        "pybadge",
        "spresense",
        "teensy40",
        "teensy41",
        "wioterminal",
    ),
    "zephyr": (
        "nucleo_f746zg",
        "stm32f746g_disco",
        "nrf5340dk_nrf5340_cpuapp",
        "mps2_an521",
    ),
}


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
            # TODO(mehrdadh): skip usb attach if it's already attached
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


def generate_packer_config(platform, file_path, providers):
    builders = []
    provisioners = []
    for provider_name in providers:
        builders.append(
            {
                "name": f"{provider_name}",
                "type": "vagrant",
                "box_name": f"microtvm-base-{provider_name}",
                "output_dir": f"output-packer-{provider_name}",
                "communicator": "ssh",
                "source_path": "generic/ubuntu1804",
                "provider": provider_name,
                "template": "Vagrantfile.packer-template",
            }
        )

    repo_root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], encoding="utf-8"
    ).strip()
    for script in EXTRA_SCRIPTS[platform]:
        script_path = os.path.join(repo_root, script)
        filename = os.path.basename(script_path)
        provisioners.append({"type": "file", "source": script_path, "destination": f"~/{filename}"})

    provisioners.append(
        {
            "type": "shell",
            "script": "base_box_provision.sh",
        }
    )

    with open(file_path, "w") as f:
        json.dump(
            {
                "builders": builders,
                "provisioners": provisioners,
            },
            f,
            sort_keys=True,
            indent=2,
        )


def build_command(args):
    generate_packer_config(
        args.platform,
        os.path.join(THIS_DIR, args.platform, "base-box", PACKER_FILE_NAME),
        args.provider or ALL_PROVIDERS,
    )
    env = copy.copy(os.environ)
    packer_args = ["packer", "build"]
    env["PACKER_LOG"] = "1"
    env["PACKER_LOG_PATH"] = "packer.log"
    if args.debug_packer:
        packer_args += ["-debug"]

    packer_args += [PACKER_FILE_NAME]
    subprocess.check_call(
        packer_args, cwd=os.path.join(THIS_DIR, args.platform, "base-box"), env=env
    )


REQUIRED_TEST_CONFIG_KEYS = {
    "vid_hex": str,
    "pid_hex": str,
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


def do_run_release_test(release_test_dir, platform, provider_name, test_config, test_device_serial):
    with open(
        os.path.join(release_test_dir, ".vagrant", "machines", "default", provider_name, "id")
    ) as f:
        machine_uuid = f.read()

    # Check if target is not QEMU
    if test_config["vid_hex"] and test_config["pid_hex"]:
        ATTACH_USB_DEVICE[provider_name](
            machine_uuid,
            vid_hex=test_config["vid_hex"],
            pid_hex=test_config["pid_hex"],
            serial=test_device_serial,
        )
    tvm_home = os.path.realpath(os.path.join(THIS_DIR, "..", "..", ".."))

    def _quote_cmd(cmd):
        return " ".join(shlex.quote(a) for a in cmd)

    test_cmd = (
        _quote_cmd(["cd", tvm_home])
        + " && "
        + _quote_cmd(
            [
                f"apps/microtvm/reference-vm/{platform}/base-box/base_box_test.sh",
                test_config["microtvm_board"],
            ]
        )
    )
    subprocess.check_call(["vagrant", "ssh", "-c", f"bash -ec '{test_cmd}'"], cwd=release_test_dir)


def test_command(args):
    user_box_dir = os.path.join(THIS_DIR, args.platform)
    base_box_dir = os.path.join(THIS_DIR, args.platform, "base-box")
    test_config_file = os.path.join(base_box_dir, "test-config.json")
    with open(test_config_file) as f:
        test_config = json.load(f)

        # select microTVM test config
        microtvm_test_config = test_config[args.microtvm_board]

        for key, expected_type in REQUIRED_TEST_CONFIG_KEYS.items():
            assert key in microtvm_test_config and isinstance(
                microtvm_test_config[key], expected_type
            ), f"Expected key {key} of type {expected_type} in {test_config_file}: {test_config!r}"

        microtvm_test_config["vid_hex"] = microtvm_test_config["vid_hex"].lower()
        microtvm_test_config["pid_hex"] = microtvm_test_config["pid_hex"].lower()
        microtvm_test_config["microtvm_board"] = args.microtvm_board

    providers = args.provider

    release_test_dir = os.path.join(THIS_DIR, f"release-test-{args.platform}")

    if args.skip_build or args.skip_destroy:
        assert (
            len(providers) == 1
        ), "--skip-build and/or --skip-destroy was given, but >1 provider specified"

    test_failed = False
    for provider_name in providers:
        try:
            if not args.skip_build:
                do_build_release_test_vm(
                    release_test_dir, user_box_dir, base_box_dir, provider_name
                )
            do_run_release_test(
                release_test_dir,
                args.platform,
                provider_name,
                microtvm_test_config,
                args.test_device_serial,
            )

        except subprocess.CalledProcessError:
            test_failed = True
            sys.exit(
                f"\n\nERROR: Provider '{provider_name}' failed the release test. "
                "You can re-run it to reproduce the issue without building everything "
                "again by passing the --skip-build and specifying only the provider that failed. "
                "The VM is still running in case you want to connect it via SSH to "
                "investigate further the issue, thus it's necessary to destroy it manually "
                "to release the resources back to the host, like a USB device attached to the VM."
            )

        finally:
            # if we reached out here do_run_release_test() succeeded, hence we can
            # destroy the VM and release the resources back to the host if user haven't
            # requested to not destroy it.
            if not (args.skip_destroy or test_failed):
                subprocess.check_call(["vagrant", "destroy", "-f"], cwd=release_test_dir)
                shutil.rmtree(release_test_dir)

    print(f'\n\nThe release tests passed on all specified providers: {", ".join(providers)}.')


def release_command(args):
    vm_name = f"tlcpack/microtvm-{args.platform}-{args.platform_version}"

    if not args.skip_creating_release_version:
        subprocess.check_call(
            [
                "vagrant",
                "cloud",
                "version",
                "create",
                vm_name,
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
                vm_name,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automates building, testing, and releasing a base box"
    )
    subparsers = parser.add_subparsers(help="Action to perform.")
    parser.add_argument(
        "--provider",
        choices=ALL_PROVIDERS,
        action="append",
        required=True,
        help="Name of the provider or providers to act on",
    )

    # "test" has special options for different platforms, and "build", "release" might
    # in the future, so we'll add the platform argument to each one individually.
    platform_help_str = "Platform to use (e.g. Arduino, Zephyr)"

    # Options for build subcommand
    parser_build = subparsers.add_parser("build", help="Build a base box.")
    parser_build.set_defaults(func=build_command)
    parser_build.add_argument("platform", help=platform_help_str, choices=ALL_PLATFORMS)
    parser_build.add_argument(
        "--debug-packer",
        action="store_true",
        help=("Run packer in debug mode, and write log to the base-box directory."),
    )

    # Options for test subcommand
    parser_test = subparsers.add_parser("test", help="Test a base box before release.")
    parser_test.set_defaults(func=test_command)
    parser_test.add_argument(
        "--skip-build",
        action="store_true",
        help=(
            "If given, assume a box has already been built in the release-test subdirectory, "
            "so use that box to execute the release test script. If the tests fail the VM used "
            "for testing will be left running for further investigation and will need to be "
            "destroyed manually. If all tests pass on all specified providers no VM is left running, "
            "unless --skip-destroy is given too."
        ),
    )
    parser_test.add_argument(
        "--skip-destroy",
        action="store_true",
        help=(
            "Skip destroying the test VM even if all tests pass. Can only be used if a single "
            "provider is specified. Default is to destroy the VM if all tests pass (and always "
            "skip destroying it if a test fails)."
        ),
    )
    parser_test.add_argument(
        "--test-device-serial",
        help=(
            "If given, attach the test device with this USB serial number. Corresponds to the "
            "iSerial field from `lsusb -v` output."
        ),
    )
    parser_test_platform_subparsers = parser_test.add_subparsers(help=platform_help_str)
    for platform in ALL_PLATFORMS:
        platform_specific_parser = parser_test_platform_subparsers.add_parser(platform)
        platform_specific_parser.set_defaults(platform=platform)
        platform_specific_parser.add_argument(
            "--microtvm-board",
            choices=ALL_MICROTVM_BOARDS[platform],
            required=True,
            help="MicroTVM board used for testing.",
        )

    # Options for release subcommand
    parser_release = subparsers.add_parser("release", help="Release base box to cloud.")
    parser_release.set_defaults(func=release_command)
    parser_release.add_argument("platform", help=platform_help_str, choices=ALL_PLATFORMS)
    parser_release.add_argument(
        "--release-version",
        required=True,
        help="Version to release, in the form 'x.y.z'. Must be specified with release.",
    )
    parser_release.add_argument(
        "--skip-creating-release-version",
        action="store_true",
        help="Skip creating the version and just upload for this provider.",
    )
    parser_release.add_argument(
        "--platform-version",
        required=True,
        help=(
            "For Zephyr, the platform version to release, in the form 'x.y'. "
            "For Arduino, the version of arduino-cli that's being used, in the form 'x.y.z'."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.sep in args.platform or not os.path.isdir(os.path.join(THIS_DIR, args.platform)):
        sys.exit(f"<platform> must be a sub-direcotry of {THIS_DIR}; got {args.platform}")

    args.func(args)


if __name__ == "__main__":
    main()
