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

import json
import csv
import subprocess
import argparse
import os
from re import search, compile
from collections import OrderedDict

ENABLE_DEBUG = False
"""
Process lightweight profiling output and generate a CSV file with processor
cycles for the instrumented functions and loops.

Please note that some assumptions have been made while processing
the lightweight profiling output. They are as follows:

1) We don't expect profiled functions to call another profiled function.
  This constraint can be relaxed if needed but it simplifies the processing
  significantly without introducing any limitations for our use case.
2) For now, it's also assumed that every unique section (loop) ID has same start
  and end offset which will not be true while a loop gets unrolled as it will
  create multiple profiling section with the same ID. The current
  implementation doesn't handle this case.

"""


def get_func_info(model_so):
    """Get all the .text sections along with their start and end offset values"""
    hexagon_nm_path = os.environ["HEXAGON_TOOLCHAIN"] + "/bin/hexagon-nm"
    out = subprocess.Popen(
        [hexagon_nm_path, "--print-size", model_so],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    stdo, stde = out.communicate()
    stdo = stdo.decode("utf-8")

    func_info = []
    for l in stdo.split("\n"):
        info = {}
        if search(" (T|t) ", l):  # If .text section
            parts = l.split(" ")
            assert len(parts) == 4
            info["start"] = int(parts[0], base=16)
            info["end"] = int(parts[0], base=16) + int(parts[1], base=16)
            info["name"] = parts[3]
            func_info.append(info)

    # Sort the entries in the increasing order of the start offset value.
    func_info = sorted(func_info, key=lambda d: d["start"])

    if ENABLE_DEBUG:
        print("func_info :\n ")
        for f in func_info:
            print(f)
    return func_info


def find_func(func_info, offset):
    """For a given offset, find the function it belongs to."""
    fidx = 0
    lidx = len(func_info) - 1
    while fidx <= lidx:
        midx = (fidx + lidx) // 2
        ms = func_info[midx]["start"]
        me = func_info[midx]["end"]
        if fidx == lidx:
            assert (
                offset >= ms and offset <= me
            ), f"Couldn't find a function for this offset: {offset}"
            return fidx
        else:
            if offset > me:
                fidx = midx + 1
            elif offset < ms:
                lidx = midx - 1
            else:
                return midx
    assert False, "Possible mismatch between model .so and LWP data"


def accumulate_cycles(overall_cycles, func_cycles, func_name):
    """Accumulate function cycles"""
    acc_cycles = overall_cycles[func_name]
    for id in func_cycles:
        assert id in acc_cycles, f"id [{id}] missing in the existing function record"
        assert (
            acc_cycles[id]["start"] == func_cycles[id]["start"]
        ), "Offset value doesn't match with the existing function record."
        acc_cycles[id]["cycles"] += func_cycles[id]["cycles"]
        acc_cycles[id]["count"] += func_cycles[id]["count"]
    overall_cycles.update({func_name: acc_cycles})
    return overall_cycles


def adjust_per_loop_counts(overall_cycles, data):
    """
    Use execution count and the number of entries recorded for each function/loop
    to compute the overall cycles spent on them.
    """
    for func in overall_cycles:
        func_cycles = overall_cycles[func]
        for id in func_cycles:
            exec_count = data["loop_counts"][id]
            rec_count = func_cycles[id]["count"]
            assert exec_count != 0, "Execution count should have been non-zero."
            assert rec_count != 0, "Entry count should have been non-zero."
            exec_cycles = ((int(func_cycles[id]["cycles"])) * exec_count) // rec_count
            func_cycles[id]["cycles"] = exec_cycles
            func_cycles[id]["count"] = exec_count
        overall_cycles.update({func: OrderedDict(sorted(func_cycles.items()))})
    return overall_cycles


def create_csv_report(overall_cycles, fname):
    """Create csv report"""
    header = [
        "function name",
        "loop/function id",
        "loop depth",
        "start offset",
        "end offset",
        "pcycles",
        "parent count",
    ]
    with open(fname, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for func in overall_cycles:
            func_cycles = overall_cycles[func]
            data = []
            root = -1
            outer_most = -1
            for key, value in func_cycles.items():
                if value["parent"] == -1:
                    assert root == -1, "Can't have multiple root nodes."
                    root = key

                data.append(func)
                data.append(key)
                if value["parent"] == -1:
                    data.append("-")  # Total cycles over all invocations of this function.
                elif value["parent"] == root:
                    data.append(0)
                    outer_most = key
                else:
                    if outer_most > -1:
                        data.append(key - outer_most)
                    else:
                        data.append(key - value["parent"])
                data.append(hex(value["start"]))
                data.append(hex(value["end"]))
                data.append(value["cycles"])
                data.append(value["count"])
                writer.writerow(data)
                data.clear()


def process_data(data, func_info, so_ld_addr):
    """Process data"""
    # Keep an ordered list of loop IDs as they are being visited. This is used
    # to match entry and exit pairs. Once the function/loop is processed, it's
    # removed from the list.
    ordered_visited_list = []
    # Store information regarding visited nodes as they are being processed. Once
    # the function/loop is processed, it's removed from the set.
    visited_set = {}
    # Dictionary to store cycles for the entire model which is grouped into functions.
    overall_cycles = {}
    func_cycles = {}

    func_idx = -1
    func_name = ""
    prev_func_name = ""
    func_start = 0
    func_end = 0
    save_data = False
    # Iterate over all the entries in the LWP data file and process them
    # to construct a report.
    for entry in data["entries"]:
        id = entry["id"]
        offset = entry["ret"] - so_ld_addr

        # Recorded return address should fall within the function begin and end
        # offsets. If not, find the function it belongs to.
        if offset < func_start or offset > func_end:
            prev_func_name = func_name
            if ENABLE_DEBUG:
                print("offset : ", offset)
                print("id : ", id)

            func_idx = find_func(func_info, offset)
            func_name = func_info[func_idx]["name"]
            func_start = func_info[func_idx]["start"]
            func_end = func_info[func_idx]["end"]
            if ENABLE_DEBUG:
                print("func_name : ", func_name)

            if save_data:
                # overall_cycles = save_func_cycles(prev_func_name, overall_cycles, func_cycles, ordered_visited_list)
                # Done processing the previous function, copy its info into 'overall_cycles'.
                if prev_func_name not in overall_cycles:
                    overall_cycles[prev_func_name] = func_cycles.copy()
                else:
                    # Accumulate cycles into existing function entry.
                    overall_cycles = accumulate_cycles(overall_cycles, func_cycles, prev_func_name)
                # We don't allow for fused operators (functions) calling another operator.
                if ENABLE_DEBUG:
                    print("ordered_visited_list : ", ordered_visited_list)

                assert len(ordered_visited_list) == 0, (
                    f"\nDone processing function [{prev_func_name}] but ordered_visited_list not empty.\n"
                    f"\t Possible reasons -- \n"
                    f"\t\t1) Mismatch between model .so and json file.\n"
                    f"\t\t2) LWP buffer may have overflowed resulting into missing entries!"
                )
                func_cycles.clear()

            save_data = True

        if id not in visited_set:  # Found 'entry' record
            visited_info = {"func_idx": func_idx, "ret": offset, "cyc": entry["cyc"]}
            visited_set[id] = visited_info
            ordered_visited_list.append(id)
        else:  # Found 'exit' record
            # This should be the last entry in the ordered_visited_list. If not, error out.
            assert ordered_visited_list[-1] == id, (
                "Problem with LWP output - Interleaved handler calls found."
                f"Loop [{ordered_visited_list[-1]}] hasn't exited yet."
            )
            ordered_visited_list.pop()
            entry_node = visited_set.pop(id)
            assert (
                entry_node["func_idx"] == func_idx
            ), f'Error - Found under a different function name : {entry_node["func_idx"]}'
            cycles = entry["cyc"] - entry_node["cyc"]
            parent = -1
            if ordered_visited_list:
                parent = int(ordered_visited_list[-1])
            if id in func_cycles:
                fcycles = func_cycles[id]
                fcycles["cycles"] += cycles
                fcycles["count"] += 1
                func_cycles[id] = fcycles
            else:
                func_cycles[id] = {
                    "cycles": cycles,
                    "start": entry_node["ret"],
                    "end": offset,
                    "parent": parent,
                    "count": 1,
                }

    # Done processing the previous function, copy its info into 'overall_cycles'.
    if func_name not in overall_cycles:
        overall_cycles[func_name] = func_cycles.copy()
    else:
        # Accumulate cycles into existing function entry.
        overall_cycles = accumulate_cycles(overall_cycles, func_cycles, func_name)
    # We don't allow for fused operators (functions) calling another operator.
    if ENABLE_DEBUG:
        print("ordered_visited_list : ", ordered_visited_list)

    assert len(ordered_visited_list) == 0, (
        f"\nDone processing function [{prev_func_name}] but ordered_visited_list not empty.\n"
        f"\t Possible reasons -- \n"
        f"\t\t1) Mismatch between model .so and json file.\n"
        f"\t\t2) LWP buffer may have overflowed resulting into missing entries!"
    )

    overall_cycles = adjust_per_loop_counts(overall_cycles, data)
    return overall_cycles


def get_load_addr(serial_number: str, lwp_json: str, run_log: str):
    """Get load address of the binary file"""
    if serial_number == "simulator":
        basedir = os.path.dirname(lwp_json)
        if run_log is None:
            run_log = os.path.join(basedir, "stdout.txt")
        else:
            # If the directory name is specified for the run_log of the
            # simulator (stdout.txt) then it must be same as lwp_json.
            run_log_dir = os.path.dirname(run_log)
            assert (
                run_log_dir == "" or run_log_dir == basedir
            ), f"stdout.txt and {os.path.basename(lwp_json)} must be in the same directory"
            run_log = os.path.join(basedir, os.path.basename(run_log))
        # To extract load address for the simulator run
        pattern = compile(r"Model.*: (\w+):")
    else:
        # To extract load address for on-device run
        pattern = compile(r"Model.*: (\w+)")

    with open(run_log, "r") as f:
        lines = f.read()
        a = pattern.search(lines)
        load_addr = int(a.group(1), 16)
    if ENABLE_DEBUG:
        print("load_addr : ", load_addr)
    return load_addr


def process_lwp_output(
    binary_path: str,
    serial_number: str,
    lwp_json: str,
    run_log: str,
    lwp_out: str,
    enable_debug: bool = False,
):
    """Process lightweight profiling data"""
    # Enable debug messages
    global ENABLE_DEBUG
    ENABLE_DEBUG = enable_debug

    # Get load address for the binary
    load_addr = get_load_addr(serial_number, lwp_json, run_log)
    # Opening JSON file
    with open(lwp_json, "r") as f:
        # Returns JSON object as a dictionary
        data = json.load(f)

    # Get function names, and their start and end offsets from the model .so
    func_info = get_func_info(binary_path)

    # Get the load address for model .so.
    so_ld_addr = load_addr

    # Process profiling data to construct a CSV report.
    overall_cycles = process_data(data, func_info, so_ld_addr)
    create_csv_report(overall_cycles, lwp_out)
    print("lwp processed output written to -- ", lwp_out)
    print("[NOTE: Use '--hexagon-debug' to keep the temp directory]")


def get_args():
    """Add commandline arguments to run the script manually if needed"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lwp-json", help="LWP json file", required=True)
    parser.add_argument("--serial-num", help="device-id/simulator", required=True)
    parser.add_argument("--test-so", help="Test shared library", required=True)
    parser.add_argument(
        "--run-log",
        help="Logcat file for on-device run and stdout.txt for simulator run",
        required=True,
    )
    parser.add_argument("--lwp-out", help="LWP output file name", required=True)
    parser.add_argument(
        "--debug",
        help="Enable debug output from the script",
        dest="debug",
        action="store_true",
        required=False,
    )
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    global ENABLE_DEBUG
    ENABLE_DEBUG = args.debug

    return args


if __name__ == "__main__":
    args = get_args()
    process_lwp_output(
        args.test_so, args.serial_num, args.lwp_json, args.run_log, args.lwp_out, args.debug
    )
