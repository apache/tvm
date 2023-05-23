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

""" Utility functions used for benchmarks """

import csv
import os
import tempfile

import pytest

from tvm.contrib.hexagon.tools import HEXAGON_SIMULATOR_NAME


def skip_benchmarks_flag_and_reason():
    """
    Returns one of these tuples:
        (False, '') or
        (True, (a string describing why the test should be skipped))

    NOTE: This function is a temporary measure to prevent the TVM CI system
    running benchmark scripts every time the CI pre-commit hook executes.
    This should go away when a better system is in place to govern when various
    tests / benchmarks are executed.
    """
    asn = os.environ.get("ANDROID_SERIAL_NUMBER")

    if asn == HEXAGON_SIMULATOR_NAME:
        return (True, "Skipping benchmarks when  ANDROID_SERIAL_NUMBER='simluator'")

    return (False, "")


class UnsupportedException(Exception):
    """
    Indicates that the specified benchmarking configuration is known to
    currently be unsupported.  The Exception message may provide more detail.
    """


class NumericalAccuracyException(Exception):
    """
    Indicates that the benchmarking configuration appeared to run successfully,
    but the output data didn't have the expected accuracy.
    """


class BenchmarksTable:
    """
    Stores/reports the result of benchmark runs.

    Each line item has a status: success, fail, or skip.

    Each 'success' line item must include benchmark data,
    in the form provided by TVM's `time_evaluator` mechanism.

    Each line item may also specify values for any subset of
    the columns provided to the table's construstor.
    """

    BUILTIN_COLUMN_NAMES = set(
        [
            "row_status",
            "timings_min_usecs",
            "timings_max_usecs",
            "timings_median_usecs",
            "timings_mean_usecs",
            "timings_stddev_usecs",
        ]
    )

    def __init__(self):
        self._line_items = []

    def validate_user_supplied_kwargs(self, kwarg_dict):
        name_conflicts = set(kwarg_dict).intersection(self.BUILTIN_COLUMN_NAMES)

        if name_conflicts:
            name_list = ", ".join(name_conflicts)
            raise Exception(f"Attempting to supply values for built-in column names: {name_list}")

    def record_success(self, timings, **kwargs):
        """
        `timings` : Assumed to have the structure and meaning of
          the timing results provided by TVM's `time_evaluator`
          mechanism.

        `kwargs` : Optional values for any of the other columns
          defined for this benchmark table.
        """
        self.validate_user_supplied_kwargs(kwargs)
        line_item = kwargs

        line_item["row_status"] = "SUCCESS"

        line_item["timings_min_usecs"] = timings.min * 1000000
        line_item["timings_max_usecs"] = timings.max * 1000000
        line_item["timings_median_usecs"] = timings.median * 1000000
        line_item["timings_stddev_usecs"] = timings.std * 1000000
        line_item["timings_mean_usecs"] = timings.mean * 1000000

        self._line_items.append(line_item)

    def record_skip(self, **kwargs):
        self.validate_user_supplied_kwargs(kwargs)

        line_item = dict(kwargs)
        line_item["row_status"] = "SKIP"
        self._line_items.append(line_item)

    def record_fail(self, **kwargs):
        self.validate_user_supplied_kwargs(kwargs)

        line_item = dict(kwargs)
        line_item["row_status"] = "FAIL"
        self._line_items.append(line_item)

    def has_fail(self):
        """
        Returns True if the table contains at least one 'fail' line item,
        otherwise returns False.
        """
        return any(item["row_status"] == "FAIL" for item in self._line_items)

    def print_csv(self, f, column_name_order, timing_decimal_places=3):
        """
        Print the benchmark results as a csv.

        `f` : The output stream.

        `column_name_order`: an iterable sequence of column names, indicating the
           left-to-right ordering of columns in the CSV output.

           The CSV output will contain only those columns that are mentioned in
           this list.

        `timing_decimal_places`: for the numeric timing values, this is the
           number of decimal places to provide in the printed output.
           For example, a value of 3 is equivalent to the Python formatting string
           `'{:.3f}'`
        """
        writer = csv.DictWriter(
            f, column_name_order, dialect="excel-tab", restval="", extrasaction="ignore"
        )

        writer.writeheader()

        for line_item_dict in self._line_items:
            # Use a copy of the line-item dictionary, because we might do some modifications
            # for the sake of rendering...
            csv_line_dict = dict(line_item_dict)

            for col_name in [
                "timings_min_usecs",
                "timings_max_usecs",
                "timings_median_usecs",
                "timings_stddev_usecs",
                "timings_mean_usecs",
            ]:
                if col_name in csv_line_dict:
                    old_value = csv_line_dict[col_name]
                    assert isinstance(old_value, float), (
                        f"Formatting code assumes that column {col_name} is"
                        f" some col_nameind of float, but its actual type is {type(old_value)}"
                    )
                    str_value = f"{old_value:>0.{timing_decimal_places}f}"
                    csv_line_dict[col_name] = str_value

            writer.writerow(csv_line_dict)


def get_benchmark_id(keys_dict):
    """
    Given a dictionary with the distinguishing characteristics of a particular benchmark
    line item, compute a string that uniquely identifies the benchmark.

    The returned string:
    - is a valid directory name on the host's file systems, and
    - should be easy for humans to parse

    Note that the insertion order for `keys_dict` affects the computed name.
    """
    # Creat a copy, because we might be modifying it.
    keys_dict_copy = dict(keys_dict)

    # Sniff for shape-like lists, because we want them in a form that's both
    # readable and filesystem-friendly...
    for k, v in keys_dict_copy.items():
        if isinstance(v, (list, tuple)):
            v_str = "_".join([str(x) for x in v])
            keys_dict_copy[k] = v_str

    return "-".join([f"{k}:{v}" for k, v in keys_dict_copy.items()])


def get_benchmark_decription(keys_dict):
    """
    Similar to `get_benchmark_id`, but the focus is on human-readability.

    The returned string contains no line-breaks, but may contain spaces and
    other characters that make it unsuitable for use as a filename.
    """
    return " ".join([f"{k}={v}" for k, v in keys_dict.items()])


@pytest.fixture(scope="class")
def benchmark_group(request):
    """This fixture provides some initialization / finalization logic for groups of related
    benchmark runs.
    See the fixture implementation below for details.

    The fixture's mechanics are described here: https://stackoverflow.com/a/63047695

    TODO: There may be cleaner ways to let each class that uses this fixture provide its
    own value for `csv_column_order`.

    TODO: In the future we may wish to break this fixture up in to several smaller ones.

    The overall contract for a class (e.g. `MyTest`) using this fixture is as follows:

        https://stackoverflow.com/a/63047695

        @pytest.mark.usefixtures("benchmark_group")
        class MyTest:

        # The fixture requires that this class variable is defined before
        # the fixture's finalizer-logic executes.
        #
        # This is used as an argument to BenchmarkTable.print_csv(...) after
        # all of MyTest's unit tests have executed.
        csv_column_order = [
            ...
            ]

        # Before the MyTest's first unit test executes, the fixture will populate the
        # following class variables:
        MyTest.working_dir     : str
        MyTest.benchmark_table : BenchmarkTable"""
    working_dir = tempfile.mkdtemp()
    table = BenchmarksTable()

    request.cls.working_dir = working_dir
    request.cls.benchmark_table = table

    yield

    tabular_output_filename = os.path.join(working_dir, "benchmark-results.csv")

    if not hasattr(request.cls, "csv_column_order"):
        raise Exception('Classes using this fixture must have a member named "csv_column_order"')

    with open(tabular_output_filename, "w", encoding="UTF-8") as csv_file:
        table.print_csv(csv_file, request.cls.csv_column_order)

    print()
    print("*" * 80)
    print(f"BENCHMARK RESULTS FILE: {tabular_output_filename}")
    print("*" * 80)
    print()

    if table.has_fail() > 0:
        pytest.fail("At least one benchmark configuration failed", pytrace=False)
