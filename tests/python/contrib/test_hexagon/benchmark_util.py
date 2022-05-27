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

import csv


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
                    assert isinstance(
                        old_value, float
                    ), f"Formatting code assumes that column {col_name} is some col_nameind of float, but its actual type is {type(old_value)}"
                    str_value = f"{old_value:>0.{timing_decimal_places}f}"
                    csv_line_dict[col_name] = str_value

            writer.writerow(csv_line_dict)
