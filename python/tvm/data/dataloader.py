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

"""A universal API for handling data in TVM. Useful for when you may want to use datasets from
different machine learning frameworks interchangably. """


class DataLoader:
    """Wrapper class for data loader or data set classes implemented by other machine learning
    frameworks. Use this class when you want to use different machine learning framework datasets
    interchangably."""

    def __iter__(self):
        """Returns the DataLoaderIterator."""
        return self

    def __next__(self):
        """Returns the next batch of data.

        Returns
        -------
        inputs : List of ndarray
            The inputs to be provided to the graph.
            The list is of the form [batched_input_1, batched_input_2, ..., batched_input_n]

        labels: List
            The expected outputs of the graph.
            The length of labels should be equal to the batch size. If the DataLoader doesn't
            have labels, labels will be None.
        """
        raise NotImplementedError

    def get_num_batches(self):
        """Returns the number of batches the DataLoader has.

        Returns
        ------
        num_batches : int
            The number of batches the DataLoader contains.
        """
        raise NotImplementedError

    def get_batch_size(self):
        """Gets the batch size.

        Returns
        -------
        batch_size : int
            The size of the batch returned by the DataLoader.
        """
        return raise NotImplementedError
