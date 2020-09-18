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

"""Wrapper classes to expose datasets during quantization."""

import numpy as np


class DatasetManager:
    """Simple wrapper class to expose datasets in quantization."""

    def get_next_batch(self):
        """Returns the next batch of data.

        Returns
        -------
        inputs : List
            The inputs to be provided to the graph.
            The list is of the form [batched_input_1, batched_input_2, ..., batched_input_n]

        labels: List
            The expected outputs of the graph.
            The length of labels should be equal to the batch size.
        """
        raise NotImplementedError

    def batch_size(self):
        """Returns the size of each batch the dataset manager has.

        Returns
        -------
        batch_size : int
            The number of inputs in each batch.
        """

    def num_batches(self):
        """Returns the number of batches the dataset manager has.

        Returns
        ------
        num_batches : int
            The number of batches the dataset manager contains.
        """
        raise NotImplementedError

    def is_empty(self):
        """Checks whether the dataset manager has gone through
        all its batches.
        Returns
        -------
        is_empty : bool
            True if there are batches left, False if there are no more
            batches.
        """
        raise NotImplementedError

    def reset(self):
        """Resets the counter in the dataset manager to the beginning."""
        raise NotImplementedError


class TFDatasetManager(DatasetManager):
    """DatasetManager wrapping a tensorflow dataset."""

    def __init__(self, tf_dataset, batch_size, total_batches):
        self.idx = 0
        self.total_batches = total_batches
        self.batch_sz = batch_size
        self.tf_dataset = tf_dataset
        self.tf_iter = iter(self.tf_dataset)

    def get_next_batch(self):
        if self.is_empty():
            raise IndexError
        self.idx += 1

        data, label = next(self.tf_iter)

        return [data.numpy()], label.numpy()

    def num_batches(self):
        return self.total_batches

    def batch_size(self):
        return self.batch_sz

    def is_empty(self):
        return self.idx >= self.total_batches

    def reset(self):
        self.tf_iter = iter(self.tf_dataset)
        self.idx = 0


class RandomDatasetManager(DatasetManager):
    """DatasetManager that creates a random input of a specific shape.
    This class is mostly used for testing, and as an example of how to
    implement a DatasetManager.
    """

    def __init__(self, data_shape, dtype, batch_size, total_batches):
        self.idx = 0
        self.data_shape = data_shape
        self.dtype = dtype
        self.batch_sz = batch_size
        self.total_batches = total_batches

    def get_next_batch(self):
        if self.is_empty():
            raise IndexError
        self.idx += 1
        return [np.random.randn(*self.data_shape).astype(self.dtype)], None

    def batch_size(self):
        return self.batch_sz

    def num_batches(self):
        return self.total_batches

    def is_empty(self):
        return self.idx >= self.total_batches

    def reset(self):
        self.idx = 0
