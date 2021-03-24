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

"""DataLoader that returns random output of specific shapes. Useful for testing."""

from tvm.data import DataLoader
import numpy as np


class RandomDataLoader(DataLoader):
    """DataLoader that creates a random input of a specific shape.
    This class is mostly used for testing, and as an example of how to
    implement a DatasetManager.

    Parameters
    ----------
    data_shapes : List of tuple of int
        List of shapes of each graph input

    dtypes : List of str
        List of dtypes of each graph input

    batch_size : int
        Size of the batch. Make sure this matches batch size in the batch dimension
        in data_shapes.

    num_batches : int
        Total number of batches in the DataLoader before we need to reset.
    """

    def __init__(self, data_shapes, dtypes, batch_size, num_batches):
        self.idx = 0
        self.data_shapes = data_shapes
        self.dtypes = dtypes
        assert len(self.data_shapes) == len(self.dtypes)
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        """Resets the DataLoader to the beginning of all the batches."""
        self.idx = 0
        return self

    def __next__(self):
        """
        Returns the next batch.

        Returns
        -------
        inputs : List of data
            List of inputs for your graph, with shapes corresponding to shapes in
            data_shapes.
        labels : None
            There is no correct label for the randomly generated data.
        """
        if self.idx >= self.num_batches:
            raise StopIteration
        self.idx += 1
        return [
            np.random.randn(*data_shape).astype(dtype)
            for (data_shape, dtype) in zip(self.data_shapes, self.dtypes)
        ], None

    def get_batch_size(self):
        """Returns the batch size.

        Returns
        -------
        batch_size : int
            The size of each batch.
        """
        return self.batch_size

    def get_num_batches(self):
        """Returns the number of batches before we need to reset.

        Returns
        -------
        num_batches : int
            The number of batches in the DataLoader.
        """
        return self.num_batches
