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

"""Dataloader wrapping TFDatasets."""

from tvm.relay.data import Dataloader


class TFDataloader(Dataloader):
    """DatasetManager wrapping a tensorflow dataset.
    See https://www.tensorflow.org/datasets/keras_example for an example of how to construct a tensorflow
    dataset to use as input to this class.

    Parameters
    ----------
    tf_dataset : Tensorflow dataset
        Tensorflow keras dataset

    batch_size : int
        Size of the batch to get. Must match the batch size of the tensorflow dataset passed in.

    num_batches : int
        Total number of batches to return before you need to reset. num_batches should be less than
        or equal to the number of batches in the tf_dataset.
    """

    def __init__(self, tf_dataset, batch_size, num_batches):
        self.idx = 0
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.tf_dataset = tf_dataset
        self.tf_iter = iter(self.tf_dataset)

    def get_next_batch(self):
        """Returns the next batch from the tensorflow dataset and its labels.

        Returns
        -------
        data : List of ndarray
            List containing the data from the tensorflow dataset.

        label : List of int
            List of the labels from the tensorflow dataset. Length is equal to batch size.
        """
        if self.is_empty():
            raise IndexError
        self.idx += 1

        data, label = next(self.tf_iter)

        return [data.numpy()], label.numpy()

    def get_num_batches(self):
        """Gets the number of batches.
        Returns
        -------
        num_batches : int
            The total number of batches in the Dataloader.
        """
        return self.num_batches

    def get_batch_size(self):
        """Gets the batch size.

        Returns
        -------
        batch_size : int
            The size of the batch returned by the Dataloader.
        """
        return self.batch_size

    def is_empty(self):
        """Checks whether the Dataloader has any batches left.

        Returns
        -------
        is_empty : bool
            Whether there are any batches left in the Dataloader.
        """
        return self.idx >= self.total_batches

    def reset(self):
        """Resets the Dataloader to the beginning of all the datapoints."""
        self.tf_iter = iter(self.tf_dataset)
        self.idx = 0
