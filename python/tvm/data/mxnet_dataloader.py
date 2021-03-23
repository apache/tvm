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

"""DataLoader wrapping Mxnet datasets."""

from tvm.data import DataLoader


class MxnetDataLoader(DataLoader):
    """DataLoader wrapping an mxnet dataset.
    See
    https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/gluon/data/vision/datasets/index.html
    for an example of how to construct the keras dataset to pass into this class.

    Parameters
    ----------
    mxnet_dataset : Mxnet dataset
        Mxnet dataset containing the data.

    numpy_labels : ndarray
        An ndarray containing all the labels. The length of the ndarray must be the same as the
        Nth dimension of numpy_dataset.

    batch_size : int, optional
        Number of datapoints to put in a batch.

    num_batches : int, optional
        Number of batches to iterate through.

    layout : str, optional
        String representing the layout the numpy_dataset is in. Currently we only support NCHW as
        the layout.
    """

    def __init__(self, mxnet_loader, batch_size, num_batches):
        self.data_loader = mxnet_loader
        self.iter = iter(mxnet_loader)
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.idx = 0

    def get_next_batch(self):
        """Returns the next batch from the mxnet dataset and its labels.

        Returns
        -------
        data : List of ndarray
            List containing the data from the mxnet dataset.

        label : List of int
            List of the labels from the mxnet dataset. Length is equal to batch size.
        """
        if self.is_empty():
            raise IndexError
        self.idx += 1
        data, label = next(self.iter)
        return [data.asnumpy()], label

    def get_num_batches(self):
        """Gets the number of batches.
        Returns
        -------
        num_batches : int
            The total number of batches in the DataLoader.
        """
        return self.num_batches

    def get_batch_size(self):
        """Gets the batch size.

        Returns
        -------
        batch_size : int
            The size of the batch returned by the DataLoader.
        """
        return self.batch_size

    def is_empty(self):
        """Checks whether the DataLoader has any batches left.

        Returns
        -------
        is_empty : bool
            Whether there are any batches left in the DataLoader.
        """
        return self.idx >= self.num_batches

    def reset(self):
        """Resets the DataLoader to the beginning."""
        self.iter = iter(self.mxnet_loader)
        self.idx = 0
