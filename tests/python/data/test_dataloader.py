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

from tvm.data import RandomDataLoader, TFDataLoader, NumpyDataLoader
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

tf.enable_v2_behavior()

import tensorflow.keras.datasets as keras_datasets
import numpy as np


def verify_dataloader_output(
    dataloader, data_shapes, data_dtypes, expected_batch_size, expected_num_batches, labels=False
):
    num_batches = 0

    assert dataloader.get_batch_size() == expected_batch_size
    assert dataloader.get_num_batches() == expected_num_batches

    actual_num_batches = 0
    while not dataloader.is_empty():
        data, labels = dataloader.get_next_batch()
        # Labels is None or there are batch_size number of labels
        assert labels is None or len(labels == expected_batch_size)
        assert len(data) == len(data_shapes) and len(data) == len(data_dtypes)
        for i in range(len(data)):
            assert data[i].shape == data_shapes[i]
            print(data[i].dtype)
            assert data[i].dtype == data_dtypes[i]
        actual_num_batches += 1

    assert actual_num_batches == expected_num_batches


def verify_random_dataloader(data_shapes, data_dtypes, batch_size, num_batches):
    dl = RandomDataLoader(data_shapes, data_dtypes, batch_size, num_batches)
    verify_dataloader_output(dl, data_shapes, data_dtypes, batch_size, num_batches)


def test_random_dataloader():
    verify_random_dataloader([(1, 2, 3)], ["float32"], 2, 4)
    verify_random_dataloader([(3, 4, 5), (1, 4, 3)], ["float32", "int8"], 5, 4)
    verify_random_dataloader([(3, 4, 5), (1, 4, 3)], ["float32", "int8"], 0, 4)
    verify_random_dataloader([], [], 0, 0)


def verify_tf_dataloader(batch_size, num_batches):
    # TFDS loading from https://www.tensorflow.org/datasets/keras_example
    (_, ds_test), ds_info = tfds.load(
        "mnist", split=["train", "test"], shuffle_files=True, as_supervised=True, with_info=True
    )

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255.0, label

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    dl = TFDataLoader(ds_test, batch_size, num_batches)
    verify_dataloader_output(dl, [(batch_size, 28, 28, 1)], ["float32"], batch_size, num_batches)


def test_tf_dataloader():
    verify_tf_dataloader(2, 4)
    verify_tf_dataloader(5, 20)


def verify_numpy_dataloader(test_images, test_labels, batch_size, num_batches):
    dl = NumpyDataLoader(test_images, test_labels, batch_size, num_batches)
    # If num_batches is None, we calculate the number of batches in the DataLoader
    if num_batches is None:
        num_batches = test_images.shape[0] // batch_size
    verify_dataloader_output(dl, [(batch_size, 32, 32, 3)], ["float32"], batch_size, num_batches)


def test_numpy_dataloader():
    # Keras dataloader from https://www.tensorflow.org/tutorials/images/cnn
    (_, _), (test_images, test_labels) = keras_datasets.cifar10.load_data()
    normal_test_images = (test_images / 255.0).astype("float32")
    flat_test_labels = np.ndarray.flatten(test_labels)

    verify_numpy_dataloader(normal_test_images, flat_test_labels, 2, 2)
    verify_numpy_dataloader(normal_test_images, flat_test_labels, 5, 4)
    verify_numpy_dataloader(normal_test_images[0:4], flat_test_labels[0:4], 1, None)
    verify_numpy_dataloader(normal_test_images[0:4], flat_test_labels[0:4], 3, None)


def verify_mxnet_dataloader():
    pass


def test_mxnet_dataloader():
    pass


if __name__ == "__main__":
    test_random_dataloader()
    test_tf_dataloader()
    test_numpy_dataloader()
