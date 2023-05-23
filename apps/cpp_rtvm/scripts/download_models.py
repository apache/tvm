#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

tmp_dir = "./model_data/"
dload_models = []

# Keras : Resnet50
try:
    from tensorflow.keras.applications.resnet50 import ResNet50

    model_file_name = "{}/{}".format(tmp_dir + "keras-resnet50", "resnet50.h5")
    model = ResNet50(include_top=True, weights="imagenet", input_shape=(224, 224, 3), classes=1000)
    model.save(model_file_name)
    dload_models.append(model_file_name)
except ImportError:
    LOG.warning("Keras is not installed, skipping Keras models")


print("Models:", dload_models)
