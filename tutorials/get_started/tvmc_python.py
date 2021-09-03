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
"""
Getting Starting using TVMC Python, the simplified tvm API
==========================================
**Authors**:
`Jocelyn Shiue <https://github.com/CircleSpin>`_,

<H1> Welcome to TVMC Python #TODO
Hi! Here we explain the scripting tool designed for the complete TVM beginner. 🙂 

################################################################################
# Step 0: Imports
# ---------------
# 
# .. code-block:: python
#
#     from tvm.driver import tvmc
#

################################################################################
# Step 1: Load a model
# --------------------
# Let's import our model into tvmc.
# Let's import our model into tvmc. This step converts a machine learning model from 
# a supported framework into tvm's high level graph representation language called relay. 
# This is to have a unified starting point for all models in tvm. The frameworks we currently 
# support are: Keras, Onnx, Tensorflow, TFLite, and Pytorch.
# 
# .. code-block:: python
#      model = tvmc.load('my_model.onnx') #Step 1: Load
# 
# If you'd like to see the relay, you can run: 
# ``model.summary()``
# 

################################################################################
# Step 2: Compile
# ----------------
# Now that our model is in relay, our next step is to compile it to a desired 
# hardware to run on. We refer to this hardware as a target. This compilation process 
# translates the model from relay into a lower-level language that the 
# target machine can understand. 
# 
# In order to compile a model a tvm.target string is required. 
# To learn more about tvm.targets and their options look at the `documentation <https://tvm.apache.org/docs/api/python/target.html>`_. 
# Some examples include:
# 1. cuda (nvidia gpu)
# 2. llvm (cpu)
# 3. llvm -mcpu=cascadelake (intel cpu)
#
#  .. code-block:: python
#      package = tvmc.compile(model, target="llvm") #Step 2: Compile
# 
# The compilation step returns a package.
# 

################################################################################
# Step 3: Run
# -----------
# The compiled package can now be run on the hardware target. The device 
# input options are: cpu, cuda, cl, metal, and vulkan.
# 
#  .. code-block:: python
#      result = tvmc.run(package, device="cpu") #Step 3: Run
# 
# And you can print the results:
# ``print(results)``
# 

################################################################################
# Step 1.5: Tune [Optional & Recommended]
# ---------------------------------------
# Run speed can further be improved by tuning. This optional step uses 
# machine learning to look at each operation within a model (a function) and 
# tries to find a faster way to run it. We do this through a cost model, and 
# bench marking possible schedules.
# 
# The target is the same as compile. 
# 
#  .. code-block:: python
#      tvmc.tune(model, target="llvm") #Step 1.5: Optional Tune
# 
# The terminal output should look like: 
# [Task  1/13]  Current/Best:   82.00/ 106.29 GFLOPS | Progress: (48/769) | 18.56 s
# [Task  1/13]  Current/Best:   54.47/ 113.50 GFLOPS | Progress: (240/769) | 85.36 s
# .....
# 
# There may be UserWarnings that can be ignored. 
# This should make the end result faster, but it can take hours to tune. 
# 

################################################################################
# Save and then start the process in the terminal:
# ------------------------------------------------
# 
#  .. code-block:: python
#      python my_tvmc_script.py
# 
# Note: Your fans may become very active
# 

################################################################################
# Example results:
# ----------------
# 
#   .. code-block:: python
#      Time elapsed for training: 18.99 s
#      Execution time summary:
#      mean (ms)   max (ms)   min (ms)   std (ms) 
#        25.24      26.12      24.89       0.38 
#      
#      Output Names:
#       ['output_0']
#

<H1> Additional TVMC Functionalities #TODO

################################################################################
# Saving the model
# ----------------
# 
# To make things faster for later, after loading the model (Step 1) save the relay version. 
# 
#   .. code-block:: python
#      model = tvmc.load('my_model.onnx') #Step 1: Load
#      model.save(model_path) 
# 

################################################################################
# Saving the package
# ------------------
# 
# After the model has been compiled (Step 2) the package also is also saveable.  
# 
#    .. code-block:: python
#      tvmc.compile(model, target="llvm", package_path="whatever") #Step 2: Compile
#      
#      new_package = tvmc.TVMCPackage(package_path="whatever") 
#      result = tvmc.run(new_package) #Step 3: Run
# 

################################################################################
# Using Autoscheduler
# -------------------
# Use the next generation of tvm to enable potentially faster run speed results. 
# The search space of the schedules is automatically generated unlike 
# previously where they needed to be hand written. (Learn more: 1, 2) 
# 
#    .. code-block:: python
#      tvmc.tune(model, target="llvm", enable_autoscheduler = True) #Step 1.5: Optional Tune
#

################################################################################
# Saving the tuning results
# -------------------------
# 
# The tuning results can be saved in a file for later reuse.
# 
# Method 1:
#    .. code-block:: python
#      log_file = "hello.json"
#      
#      # Run tuning
#      tvmc.tune(model, target="llvm",tuning_records=log_file)
#      
#      ...
#      
#      # Later run tuning and reuse tuning results
#      tvmc.tune(model, target="llvm",tuning_records=log_file)
#  
# Method 2:
#    .. code-block:: python
#      # Run tuning
#      tuning_records = tvmc.tune(model, target="llvm")
#      
#      ...
#      
#      # Later run tuning and reuse tuning results
#      tvmc.tune(model, target="llvm",tuning_records=tuning_records)
# 

################################################################################
# Using an RPC Server:
# 
# 
# This thing needs some love
# 
# 
# 
# 
# 
# 

################################################################################
# Tuning a more complex model:
# ----------------------------
# If you notice T's (timeouts) printed, increase the searching time frame: 
# 
#    .. code-block:: python
#      tvmc.tune(model,trials=10000,timeout=10,)
#

"""