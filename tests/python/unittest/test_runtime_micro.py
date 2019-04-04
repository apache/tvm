import tvm
import os
import logging
import time

import numpy as np
from tvm.contrib import util
import tvm.micro

# adds two arrays and stores result into third array
def test_micro_add():
    tvm.module.load("lol", "micro_dev")
    ctx = tvm.micro_dev(0)
    pass

if __name__ == "__main__":
    test_micro_add()
