TVM Documentations
==================
This folder contains the source of TVM documents

- A hosted version of doc is at https://tvm.apache.org/docs
- pip install "sphinx>=1.5.5" sphinx-gallery sphinx_rtd_theme matplotlib Image recommonmark "Pillow<7" "autodocsumm<0.2.0" tlcpack-sphinx-addon "docutils<0.17"
- (Versions 0.2.0 to 0.2.2 of autodocsumm are incompatible with sphinx>=3.4, https://github.com/Chilipp/autodocsumm/pull/42 )
- Build tvm first in the root folder.
- Run the following command
```bash
TVM_TUTORIAL_EXEC_PATTERN=none make html
```

```TVM_TUTORIAL_EXEC_PATTERN=none``` skips the tutorial execution to make it work on most environment(e.g. Mac book).

See also the instructions below to run a specific tutorial. Note that some of the tutorials need GPU support.


Only Execute Specified Tutorials
--------------------------------
The document build process will execute all the tutorials in the sphinx gallery.
This will cause failure in some cases when certain machines do not have necessary
environment. You can set ```TVM_TUTORIAL_EXEC_PATTERN``` to only execute
the path that matches the regular expression pattern.

For example, to only build tutorials under /vta/tutorials, run

```bash
TVM_TUTORIAL_EXEC_PATTERN=/vta/tutorials make html
```

To only build one specific file, do

```bash
# The slash \ is used to get . in regular expression
TVM_TUTORIAL_EXEC_PATTERN=file_name\.py make html
```

Helper Scripts
--------------

You can run the following script to reproduce the CI sphinx pre-check stage.
This script skips the tutorial executions and is useful for quickly check the content.

```bash
./tests/scripts/task_sphinx_precheck.sh
```

The following script runs the full build which includes tutorial executions.
You will need a gpu CI environment.

```bash
./tests/scripts/task_python_docs.sh
```

Define the Order of Tutorials
-----------------------------
You can define the order of tutorials with `conf.py::subsection_order` and `conf.py::within_subsection_order`.
By default, the tutorials within one subsection is sorted by filename.
