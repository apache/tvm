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

import sys
import importlib
import inspect
import pkgutil
import warnings
from abc import abstractmethod


_extensions = []


class TVMExtension(object):
    @abstractmethod
    def uma_backends(self):
        return []


def get_extensions():
    for ext in _extensions:
        yield ext


def load_extensions(paths):
    path_backup = sys.path
    sys.path.extend(paths)

    top_modules = []
    try:
        mod = importlib.import_module("tvm_extension")
        top_modules.append(mod)
    except ImportError:
        pass

    sys.path = path_backup

    extension_classes = _scan_all(top_modules)
    for ext_cls in extension_classes:
        _extensions.append(ext_cls())


def _scan_all(top_level):
    scanned_extensions = []
    for mdl in top_level:
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=mdl.__path__, prefix=mdl.__name__ + ".", onerror=lambda x: None
        ):
            try:
                module_name = modname.rsplit(".", 1)[-1]
                # If module's name starts with "_", do not load the module.
                # But if the module's name starts with a "__", then load the
                # module.
                if module_name.startswith("_") and not module_name.startswith("__"):
                    continue

                with warnings.catch_warnings(record=True) as recorded_warnings:
                    if sys.version_info < (3, 10):
                        m = importer.find_module(modname)  # type: ignore
                        assert m is not None
                        loaded_mod = m.load_module(modname)
                    else:
                        spec = importer.find_spec(modname)
                        assert spec is not None
                        if modname in sys.modules:
                            loaded_mod = sys.modules[modname]
                        else:
                            loaded_mod = importlib.util.module_from_spec(spec)
                        if loaded_mod is not None:
                            spec.loader.exec_module(loaded_mod)
                            sys.modules[modname] = loaded_mod

                if len(recorded_warnings) > 0:
                    for w in recorded_warnings:
                        warnings.showwarning(
                            message=w.message,
                            category=w.category,
                            filename=w.filename,
                            lineno=w.lineno,
                            file=w.file,
                            line=w.line,
                        )

                if loaded_mod is not None:
                    for _name, obj in inspect.getmembers(loaded_mod):
                        if _is_concrete_extension_type(obj):
                            scanned_extensions.append(obj)
            except ImportError as e:
                warnings.warn(
                    message=f"\n"
                    f"\tError importing extension '{modname}'.\n"
                    f"\t\t{type(e).__name__} : {e}",
                    category=UserWarning,
                )

    return scanned_extensions


def _is_concrete_extension_type(obj):
    return inspect.isclass(obj) and issubclass(obj, TVMExtension) and not inspect.isabstract(obj)
