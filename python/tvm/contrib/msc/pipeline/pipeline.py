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
# pylint: disable=unused-argument
"""tvm.contrib.msc.pipeline.pipeline"""

import os
import json
from typing import Any, Union, List, Tuple
import traceback

from tvm.contrib.msc.core.tools import get_tool_cls, BaseTool
from tvm.contrib.msc.core.utils.namespace import MSCFramework, MSCMap, MSCKey
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.plugin.utils import export_plugins, load_plugins
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.core import _ffi_api
from .utils import support_tool, get_tool_stage, map_tools
from .worker import BasePipeWorker


class BasePipeline(object):
    """Base Pipeline of MSC

    Parameters
    ----------
    model: Any
        The raw model in framwork.
    config: dict
        The config for pipeline.
    plugins: dict
        The plugins for pipeline.
    run_optimize: bool
        Whether to run optimize.
    run_compile: bool
        Whether to run compile.
    root: str
        The root path for files.
    """

    def __init__(
        self,
        model: Any,
        config: dict,
        plugins: dict = None,
        run_optimize: bool = True,
        run_compile: bool = True,
        root: str = None,
    ):
        # change path to root path
        if root:

            def _from_root_mark(val):
                if isinstance(val, str) and MSCKey.ROOT_MARK in val:
                    return val.replace(MSCKey.ROOT_MARK, root)
                return val

            if isinstance(model, dict):
                model = msc_utils.map_dict(model, _from_root_mark)
            elif isinstance(model, str):
                model = _from_root_mark(model)
            config = msc_utils.map_dict(config, _from_root_mark)
            plugins = msc_utils.map_dict(plugins, _from_root_mark)

        MSCMap.reset()
        self._model, self._meta_config = model, config
        self._config = msc_utils.copy_dict(config)
        if not run_optimize and MSCStage.OPTIMIZE in self._config:
            self._config.pop(MSCStage.OPTIMIZE)
        if not run_compile and MSCStage.COMPILE in self._config:
            self._config.pop(MSCStage.COMPILE)
        for stage in [MSCStage.PREPARE, MSCStage.PARSE, MSCStage.EXPORT]:
            self._config.setdefault(stage, {})
        self._verbose = self._config.get("verbose", "info")
        use_cache = self._config.get("use_cache", True)
        if "workspace" in self._config:
            self._workspace = msc_utils.set_workspace(self._config.pop("workspace"), use_cache)
        else:
            self._workspace = msc_utils.set_workspace("msc_workspace", use_cache)
        if "logger" in self._config:
            self._logger = self._config.pop("logger")
            MSCMap.set(MSCKey.GLOBALE_LOGGER, self._logger)
        else:
            if "log_file" in self._config:
                log_file = self._config.pop("log_file")
            else:
                log_file = self._workspace.relpath("MSC_LOG", keep_history=False)
            self._logger = msc_utils.set_global_logger(self._verbose, log_file)
        self._plugins = load_plugins(plugins) if plugins else {}
        self.change_stage(MSCStage.SETUP)
        self._logger.info(msc_utils.msg_block(self.pipe_mark("SETUP"), self.setup()))

    def setup(self) -> dict:
        """Setup the pipeline

        Returns
        -------
        info: dict
            The setup info.
        """

        # define run type
        self._model_type = self._config["model_type"]
        self._optimize_type = self._config.get(MSCStage.OPTIMIZE, {}).get(
            "run_type", self._model_type
        )
        self._compile_type = self._config.get(MSCStage.COMPILE, {}).get(
            "run_type", self._model_type
        )
        self._optimized, self._compiled = False, False

        # map tools
        self._tools_config = map_tools(self._config.get("tools", []))

        # register plugins
        if self._plugins:
            for t in [self._model_type, self._optimize_type, self._compile_type]:
                assert t in self._plugins, "Missing plugin for {}".format(t)
            for name, plugin in self._plugins[self._model_type].get_ops_info().items():
                _ffi_api.RegisterPlugin(name, msc_utils.dump_dict(plugin))

        # status
        self._current_stage = None
        self._report = {
            "success": False,
            "info": {},
            "duration": {},
        }
        return {
            "workspace": self._workspace.path,
            "log_file": msc_utils.get_log_file(self._logger),
            "verbose": self._verbose,
            "plugins": self._plugins,
            "config": self._config,
        }

    def run_pipe(self) -> dict:
        """Run the pipeline and return object.

        Returns
        -------
        report:
            The pipeline report.
        """

        err_msg, err_info = None, None
        try:
            self.prepare()
            self.parse()
            if MSCStage.BASELINE in self._config:
                self.baseline()
            if MSCStage.OPTIMIZE in self._config:
                self.optimize()
            if MSCStage.COMPILE in self._config:
                self.compile()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            err_msg = "Pipeline failed: " + str(exc)
            err_info = traceback.format_exc()
        self.summary(err_msg, err_info)
        self._logger.info(msc_utils.msg_block(self.pipe_mark("SUMMARY"), self._report, 0))
        self._workspace.finalize()
        return self._report

    def change_stage(self, stage: str, log_stage: bool = True) -> str:
        """Change stage

        Parameters
        ----------
        stage: str
            The stage name.
        log_stage: bool
            Whether to log the stage.

        Returns
        -------
        stage: str
            The stage name.
        """

        self._current_stage = stage
        msc_utils.time_stamp(stage, log_stage)
        return stage

    def prepare(self):
        """Prepare datas for the pipeline."""

        self.change_stage(MSCStage.PREPARE)
        info, report = self._prepare(self._get_loader(MSCStage.PREPARE))
        self._record_stage(MSCStage.PREPARE, info, report)

    def _prepare(self, data_loader: Any) -> Tuple[dict, dict]:
        """Prepare datas for the pipeline.

        Parameters
        ----------
        data_loader:
            The data loader.

        Returns
        -------
        info: dict
            The info of prepare.
        report: dict
            The report of prepare.
        """

        raise NotImplementedError("_prepare is not implemented in " + str(self.__class__))

    def parse(self):
        """Parse relax module for the pipeline."""

        self.change_stage(MSCStage.PARSE)
        info, report = self._parse()
        self._record_stage(MSCStage.PARSE, info, report)

    def _parse(self) -> Tuple[dict, dict]:
        """Parse relax module for the pipeline.

        Returns
        -------
        info: dict
            The info of parse.
        report: dict
            The report of parse.
        """

        raise NotImplementedError("_parse is not implemented in " + str(self.__class__))

    def baseline(self):
        """Run the baseline."""

        self._run_stage(MSCStage.BASELINE)

    def optimize(self) -> Tuple[dict, dict]:
        """Run the optimize.

        Returns
        -------
        info: dict
            The info of stage.
        report: dict
            The report of stage.
        """

        self._run_stage(MSCStage.OPTIMIZE)
        self._optimized = True

    def compile(self) -> Tuple[dict, dict]:
        """Run the compile.

        Returns
        -------
        info: dict
            The info of stage.
        report: dict
            The report of stage.
        """

        self._run_stage(MSCStage.COMPILE)
        self._compiled = True

    def _run_stage(self, stage: str) -> Tuple[dict, dict]:
        """Run the stage.

        Parameters
        ----------
        stage: str
            The pipeline stage.

        Returns
        -------
        info: dict
            The info of stage.
        report: dict
            The report of stage.
        """

        self.change_stage(stage)
        tools = []
        for tool in self._config.get("tools", []):
            run_type = tool.get("run_type", self._config[stage]["run_type"])
            if not support_tool(tool, stage, run_type):
                continue
            tools.append(tool["tool_type"])
            tool_cls, tool_stage = self.get_tool_cls(tool, run_type), get_tool_stage(
                tool["tool_type"]
            )
            t_stage = self.change_stage(stage + "." + tool_stage)
            if self._tool_applied(tool["tool_type"]):
                if tool_cls.apply_once():
                    msg = "Remove apply once tool " + str(tool["tool_type"])
                    self._logger.info(self.pipe_mark(msg))
                    tools = tools[:-1]
                else:
                    self._logger.info(self.pipe_mark("Apply planed tool " + str(tool["tool_type"])))
                continue
            self.change_stage(t_stage + ".build", False)
            info, report = self._create_runtime(
                t_stage, tools, run_type=run_type, visualize=False, profile=False, use_cache=False
            )
            self._record_stage(t_stage, info, report)
            knowledge, loader = None, self._get_loader(tool_stage)
            if "gym_configs" in tool:
                for idx, config in enumerate(tool["gym_configs"]):
                    knowledge_file = self._workspace.create_dir("Gym").relpath(
                        "knowledge_{}.json".format(idx)
                    )
                    gym_mark = "GYM[{}/{}]({} @ {}) ".format(
                        idx, len(tool["gym_configs"]), self._config[stage]["run_type"], tool_stage
                    )
                    if os.path.isfile(knowledge_file):
                        knowledge = knowledge_file
                        msg = "{}Load from {}".format(gym_mark, knowledge)
                        self._logger.info(self.pipe_mark(msg))
                    else:
                        self.change_stage(tool_stage + ".gym_{}".format(idx))
                        self._logger.info(self.pipe_mark(gym_mark + "Start search"))
                        knowledge = self._run_gym(tool_stage, config, knowledge, loader)
                        msc_utils.save_dict(knowledge, knowledge_file)
                    knowledge = msc_utils.load_dict(knowledge)
            self.change_stage(t_stage + ".apply", False)
            info, report = self._apply_tool(tool["tool_type"], knowledge, loader)
            self._record_stage(t_stage, info, report)
            if tool_cls.apply_once():
                msg = "Remove apply once tool " + str(tool["tool_type"])
                self._logger.info(self.pipe_mark(msg))
                tools = tools[:-1]
        self.change_stage(stage + ".build", False)
        info, report = self._create_runtime(stage, tools)
        self._record_stage(stage, info, report)

    def _tool_applied(self, tool_type: str) -> bool:
        """Check if the tool is applied

        Parameters
        ----------
        tool_type: str
            The tool type.

        Returns
        -------
        applied: bool
            Whether the tool is applied.
        """

        return False

    def _apply_tool(
        self, tool_type: str, knowledge: dict = None, data_loader: Any = None
    ) -> Tuple[dict, dict]:
        """Apply tool with runner

        Parameters
        ----------
        tool_type: str
            The tool type to apply.
        knowledge: dict
            The pre knowledge.
        data_loader:
            The data loader.

        Returns
        -------
        info: dict
            The info of apply tool.
        report: dict
            The report of apply tool.
        """

        raise NotImplementedError("_apply_tool is not implemented in " + str(self.__class__))

    def _create_runtime(
        self,
        stage: str,
        tools: List[str] = None,
        run_type: str = None,
        run_config: dict = None,
        visualize: bool = True,
        profile: bool = True,
        use_cache: bool = True,
    ) -> Tuple[dict, dict]:
        """Create runtime.

        Parameters
        ----------
        stage: str
            The pipeline stage.
        tools: list<str>
            The tools to apply.
        run_type: str
            The type of runner.
        run_config: dict
            The config of runner.
        visualize: bool
            Whether to visualize the runner
        profile: bool
            Whether to profile the runner.
        use_cache: bool
            Whether to use cache.

        Returns
        -------
        info: dict
            The info of stage.
        report: dict
            The report of stage.
        """

        raise NotImplementedError("_create_runtime is not implemented in " + str(self.__class__))

    def _run_gym(self, stage: str, config: dict, knowledge: dict, data_loader: Any) -> dict:
        """Run gym.

        Parameters
        ----------
        stage: str
            The pipeline stage.
        config: dict
            The gym config.
        knowledge: dict
            The pre knowledge.
        data_loader:
            The data loader.

        Returns
        -------
        knowledge: dict
            The learned knowledge.
        """

        raise NotImplementedError("_run_gym is not implemented in " + str(self.__class__))

    def summary(self, err_msg: str = None, err_info: str = None) -> dict:
        """Summary the pipeline.

        Parameters
        ----------
        err_msg: str
            The error message.
        err_info: str
            The error info.

        Returns
        -------
        report: dict
            The report of the pipeline.
        """

        self.change_stage(MSCStage.SUMMARY, False)
        if err_msg:
            self._report.update({"success": False, "err_msg": err_msg, "err_info": err_info})
        else:
            self._report["success"] = True
        self._report["duration"] = msc_utils.get_duration()
        return self._report

    def export(self, path: str = None, dump: bool = True) -> Union[str, dict]:
        """Export the pipeline

        Parameters
        ----------
        path: str
            The export path.
        dump: bool
            Whether to dump the info.

        Returns
        -------
        export_path/pipeline: str/dict
            The exported path/pipeline info.
        """

        path = path or "msc_export"
        if path.endswith(".tar.gz"):
            folder, dump = msc_utils.msc_dir(path.replace(".tar.gz", ""), keep_history=False), True
        else:
            folder = msc_utils.msc_dir(path, keep_history=False)

        if self._compiled:
            stage = MSCStage.COMPILE
        elif self._optimized:
            stage = MSCStage.OPTIMIZE
        else:
            stage = MSCStage.SETUP

        def _to_root_mark(val):
            if isinstance(val, str) and folder.path != val and folder.path in val:
                return val.replace(folder.path, MSCKey.ROOT_MARK)
            return val

        def _export_plugins(folder: msc_utils.MSCDirectory):
            if self._compiled:
                if dump and self.compile_type in self._plugins:
                    return self._plugins[self.compile_type].copy_libs(folder)
                return self._plugins.get(self.compile_type)
            if dump:
                return export_plugins(self._plugins, folder)
            return self._plugins

        export = {
            "logger": folder.copy(msc_utils.get_log_file(self._logger)),
            "report": self._report,
            "info": self._export_info(stage, folder.create_dir("info")),
            "model": self._export_model(stage, folder.create_dir("model"), dump),
            "plugins": _export_plugins(folder.create_dir("plugins")),
        }
        if self._compiled:
            # save golden
            num_golden = self._config[MSCStage.EXPORT].get("num_golden", 5)
            if num_golden > 0:
                saver_options = {
                    "input_names": [i[0] for i in self._config["inputs"]],
                    "output_names": self._config["outputs"],
                }
                batch_cnt, export["golden"] = 0, folder.create_dir("golden").path
                with msc_utils.IODataSaver(export["golden"], saver_options) as saver:
                    for inputs in self._get_loader()():
                        if batch_cnt >= num_golden:
                            break
                        batch_cnt = saver.save_batch(inputs, self.get_runtime().run(inputs))
        else:
            export["config"] = self.export_config(folder, dump)
        export = msc_utils.map_dict(export, _to_root_mark)
        if not dump:
            return export
        with open(folder.relpath("export.json"), "w") as f:
            f.write(json.dumps(export, indent=2))
        folder.finalize()
        if path.endswith(".tar.gz"):
            msc_utils.pack_folder(path.replace(".tar.gz", ""), "tar.gz")
        return path

    def export_config(self, folder: msc_utils.MSCDirectory, dump: bool = True) -> dict:
        """Export the config

        Parameters
        ----------
        folder: MSCDirectory
            The export folder.
        dump: bool
            Whether to dump info.

        Returns
        -------
        config: dict
            The updated config.
        """

        # dump the dataloader
        def _export_dataset(name, info, dump: bool):
            loader, max_batch = info["loader"], info.get("max_batch", -1)
            data_folder = folder.create_dir("dataset")
            if isinstance(loader, str) and msc_utils.is_callable(loader):
                path, func_name = loader.split(":")
                exp_loader = data_folder.copy(path) + ":" + func_name
            elif msc_utils.is_io_dataset(loader):
                exp_loader = data_folder.copy(loader, name)
            elif callable(loader) and dump:
                saver_options = {"input_names": [i[0] for i in self._config["inputs"]]}
                batch_cnt, exp_loader = 0, data_folder.create_dir(name).path
                with msc_utils.IODataSaver(exp_loader, saver_options) as saver:
                    for inputs in loader():
                        if batch_cnt >= max_batch > 0:
                            break
                        batch_cnt = saver.save_batch(inputs)
            else:
                exp_loader = loader
            return {"loader": exp_loader, "max_batch": max_batch}

        config = msc_utils.copy_dict(self._meta_config)
        config["dataset"] = {
            k: _export_dataset(k, v, dump) for k, v in self._config["dataset"].items()
        }
        if self._optimized:
            config["model_type"] = MSCFramework.TVM
            for stage in [MSCStage.BASELINE, MSCStage.OPTIMIZE]:
                if stage in config:
                    config.pop(stage)
            if "profile" in config[MSCStage.COMPILE] and self.get_runtime().trained:
                config[MSCStage.COMPILE]["profile"].setdefault("check", {})["err_rate"] = -1
            config["tools"] = []
            for tool in self._config.get("tools", []):
                tool_type = tool["tool_type"]
                skip_msg = "Skip export tool " + tool_type
                if not support_tool(tool, MSCStage.COMPILE, self._compile_type):
                    self._logger.info(self.pipe_mark(skip_msg + "(unsupported)"))
                    continue
                tool_cls = self.get_tool_cls(tool, self._optimize_type)
                if not tool_cls.exportable():
                    self._logger.info(self.pipe_mark(skip_msg + "(unexportable)"))
                    continue
                config["tools"].append(self._export_tool(tool_type, folder))
        # remove not serializable items
        if dump:
            remove_keys = {"workspace", "logger"}
            config = {k: v for k, v in config.items() if k not in remove_keys}
        return config

    def _export_model(self, stage: str, folder: msc_utils.MSCDirectory, dump: bool = True) -> Any:
        """Export the model

        Parameters
        ----------
        stage: str
            The pipeline stage.
        folder: MSCDirectory
            The export folder.
        dump: bool
            Whether to dump info.

        Returns
        -------
        exported:
            The exported model.
        """

        raise NotImplementedError("_export_model is not implemented in " + str(self.__class__))

    def _export_tool(self, tool_type: str, folder: msc_utils.MSCDirectory) -> dict:
        """Export the tool

        Parameters
        ----------
        tool_type: str
            The tool type.
        folder: MSCDirectory
            The export folder.

        Returns
        -------
        tool: dict
            The exported tool.
        """

        raise NotImplementedError("_export_tool is not implemented in " + str(self.__class__))

    def _export_info(self, stage: str, folder: msc_utils.MSCDirectory) -> dict:
        """Export the info of pipeline

        Parameters
        ----------
        stage: str
            The pipeline stage.
        folder: MSCDirectory
            The export folder.

        Returns
        -------
        info: dict
            The info.
        """

        return {}

    def _get_loader(self, name: str = MSCStage.PREPARE) -> Any:
        """Get the data loader"""

        config = self._config["dataset"].get(name, self._config["dataset"][MSCStage.PREPARE])
        source_loader = config.get("loader")
        assert source_loader, "Dataset loader should be given for msc pipeline"
        if source_loader == "from_random":
            max_batch = config.get("max_batch", 5)

            def get_random():
                def _to_data(inp):
                    shape = [1 if isinstance(d, str) else d for d in inp[1]]
                    return msc_utils.random_data([shape, inp[2]])

                for _ in range(max_batch):
                    yield {i[0]: _to_data(i) for i in self._config["inputs"]}

            loader, source_type = get_random, "random"
        elif isinstance(source_loader, dict):

            def load_data():
                return [source_loader]

            loader, source_type = load_data, "dict"
        elif msc_utils.is_io_dataset(source_loader):
            max_batch = config.get("max_batch", -1)

            def load_datas():
                for inputs, _ in msc_utils.IODataLoader(source_loader, end=max_batch):
                    yield inputs

            loader, source_type = load_datas, "io_data"
        elif callable(source_loader):
            max_batch = config.get("max_batch", -1)
            load_kwargs = config.get("load_kwargs", {})
            if max_batch == -1 and not load_kwargs:
                loader, source_type = source_loader, "custom"
            else:

                def get_source():
                    for idx, inputs in enumerate(source_loader(**load_kwargs)):
                        if idx >= max_batch > 0:
                            break
                        yield inputs

                loader, source_type = get_source, "loaded_custom"
        else:
            raise TypeError(
                "Unexpected source loader {}({})".format(source_loader, type(source_loader))
            )
        msg = "Create data loader({}) {}({})".format(name, loader.__name__, source_type)
        self._logger.debug(self.pipe_mark(msg))
        return loader

    def _record_stage(self, stage: str, info: dict = None, report: dict = None):
        """Record the stage

        Parameters
        -------
        stage: str
            The compile stage
        info: dict
            The info of stage.
        report: dict
            The report of stage.
        """

        if info:
            self._logger.info(msc_utils.msg_block(self.pipe_mark(stage.upper()), info))
        if report:
            self._report["info"].setdefault(stage, {}).update(report)

    def destory(self, keep_workspace: bool = False):
        """Destroy the pipeline

        Parameters
        ----------
        keep_workspace: bool
            Whether to keep workspace.
        """

        self._destory()
        if not keep_workspace:
            self._workspace.destory()
        msc_utils.remove_loggers()

    def _destory(self):
        """Destroy the pipeline."""

        raise NotImplementedError("_destory is not implemented in " + str(self.__class__))

    def get_tool_cls(self, tool: dict, framework: str) -> BaseTool:
        """Get the tool class from tool config

        Parameters
        ----------
        tool: dict
            The tool config.
        framework: str
            The framework.

        Returns
        -------
        tool_cls:
            The tool class.
        """

        return get_tool_cls(framework, tool["tool_type"], tool["tool_config"])

    def get_runtime(self, ret_type: str = "runner") -> Any:
        """Get the runtime of pipeline

        Parameters
        ----------
        ret_type: str
            The return type runner| runnable| model.

        Returns
        -------
        runnable:
            The runnable object.
        """

        raise NotImplementedError("get_runtime is not implemented in " + str(self.__class__))

    def create_worker(self, model: Any, name: str, config: dict = None):
        """Create pipe worker

        Parameters
        -------
        model: Any
            The raw model in framwork.
        name: str
            The name of worker.
        worker_config: dict
            The extra config for worker.

        Returns
        -------
        worker: str
            The message with mark.
        """

        return self.worker_cls(
            model,
            config or self._config,
            self._workspace,
            self._plugins,
            self._logger,
            name=name,
        )

    def pipe_mark(self, msg: Any) -> str:
        """Mark the message with pipeline info

        Parameters
        -------
        msg: str
            The message

        Returns
        -------
        msg: str
            The message with mark.
        """

        return "PIPE " + str(msg)

    @property
    def worker_cls(self):
        return BasePipeWorker

    @property
    def report(self):
        return self._report

    @property
    def model_type(self):
        return self._model_type

    @property
    def optimize_type(self):
        return self._optimize_type

    @property
    def compile_type(self):
        return self._compile_type
