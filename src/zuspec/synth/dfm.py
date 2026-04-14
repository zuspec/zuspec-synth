#****************************************************************************
#* dfm.py
#*
#* Copyright 2023-2025 Matthew Ballance and Contributors
#*
#* Licensed under the Apache License, Version 2.0 (the "License"); you may
#* not use this file except in compliance with the License.
#* You may obtain a copy of the License at:
#*
#*   http://www.apache.org/licenses/LICENSE-2.0
#*
#* Unless required by applicable law or agreed to in writing, software
#* distributed under the License is distributed on an "AS IS" BASIS,
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#* See the License for the specific language governing permissions and
#* limitations under the License.
#*
#****************************************************************************
"""DFM (dv-flow-mgr) task implementations for zuspec-synth."""

import importlib
import logging
import os
import sys
import time
from typing import List

from dv_flow.mgr import FileSet, TaskDataResult, TaskRunCtxt
from dv_flow.mgr.task_data import TaskMarker, SeverityE
from pydantic import BaseModel, Field

_log = logging.getLogger("zuspec.synth.dfm")


class SynthRTLMemento(BaseModel):
    """Memento for uptodate checking."""
    output_path: str = ""
    ref_mtime: float = 0.0
    src_files: List[str] = Field(default_factory=list)


async def check_uptodate(ctxt) -> bool:
    """Check if the synthesized SV file is up-to-date.

    Returns True if the output file exists and is newer than all
    consumed Python source files recorded in the memento.
    """
    memento = ctxt.memento
    if not memento:
        _log.debug("check_uptodate: no memento, rebuilding")
        return False

    if isinstance(memento, dict):
        memento = SynthRTLMemento(**memento)

    if not memento.output_path or not os.path.isfile(memento.output_path):
        _log.debug("check_uptodate: output file missing (%s), rebuilding", memento.output_path)
        return False

    ref_mtime = memento.ref_mtime

    for src_file in memento.src_files:
        if not os.path.isfile(src_file):
            _log.debug("check_uptodate: source file missing (%s), rebuilding", src_file)
            return False
        if os.path.getmtime(src_file) > ref_mtime:
            _log.debug("check_uptodate: source file changed (%s), rebuilding", src_file)
            return False

    _log.debug("check_uptodate: up-to-date")
    return True


async def SynthRTL(ctxt: TaskRunCtxt, input):
    """Synthesize a zuspec-dataclasses component to SystemVerilog.

    Consumes:
      - pythonSource FileSets: directories are added to sys.path so the
        component module can be imported.

    Produces:
      - systemVerilogSource FileSet pointing to the generated .sv file.
    """
    params = input.params
    markers: List[TaskMarker] = []

    # Collect Python source directories and files from consumed FileSets
    py_dirs = []
    src_files = []
    for fs in input.inputs:
        if getattr(fs, "type", None) == "std.FileSet" and \
                getattr(fs, "filetype", "") == "pythonSource":
            basedir = getattr(fs, "basedir", "")
            if basedir and basedir not in py_dirs:
                py_dirs.append(basedir)
            for f in getattr(fs, "files", []):
                full_path = os.path.join(basedir, f) if basedir else f
                if full_path not in src_files:
                    src_files.append(full_path)

    # Validate required parameter
    if not params.comp:
        markers.append(TaskMarker(
            severity=SeverityE.Error,
            msg="zuspec.synth.SynthRTL: 'comp' parameter is required (e.g. 'counter.Counter')"
        ))
        return TaskDataResult(status=1, changed=False, output=[], markers=markers)

    # Add Python source directories to sys.path for import
    for d in reversed(py_dirs):
        if d not in sys.path:
            _log.debug("Adding to sys.path: %s", d)
            sys.path.insert(0, d)

    # Parse comp parameter: "module.ClassName" or just "ClassName"
    comp_spec = params.comp
    if "." in comp_spec:
        module_name, class_name = comp_spec.rsplit(".", 1)
    else:
        module_name = comp_spec
        class_name = comp_spec

    # Import the component class
    try:
        # Re-import in case paths changed
        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)
        comp_cls = getattr(module, class_name)
        _log.debug("Imported %s from %s", class_name, module_name)
    except ImportError as e:
        markers.append(TaskMarker(
            severity=SeverityE.Error,
            msg=f"zuspec.synth.SynthRTL: failed to import '{module_name}': {e}"
        ))
        return TaskDataResult(status=1, changed=False, output=[], markers=markers)
    except AttributeError:
        markers.append(TaskMarker(
            severity=SeverityE.Error,
            msg=f"zuspec.synth.SynthRTL: class '{class_name}' not found in '{module_name}'"
        ))
        return TaskDataResult(status=1, changed=False, output=[], markers=markers)

    # Determine output filename
    output_name = params.output or f"{class_name}.sv"
    output_path = os.path.join(ctxt.rundir, output_name)

    # Run synthesis
    from zuspec.synth import synthesize
    try:
        synthesize(
            comp_cls,
            output=output_path,
            reset_style=params.reset_style,
            forward=params.forward,
        )
        _log.info("Synthesized %s -> %s", comp_spec, output_path)
    except Exception as e:
        _log.exception("Synthesis failed for %s", comp_spec)
        markers.append(TaskMarker(
            severity=SeverityE.Error,
            msg=f"zuspec.synth.SynthRTL: synthesis failed for '{comp_spec}': {e}"
        ))
        return TaskDataResult(status=1, changed=False, output=[], markers=markers)

    # Build output FileSet
    fs_out = FileSet(
        filetype="systemVerilogSource",
        basedir=ctxt.rundir,
        files=[output_name],
    )

    # Store memento for uptodate checking
    memento = SynthRTLMemento(
        output_path=output_path,
        ref_mtime=time.time(),
        src_files=src_files,
    )

    return TaskDataResult(
        status=0,
        changed=True,
        output=[fs_out],
        memento=memento,
        markers=markers,
    )
