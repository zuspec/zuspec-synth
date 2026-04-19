"""CLI plugin for zuspec-synth.

Registers the ``synth`` command and the constraint pipeline transforms with the
zuspec-cli :class:`~zuspec.cli.Registry`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, TYPE_CHECKING

from zuspec.cli.plugin import Plugin
from zuspec.cli.command import Command
from zuspec.cli.transform import Transform
from zuspec.cli.ir import IR

if TYPE_CHECKING:
    from zuspec.cli.registry import Registry


# ---------------------------------------------------------------------------
# Transforms — each wraps one ConstraintCompiler phase
# ---------------------------------------------------------------------------

class _CCTransform(Transform):
    """Base for transforms that call a single method on a ConstraintCompiler."""

    _REQUIRED = "zuspec.constraint.compiler"

    @property
    def requires_ir_kind(self):
        return self._REQUIRED

    @property
    def produces_ir_kind(self):
        return self._REQUIRED

    def _call(self, cc, args):
        raise NotImplementedError

    def run(self, ir: IR, args: argparse.Namespace) -> IR:
        self._call(ir.payload, args)
        return ir


class ComputeSupportTransform(_CCTransform):
    @property
    def name(self):
        return "compute-support"

    @property
    def description(self):
        return "Compute the support (active bit-ranges) for each constraint"

    def _call(self, cc, args):
        cc.compute_support()


class BuildCubesTransform(_CCTransform):
    @property
    def name(self):
        return "build-cubes"

    @property
    def description(self):
        return "Enumerate minterms and build per-output truth tables"

    def _call(self, cc, args):
        cc.build_cubes()


class ODCTransform(_CCTransform):
    @property
    def name(self):
        return "odc"

    @property
    def description(self):
        return "Compute observability don't-cares"

    def _call(self, cc, args):
        cc.build_odc_cubes()


class MinimizeTransform(_CCTransform):
    @property
    def name(self):
        return "minimize"

    @property
    def description(self):
        return "SOP minimization via Quine-McCluskey"

    def _call(self, cc, args):
        cc.minimize()


# ---------------------------------------------------------------------------
# SynthCommand
# ---------------------------------------------------------------------------

class SynthCommand(Command):
    """``zuspec synth`` — synthesize a constraint-action class to RTL."""

    @property
    def name(self):
        return "synth"

    @property
    def description(self):
        return "Synthesize a constraint-action class to RTL"

    def add_subparser(self, subparsers):
        p = subparsers.add_parser("synth", help=self.description)
        p.add_argument(
            "files",
            nargs="*",
            metavar="FILE",
            help="Source files (SV, Python module spec, etc.)",
        )
        p.add_argument(
            "--fe",
            default="auto",
            help="Front-end to use (default: auto-detect from file extensions)",
        )
        p.add_argument(
            "--top",
            required=True,
            metavar="CLASS",
            help="Top-level action class name (or module:Class for Python FE)",
        )
        p.add_argument(
            "--be",
            default="rtl-sv",
            help="Back-end to use (default: rtl-sv)",
        )
        p.add_argument(
            "-o", "--output",
            default="-",
            metavar="FILE",
            help="Output file (default: stdout)",
        )
        p.add_argument(
            "--prefix",
            default="d",
            dest="be_prefix",
            metavar="STR",
            help="Wire-name prefix in emitted RTL (default: d)",
        )
        p.add_argument(
            "--no-odc",
            action="store_true",
            help="Skip observability don't-care optimization",
        )
        p.add_argument(
            "--no-minimize",
            action="store_true",
            help="Skip SOP minimization",
        )
        p.add_argument(
            "--warn-only",
            action="store_true",
            help="Treat constraint validation errors as warnings",
        )
        p.add_argument(
            "-f", "--filelist",
            metavar="FILELIST",
            help="File containing a list of source files (one per line)",
        )
        return p

    def run(self, args: argparse.Namespace, registry: "Registry") -> int:
        from zuspec.cli.pipeline_runner import PipelineRunner

        files: List[str] = list(args.files or [])
        if getattr(args, "filelist", None):
            files += Path(args.filelist).read_text().splitlines()

        # Resolve frontend
        if args.fe == "auto":
            fe = registry.auto_frontend(files)
        else:
            fe = registry.get_frontend(args.fe)

        # Build transform list
        xfs = [
            registry.get_transform("compute-support"),
            registry.get_transform("build-cubes"),
        ]
        if not getattr(args, "no_odc", False):
            try:
                xfs.append(registry.get_transform("odc"))
            except KeyError:
                pass
        if not getattr(args, "no_minimize", False):
            xfs.append(registry.get_transform("minimize"))

        be = registry.get_backend(args.be)
        runner = PipelineRunner(fe, xfs, be)
        return runner.run(files, args)


# ---------------------------------------------------------------------------
# SynthPlugin
# ---------------------------------------------------------------------------

class SynthPlugin(Plugin):
    """Plugin that registers the ``synth`` command and constraint transforms."""

    @property
    def name(self):
        return "zuspec-synth"

    @property
    def description(self):
        return "Constraint synthesis: parse, optimize, emit RTL"

    def register(self, registry: "Registry") -> None:
        registry.add_command(SynthCommand())
        registry.add_transform(ComputeSupportTransform())
        registry.add_transform(BuildCubesTransform())
        registry.add_transform(ODCTransform())
        registry.add_transform(MinimizeTransform())
