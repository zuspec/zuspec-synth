"""Tests for the zuspec-synth CLI plugin (SynthCommand + transforms).

These tests exercise the plugin classes directly, without going through the
full ``zuspec`` entry-point, so they can run in isolation without entry-points
being installed.
"""
from __future__ import annotations

import argparse
import sys
import pytest

import zuspec.dataclasses as zdc

# Ensure the synth plugin can be imported
from zuspec.synth.cli_plugin import (
    SynthPlugin,
    SynthCommand,
    ComputeSupportTransform,
    BuildCubesTransform,
    ODCTransform,
    MinimizeTransform,
)
from zuspec.cli.registry import Registry
from zuspec.cli.ir import IR


# ---------------------------------------------------------------------------
# Tiny action class for testing
# ---------------------------------------------------------------------------

@zdc.dataclass
class _TwoOutput(zdc.Action):
    instr: zdc.u4 = zdc.input()
    out_a: zdc.u1 = zdc.rand()
    out_b: zdc.u1 = zdc.rand()

    @zdc.constraint
    def c(self):
        self.out_a == self.instr[0]
        self.out_b == self.instr[1]


def _make_cc():
    """Return a fresh ConstraintCompiler for _TwoOutput."""
    from zuspec.synth.sprtl.constraint_compiler import ConstraintCompiler
    cc = ConstraintCompiler(_TwoOutput)
    return cc


def _make_cc_ir():
    """Return a fully-extracted IR ready for the transform pipeline."""
    cc = _make_cc()
    cc.extract()
    return IR(payload=cc, kind="zuspec.constraint.compiler")


# ---------------------------------------------------------------------------
# Test SynthPlugin.register()
# ---------------------------------------------------------------------------

def test_synth_plugin_registers_command():
    reg = Registry()
    reg.reset()
    SynthPlugin().register(reg)
    assert reg.get_command("synth") is not None


def test_synth_plugin_registers_transforms():
    reg = Registry()
    reg.reset()
    SynthPlugin().register(reg)
    for name in ("compute-support", "build-cubes", "odc", "minimize"):
        assert reg.get_transform(name) is not None


# ---------------------------------------------------------------------------
# Test individual transforms
# ---------------------------------------------------------------------------

def test_compute_support_transform():
    ir = _make_cc_ir()
    xf = ComputeSupportTransform()
    out = xf.run(ir, argparse.Namespace())
    assert out is ir
    # After compute_support, the CC should have a support structure
    assert hasattr(ir.payload, "cset") and ir.payload.cset is not None


def test_build_cubes_transform():
    ir = _make_cc_ir()
    ComputeSupportTransform().run(ir, argparse.Namespace())
    out = BuildCubesTransform().run(ir, argparse.Namespace())
    assert out is ir
    # After build_cubes, _cubes_by_bit should be populated
    assert hasattr(ir.payload, "_cubes_by_bit") and ir.payload._cubes_by_bit is not None


def test_odc_transform():
    ir = _make_cc_ir()
    ComputeSupportTransform().run(ir, argparse.Namespace())
    BuildCubesTransform().run(ir, argparse.Namespace())
    out = ODCTransform().run(ir, argparse.Namespace())
    assert out is ir


def test_minimize_transform():
    ir = _make_cc_ir()
    ComputeSupportTransform().run(ir, argparse.Namespace())
    BuildCubesTransform().run(ir, argparse.Namespace())
    ODCTransform().run(ir, argparse.Namespace())
    out = MinimizeTransform().run(ir, argparse.Namespace())
    assert out is ir


# ---------------------------------------------------------------------------
# Test SynthCommand CLI argument parsing
# ---------------------------------------------------------------------------

def test_synth_command_parses_args():
    reg = Registry()
    reg.reset()
    SynthPlugin().register(reg)

    cmd = reg.get_command("synth")
    parent = argparse.ArgumentParser()
    subs = parent.add_subparsers(dest="command")
    cmd.add_subparser(subs)

    args = parent.parse_args(["synth", "--fe", "sv", "--top", "MyClass", "a.sv"])
    assert args.fe == "sv"
    assert args.top == "MyClass"
    assert args.files == ["a.sv"]
    assert args.be == "rtl-sv"


def test_synth_command_no_odc_flag():
    reg = Registry()
    reg.reset()
    SynthPlugin().register(reg)

    cmd = reg.get_command("synth")
    parent = argparse.ArgumentParser()
    subs = parent.add_subparsers(dest="command")
    cmd.add_subparser(subs)

    args = parent.parse_args(["synth", "--top", "X", "--no-odc"])
    assert args.no_odc is True
